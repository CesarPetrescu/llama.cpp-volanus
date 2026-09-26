#!/usr/bin/env python3
# Write llama.cpp control-vector GGUFs.
#
#   axes mode: export_cvec.py --axes out/axes.pt --out-dir DIR [--layer L]
#     -> DIR/valence.gguf, DIR/arousal.gguf with direction.<L> = hnorm[L] * W[k, L]
#   test mode: export_cvec.py --test-vectors --model HF_DIR --out-dir DIR [--layer L]
#     -> DIR/zero.gguf (zeros at L), DIR/random_d3.gguf (random unit vector * 3 * hnorm[L], L = middle layer)
#
# HF layer L (0-based, output of language_model.layers[L]) is llama.cpp il = L, i.e. tensor direction.<L>.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


def write_cvec(path: Path, layer: int, vec: np.ndarray, model_hint: str, hnorm=None):
    # hnorm: median hidden-state norm per layer (index = layer), stored as controlvector.hnorm.<il> for --cvec-max-total-dose
    if layer < 1:
        raise ValueError("llama.cpp cannot steer layer 0")
    w = gguf.GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    w.add_int32("controlvector.layer_count", 1)
    for il, h in enumerate(hnorm if hnorm is not None else []):
        if il >= 1 and h > 0:
            w.add_float32(f"controlvector.hnorm.{il}", float(h))
    w.add_tensor(f"direction.{layer}", vec.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}: direction.{layer}, |v| = {np.linalg.norm(vec):.3f}")


def measure_hnorm(model_dir: str) -> list[float]:
    # median token norm of each layer output over a few neutral chat prompts
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    inner = model.model
    layers = inner.language_model.layers if hasattr(inner, "language_model") else inner.layers
    outs = [None] * len(layers)

    # hook the layer outputs: hidden_states[-1] from HF is after the final norm
    def hook(il):
        def fn(mod, inp, out):
            outs[il] = out[0] if isinstance(out, tuple) else out
        return fn

    for il, layer in enumerate(layers):
        layer.register_forward_hook(hook(il))
    prompts = [
        "Describe the weather today in two sentences.",
        "What is a good way to spend a quiet afternoon?",
        "Summarize how a bicycle works.",
        "Write a short note to a coworker about a meeting.",
    ]
    norms = None
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)
        # skip token 0 (attention sink with outlier norm)
        cur = [h[0, 1:].float().norm(dim=-1) for h in outs]
        norms = cur if norms is None else [torch.cat([a, b]) for a, b in zip(norms, cur)]
    return [float(n.median()) for n in norms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", help="axes.pt from emo_steer.py extract")
    ap.add_argument("--test-vectors", action="store_true")
    ap.add_argument("--model", help="HF model dir (test mode)")
    ap.add_argument("--model-hint", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.axes:
        import torch
        axes = torch.load(args.axes, map_location="cpu")
        W = axes["W"].float().numpy()          # [2, n_layers, n_embd], unit rows
        hnorm = axes["hnorm"].float().numpy()  # [n_layers]
        layer = args.layer if args.layer is not None else int(axes["layer"])
        hint = args.model_hint or axes.get("model_hint", "")
        for k, name in enumerate(axes.get("axis_names", ["valence", "arousal"])):
            write_cvec(out / f"{name}.gguf", layer, hnorm[layer] * W[k, layer], hint, hnorm)
        return

    if args.test_vectors:
        cfg = json.loads((Path(args.model) / "config.json").read_text())
        n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
        n_embd = cfg.get("text_config", cfg)["hidden_size"]
        layer = args.layer if args.layer is not None else n_layers // 2
        hnorm = measure_hnorm(args.model)
        print("hnorm per layer:", " ".join(f"{h:.1f}" for h in hnorm))
        hint = args.model_hint or cfg.get("model_type", "")
        rng = np.random.default_rng(args.seed)
        d = rng.standard_normal(n_embd)
        d /= np.linalg.norm(d)
        write_cvec(out / "zero.gguf", layer, np.zeros(n_embd), hint, hnorm)
        write_cvec(out / "random_d3.gguf", layer, 3.0 * hnorm[layer] * d, hint, hnorm)
        (out / "hnorm.json").write_text(json.dumps({"layer": layer, "hnorm": hnorm}, indent=1))
        return

    ap.error("pass --axes or --test-vectors")


if __name__ == "__main__":
    main()
