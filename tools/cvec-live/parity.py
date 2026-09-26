#!/usr/bin/env python3
# T5: HF <-> llama.cpp layer parity for a control vector.
# The same vector is added in HF at layer L (forward hook on layers[L]) and in llama.cpp at il = L-1, L, L+1
# (steered prompt, not decode-only). Compare log-probs at the first steered step (after the prompt) and greedy tokens.
#
#   MODEL=HF_DIR python parity.py --gguf model-bf16.gguf --axes out/axes.pt --harness build/bin/llama-cvec-live --prompt-file p.txt

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emo_steer import Steer, layers_of, load  # noqa: E402
from export_cvec import write_cvec  # noqa: E402


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def kl(lp, lq):
    return float((np.exp(lp) * (lp - lq)).sum())


def agree(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run_harness(args, cvec_path, out_prefix):
    cmd = [args.harness, "-m", args.gguf, "--cvec", str(cvec_path), "--scale", str(args.dose), "-f", args.prompt_file,
           "-n", str(args.n), "-t", str(args.threads), "--logits-out", str(out_prefix)]
    log = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    prompt = [int(t) for t in re.search(r"prompt tokens:([\d ]+)", log).group(1).split()]
    turns = [[int(t) for t in m.split()] for m in re.findall(r"  tokens:([\d ]+)", log)]
    logits = [np.fromfile(f"{out_prefix}.turn{i}.f32", dtype=np.float32) for i in (1, 2)]
    return prompt, turns[0], turns[1], logits[0], logits[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--axes", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    axes = torch.load(args.axes)
    L = int(axes["layer"])
    vec = (axes["hnorm"][L] * axes["W"][args.axis, L]).float()

    # HF reference in fp32 (bf16 weights are exact in fp32)
    tok, model = load(torch.float32)
    prompt_text = Path(args.prompt_file).read_text()
    ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids

    def hf_run(steer_vec):
        h = Steer(layers_of(model)[L], steer_vec) if steer_vec is not None else None
        with torch.no_grad():
            lp = log_softmax(model(ids).logits[0, -1].float().numpy())
            out = model.generate(ids, max_new_tokens=args.n, do_sample=False, eos_token_id=tok.eos_token_id)
        if h:
            h.remove()
        return lp, out[0, ids.shape[1]:].tolist()

    hf_lp0, hf_tok0 = hf_run(None)
    hf_lp1, hf_tok1 = hf_run(args.dose * vec)
    del model  # free RAM for llama.cpp

    tmp = Path(tempfile.mkdtemp())
    rows = []
    cpp_tok0 = None
    for off in (-1, 0, 1):
        il = L + off
        path = tmp / f"cvec_L{il}.gguf"
        write_cvec(path, il, vec.numpy(), axes.get("model_hint", ""))
        prompt, t0, t1, lg0, lg1 = run_harness(args, path, tmp / f"L{il}")
        if prompt != ids[0].tolist():
            raise SystemExit(f"prompt tokenization differs: HF {ids[0].tolist()} vs llama.cpp {prompt}")
        cpp_tok0 = t0
        cpp_lp0, cpp_lp1 = log_softmax(lg0), log_softmax(lg1)
        rows.append((il, kl(hf_lp1, cpp_lp1), f"{agree(hf_tok1, t1)} (HF {len(hf_tok1)}, cpp {len(t1)})", int(np.argmax(cpp_lp1)) == int(np.argmax(hf_lp1)), kl(hf_lp0, cpp_lp0)))

    print(f"T5 parity: axis {args.axis} at HF layer L = {L}, dose {args.dose}, |v| = {float(args.dose * vec.norm()):.3f}, prompt {ids.shape[1]} tokens")
    print(f"unsteered: KL(HF || cpp) = {rows[0][4]:.5f}, greedy agreement = {agree(hf_tok0, cpp_tok0)} (HF {len(hf_tok0)}, cpp {len(cpp_tok0)} tokens)")
    print("cpp il | KL(HF@L || cpp@il) | greedy agreement (lengths) | same argmax")
    for il, k, a, same, _ in rows:
        mark = " <- min KL" if k == min(r[1] for r in rows) else ""
        print(f"{il:6d} | {k:18.5f} | {a:26s} | {str(same):11s}{mark}")
    best = min(rows, key=lambda r: r[1])[0]
    print(f"result: {'PASS' if best == L else 'FAIL'} (min KL at il = {best}, expected {L})")
    print(f"HF steered text: {tok.decode(hf_tok1)!r}")


if __name__ == "__main__":
    os.environ.setdefault("THREADS", "4")
    main()
