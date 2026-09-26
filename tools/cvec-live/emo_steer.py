#!/usr/bin/env python3
# Stand-in for emo_steer.py (the original was not available): build valence/arousal axes and run a dose sweep in HF transformers.
#
#   MODEL=Qwen/Qwen3.5-2B OUT=out python emo_steer.py corpus   # templated corpus with circumplex labels -> out/corpus.json
#   MODEL=... OUT=out python emo_steer.py extract                # per-layer diff-of-means axes + held-out probe r -> out/axes.pt
#   MODEL=... OUT=out python emo_steer.py steer                  # dose sweep with a hook on layers[L], judged by the unsteered model
#   MODEL=... OUT=out python emo_steer.py score FILE.json        # judge + perplexity for generations made elsewhere (e.g. llama-server)
#
# axes.pt: W [2, n_layers, n_embd] unit directions (valence, arousal), hnorm [n_layers] median token norm of each layer output,
# layer = chosen layer. Steering adds dose * hnorm[L] * W[k, L] to the output of layers[L] at every position.

import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
OUT = Path(os.environ.get("OUT", "out"))

# approximate circumplex coordinates (valence, arousal) in [-1, 1]
EMOTIONS = {
    "ecstatic": (0.9, 0.9), "thrilled": (0.8, 0.8), "excited": (0.7, 0.7), "delighted": (0.8, 0.5),
    "happy": (0.8, 0.3), "grateful": (0.7, 0.0), "content": (0.6, -0.4), "relaxed": (0.5, -0.7),
    "calm": (0.3, -0.8), "sleepy": (0.0, -0.9), "bored": (-0.4, -0.7), "tired": (-0.3, -0.8),
    "gloomy": (-0.6, -0.5), "sad": (-0.7, -0.4), "miserable": (-0.9, -0.3), "lonely": (-0.7, -0.3),
    "annoyed": (-0.4, 0.4), "frustrated": (-0.6, 0.5), "stressed": (-0.5, 0.7), "anxious": (-0.6, 0.7),
    "furious": (-0.8, 0.9), "terrified": (-0.8, 0.9),
}

SITUATIONS = [
    "my train was ten minutes late this morning", "I finished the quarterly report", "my sister called me after dinner",
    "the meeting got moved to Thursday", "I walked home through the park", "my landlord replied to my email",
    "I cooked pasta for the first time in weeks", "the power went out for an hour", "I got my exam results back",
    "a stranger held the door for me", "my phone screen cracked", "I cleaned the whole apartment",
    "the neighbors were painting their fence", "I read the news on my lunch break", "my flight was rebooked for tomorrow",
    "I tried the new coffee place downtown", "my manager asked to talk to me", "it rained all afternoon",
    "I found an old photo of my grandparents", "the package I ordered finally arrived", "I missed the bus by a few seconds",
    "my friend cancelled our plans", "I signed up for a pottery class", "the doctor called with my test results",
]

TEMPLATES = [
    "Today {sit}, and I feel {emo}.",
    "I am {emo} right now. {Sit}.",
]

PROMPTS = [
    "Tell me about your day in a few sentences.",
    "Describe what you see when you look out the window.",
    "Write a short note to a friend you have not seen in a while.",
    "How do you feel about Mondays?",
    "Describe your morning commute.",
    "Write two sentences about the weather today.",
    "What are you thinking about right now?",
    "Tell me about a meal you had recently.",
]

DOSES = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]
N_SAMPLES = 2
N_GEN = 48

JUDGE = ("Rate the emotional valence of the following text on a scale from 1 (very negative) to 9 (very positive). "
         "Answer with a single digit only.\n\nText: \"\"\"{text}\"\"\"")


def load(dtype=torch.bfloat16):
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    return tok, model


def layers_of(model):
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


class Steer:
    # adds vec to the output hidden states of one decoder layer
    def __init__(self, layer_mod, vec):
        self.vec = vec
        self.h = layer_mod.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
        return out + self.vec.to(out.dtype)

    def remove(self):
        self.h.remove()


def cmd_corpus():
    rows = []
    for si, sit in enumerate(SITUATIONS):
        for emo, (v, a) in EMOTIONS.items():
            for t in TEMPLATES:
                rows.append({"text": t.format(sit=sit, Sit=sit[0].upper() + sit[1:], emo=emo), "valence": v, "arousal": a, "sit": si, "emo": emo})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus.json").write_text(json.dumps(rows, indent=0))
    print(f"corpus: {len(rows)} texts ({len(SITUATIONS)} situations x {len(EMOTIONS)} emotions x {len(TEMPLATES)} templates) -> {OUT / 'corpus.json'}")


def cmd_extract():
    rows = json.loads((OUT / "corpus.json").read_text())
    tok, model = load()
    n_layers = len(layers_of(model))
    tok.padding_side = "right"

    pooled = []                            # per batch: [n_layers, B, d]
    norms = [[] for _ in range(n_layers)]  # token norms per layer
    bs = 32
    for i in range(0, len(rows), bs):
        enc = tok([r["text"] for r in rows[i:i + bs]], return_tensors="pt", padding=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        # causal model + right padding: real tokens are not affected by pads; skip token 0 (attention sink)
        m = enc.attention_mask.clone().float()
        m[:, 0] = 0
        pooled.append(torch.stack([(h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True) for h in hs]))
        for L, h in enumerate(hs):
            norms[L].append(h.float().norm(dim=-1)[m.bool()])
        print(f"\rextract: {min(i + bs, len(rows))}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    H = torch.cat(pooled, dim=1)  # [n_layers, N, d]
    hnorm = torch.tensor([torch.cat(n).median().item() for n in norms])
    y = torch.tensor([[r["valence"], r["arousal"]] for r in rows])
    # hold out every 4th emotion word, so the probe must generalize to unseen words
    held = set(list(EMOTIONS)[::4])
    test = torch.tensor([r["emo"] in held for r in rows])
    train = ~test

    W = torch.zeros(2, n_layers, H.shape[-1])
    R = torch.zeros(2, n_layers)
    for L in range(n_layers):
        for k in range(2):
            pos = train & (y[:, k] > 0)
            neg = train & (y[:, k] < 0)
            d = H[L, pos].mean(0) - H[L, neg].mean(0)
            W[k, L] = d / d.norm()
            proj = H[L, test] @ W[k, L]
            R[k, L] = torch.corrcoef(torch.stack([proj, y[test, k]]))[0, 1]

    lo, hi = n_layers // 4, 3 * n_layers // 4
    layer = lo + int(R[0, lo:hi].argmax())

    print(f"\nheld-out probe (n_train = {int(train.sum())}, n_test = {int(test.sum())}, held-out emotion words: {', '.join(sorted(held))})")
    print("layer | r_valence | r_arousal | cos(val,aro) | hnorm")
    for L in range(n_layers):
        mark = " <- chosen" if L == layer else ""
        print(f"{L:5d} | {R[0, L]:9.3f} | {R[1, L]:9.3f} | {float(W[0, L] @ W[1, L]):12.3f} | {hnorm[L]:6.2f}{mark}")
    print(f"\nchosen layer L = {layer} (best held-out valence r in layers [{lo}, {hi}))")

    hint = {"qwen3_5": "qwen35", "qwen3_5_text": "qwen35"}.get(model.config.model_type, model.config.model_type)
    torch.save({"W": W, "hnorm": hnorm, "layer": layer, "probe_r": R, "axis_names": ["valence", "arousal"],
                "model": MODEL, "model_hint": hint}, OUT / "axes.pt")
    print(f"saved {OUT / 'axes.pt'}")


def judge_and_ppl(tok, model, gens):
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 10)]
    for g in gens:
        ids = tok(chat(tok, JUDGE.format(text=g["text"].strip())), return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        p = torch.softmax(logits[digits], dim=0)
        g["judge"] = float((p * torch.arange(1, 10)).sum())

        p_ids = tok(chat(tok, g["prompt"]), return_tensors="pt").input_ids
        g_ids = tok(g["text"], return_tensors="pt", add_special_tokens=False).input_ids
        full = torch.cat([p_ids, g_ids], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(model(full).logits[0, :-1].float(), dim=-1)
        tgt = full[0, 1:]
        nll = -lp[torch.arange(len(tgt)), tgt][p_ids.shape[1] - 1:]
        g["nll_sum"] = float(nll.sum())
        g["n_tok"] = int(nll.numel())
    return gens


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def summarize(gens, title):
    print(f"\n{title}")
    print("dose  | judged valence (mean +- sd) | ppl    | ppl ratio | sample")
    by = {}
    for g in gens:
        by.setdefault(g["dose"], []).append(g)
    ppl0 = None
    means = []
    for dose in sorted(by):
        gs = by[dose]
        j = torch.tensor([g["judge"] for g in gs])
        ppl = math.exp(sum(g["nll_sum"] for g in gs) / max(1, sum(g["n_tok"] for g in gs)))
        if dose == 0.0:
            ppl0 = ppl
        means.append((dose, float(j.mean()), ppl))
    for dose, jm, ppl in means:
        gs = by[dose]
        sd = float(torch.tensor([g["judge"] for g in gs]).std())
        ratio = ppl / ppl0 if ppl0 else float("nan")
        sample = gs[0]["text"].strip().replace("\n", " ")[:70]
        print(f"{dose:+5.1f} | {jm:6.2f} +- {sd:4.2f}               | {ppl:6.2f} | {ratio:9.2f} | {sample}")
    rho_all = spearman([g["dose"] for g in gens], [g["judge"] for g in gens])
    rho_mean = spearman([m[0] for m in means], [m[1] for m in means])
    print(f"spearman(dose, judged valence): per-sample = {rho_all:.3f}, dose means = {rho_mean:.3f}")
    return {"rows": [{"dose": d, "judge_mean": jm, "ppl": p, "ppl_ratio": p / ppl0} for d, jm, p in means],
            "spearman_samples": rho_all, "spearman_means": rho_mean}


def cmd_steer():
    axes = torch.load(OUT / "axes.pt")
    L = int(axes["layer"])
    vec = axes["hnorm"][L] * axes["W"][0, L]
    tok, model = load()
    layer_mod = layers_of(model)[L]

    gens = []
    for dose in DOSES:
        steer = Steer(layer_mod, dose * vec) if dose != 0.0 else None
        for pi, prompt in enumerate(PROMPTS):
            ids = tok(chat(tok, prompt), return_tensors="pt").input_ids
            for s in range(N_SAMPLES):
                torch.manual_seed(1000 * pi + s)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=N_GEN, do_sample=True, temperature=0.7, top_p=1.0, top_k=0)
                text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                gens.append({"dose": dose, "prompt": prompt, "sample": s, "text": text})
                print(f"\rsteer: dose {dose:+.1f} prompt {pi} sample {s}", end="", file=sys.stderr)
        if steer:
            steer.remove()
    print(file=sys.stderr)

    judge_and_ppl(tok, model, gens)
    res = summarize(gens, f"HF sweep: valence axis at layer {L}, {len(PROMPTS)} prompts x {N_SAMPLES} samples, T = 0.7, {N_GEN} tokens")
    (OUT / "sweep_hf.json").write_text(json.dumps({"summary": res, "gens": gens}, indent=1))


def cmd_score(path):
    data = json.loads(Path(path).read_text())
    tok, model = load()
    gens = judge_and_ppl(tok, model, data["gens"])
    data["summary"] = summarize(gens, data.get("title", path))
    Path(path).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "corpus":
        cmd_corpus()
    elif cmd == "extract":
        cmd_extract()
    elif cmd == "steer":
        cmd_steer()
    elif cmd == "score" and len(sys.argv) > 2:
        cmd_score(sys.argv[2])
    else:
        print("usage: emo_steer.py corpus|extract|steer|score FILE")
        sys.exit(1)
