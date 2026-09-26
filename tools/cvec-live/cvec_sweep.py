#!/usr/bin/env python3
# T8: emotion dose sweep through llama-server, scored with the same judge + perplexity as the HF sweep in emo_steer.py.
#
#   llama-server -m model.gguf --cvec-dir DIR -np 4 ...
#   MODEL=HF_DIR python cvec_sweep.py --url http://127.0.0.1:8080 --out sweep_server.json [--full-prompt]

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emo_steer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--id", default="valence")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-prompt", action="store_true", help="steer the prompt too (cvec_decode_only = false), like the HF hook")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    tok, model = emo_steer.load()

    def gen(job):
        dose, pi, s = job
        prompt = emo_steer.PROMPTS[pi]
        data = {
            "prompt": emo_steer.chat(tok, prompt),
            "n_predict": emo_steer.N_GEN,
            "temperature": 0.7, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
            "seed": 1000 * pi + s,
            "cache_prompt": False,
            "cvec_decode_only": not args.full_prompt,
        }
        if dose != 0.0:
            data["cvec"] = [{"id": args.id, "scale": dose}]
        r = requests.post(f"{args.url}/completion", json=data, timeout=600)
        r.raise_for_status()
        return {"dose": dose, "prompt": prompt, "sample": s, "text": r.json()["content"]}

    jobs = [(d, pi, s) for d in emo_steer.DOSES for pi in range(len(emo_steer.PROMPTS)) for s in range(emo_steer.N_SAMPLES)]
    with ThreadPoolExecutor(args.parallel) as ex:
        gens = list(ex.map(gen, jobs))

    mode = "prompt + generation" if args.full_prompt else "decode-only"
    title = (f"server sweep: '{args.id}' via llama-server ({mode}), {len(emo_steer.PROMPTS)} prompts x {emo_steer.N_SAMPLES} samples, "
             f"T = 0.7, {emo_steer.N_GEN} tokens")
    emo_steer.judge_and_ppl(tok, model, gens)
    summary = emo_steer.summarize(gens, title)
    Path(args.out).write_text(json.dumps({"title": title, "summary": summary, "gens": gens}, indent=1))


if __name__ == "__main__":
    main()
