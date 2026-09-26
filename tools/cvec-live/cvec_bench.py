#!/usr/bin/env python3
# T10: generation throughput of llama-server with and without per-request control vectors (report only).
#
#   python cvec_bench.py --server build/bin/llama-server -m model.gguf --cvec-dir DIR [--id valence] [-t 4]

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPT = "<|im_start|>user\nTell me about your day in a few sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def start(args, n_parallel):
    port = 8099
    cmd = [args.server, "-m", args.model, "--cvec-dir", args.cvec_dir, "-np", str(n_parallel), "-c", str(2048 * n_parallel),
           "-t", str(args.threads), "--port", str(port), "--no-warmup"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                return proc, url
        except requests.RequestException:
            pass
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("server did not start")


def run(url, scales, n_predict, cvec_id):
    def one(scale):
        data = {"prompt": PROMPT, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
        if scale is not None:
            data["cvec"] = [{"id": cvec_id, "scale": scale}]
        r = requests.post(f"{url}/completion", json=data, timeout=600)
        r.raise_for_status()
        return r.json()["timings"]

    t0 = time.time()
    with ThreadPoolExecutor(len(scales)) as ex:
        timings = list(ex.map(one, scales))
    wall = time.time() - t0
    n_gen = sum(t["predicted_n"] for t in timings)
    per_req = sum(t["predicted_per_second"] for t in timings) / len(timings)
    return n_gen / wall, per_req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--cvec-dir", required=True)
    ap.add_argument("--id", default="valence")
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    configs = {
        1: [("none", [None]), ("steered", [1.0])],
        4: [("all none", [None] * 4), ("all same vector", [1.0] * 4), ("all different vectors", [1.0, -1.0, 0.5, -0.5])],
    }
    print(f"T10 throughput: n_predict = {args.n}, ignore_eos, decode-only steering, {args.repeats} repeats (median)")
    print("np | config                | aggregate tok/s | mean per-request tok/s")
    for n_parallel, cases in configs.items():
        proc, url = start(args, n_parallel)
        try:
            run(url, [None], 8, args.id)  # warm up
            for name, scales in cases:
                res = sorted(run(url, scales, args.n, args.id) for _ in range(args.repeats))
                agg, per = res[len(res) // 2]
                print(f"{n_parallel:2d} | {name:21s} | {agg:15.2f} | {per:22.2f}")
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
