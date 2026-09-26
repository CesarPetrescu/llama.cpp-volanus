# Reproducing the live control-vector tests

Every command below was run on the machine described in `REPORT.md` (CPU only). Paths assume the fork is at `$FORK`
and a scratch area at `$W`. All mechanism tests use temperature 0, thinking off, no speculative/MTP decoding and
`cache_prompt: false`.

```sh
export FORK=$PWD            # checkout of this branch
export W=$HOME/cvec-work    # models, venv, outputs
mkdir -p $W/models $W/runs $W/vectors/qwen3.5-2b $W/vectors/qwen3-1.7b
```

## 1. Build

```sh
cd $FORK
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release        # add -DGGML_CUDA=ON if nvidia-smi finds a GPU
cmake --build build -j$(nproc) --target llama-server llama-quantize llama-cli llama-cvec-live
```

## 2. Python env

```sh
python3 -m venv $W/venv && . $W/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CUDA build + flash-linear-attention + causal-conv1d on GPU
pip install "transformers>=5" accelerate safetensors huggingface_hub numpy scipy pyyaml requests
pip install -e $FORK/gguf-py
pip install -r $FORK/tools/server/tests/requirements.txt
```

## 3. Models (pinned revisions)

```sh
cd $W/models
hf download Qwen/Qwen3.5-2B      --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir Qwen3.5-2B
hf download Qwen/Qwen3-1.7B      --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir Qwen3-1.7B
hf download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --revision 90862c4b9d2787eaed51d12237eafdfe7c5f6077 --local-dir .

python $FORK/convert_hf_to_gguf.py Qwen3.5-2B --outtype bf16 --outfile qwen3.5-2b-bf16.gguf
$FORK/build/bin/llama-quantize qwen3.5-2b-bf16.gguf qwen3.5-2b-q4_k_m.gguf Q4_K_M
```

## 4. Track A: axes and vectors

```sh
cd $W
export MODEL=$W/models/Qwen3.5-2B OUT=$W/out THREADS=4
python $FORK/tools/cvec-live/emo_steer.py corpus
python $FORK/tools/cvec-live/emo_steer.py extract | tee extract.log    # probe table, chosen layer, out/axes.pt
python $FORK/tools/cvec-live/emo_steer.py steer   | tee steer.log      # HF dose sweep table, out/sweep_hf.json

python $FORK/tools/cvec-live/export_cvec.py --axes out/axes.pt --out-dir $W/vectors/qwen3.5-2b
python $FORK/tools/cvec-live/export_cvec.py --test-vectors --model $W/models/Qwen3.5-2B --model-hint qwen35 --out-dir $W/vectors/qwen3.5-2b
python $FORK/tools/cvec-live/export_cvec.py --test-vectors --model $W/models/Qwen3-1.7B  --model-hint qwen3  --out-dir $W/vectors/qwen3-1.7b
```

## 5. Track B: harness (T1-T4)

```sh
cd $W/runs
printf '<|im_start|>user\nTell me about your day in a few sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' > prompt_day.txt
H=$FORK/build/bin/llama-cvec-live; V=$W/vectors; M=$W/models

# T1 (zero) and T2 (random_d3), both models
for v in zero random_d3; do
  $H -m $M/qwen3.5-2b-q4_k_m.gguf --cvec $V/qwen3.5-2b/$v.gguf -f prompt_day.txt -n 64 -t 4 | tee q35_$v.log
  $H -m $M/Qwen3-1.7B-Q8_0.gguf   --cvec $V/qwen3-1.7b/$v.gguf -f prompt_day.txt -n 64 -t 4 | tee q3_$v.log
done

# T3 (turns none -> +v -> -v -> none) and T4 (switch at token 32), valence, both prompt modes
$H -m $M/qwen3.5-2b-q4_k_m.gguf --cvec $V/qwen3.5-2b/valence.gguf -f prompt_day.txt -n 64 -t 4               | tee q35_valence.log
$H -m $M/qwen3.5-2b-q4_k_m.gguf --cvec $V/qwen3.5-2b/valence.gguf -f prompt_day.txt -n 64 -t 4 --decode-only | tee q35_valence_do.log
```

Read `first_diff(...)` lines at the end of each log. `turn2` = +v, `turn3` = -v, `switch` = +v from token 32.

## 6. T5: HF <-> llama.cpp layer parity (BF16)

```sh
MODEL=$W/models/Qwen3.5-2B python $FORK/tools/cvec-live/parity.py --gguf $W/models/qwen3.5-2b-bf16.gguf \
  --axes $W/out/axes.pt --harness $FORK/build/bin/llama-cvec-live --prompt-file $W/runs/prompt_day.txt | tee t5.log
```

## 7. Server tests (T6, T7, T9)

On a CPU with Intel AMX, Qwen3.5 Q4_K_M gives wrong outputs when several sequences share a batch (upstream bug, see
`REPORT.md`). Disable weight repacking for every server run below: `export LLAMA_ARG_REPACK=false` (same as `--no-repack`).

```sh
export LLAMA_ARG_REPACK=false
cd $FORK/tools/server/tests
export LLAMA_SERVER_BIN_PATH=$FORK/build/bin/llama-server

# CI-sized run: tiny preset model + a random vector written by the test
python -m pytest -v unit/test_cvec.py

# acceptance run: Qwen3.5-2B Q4_K_M + valence
# note: $(cat ...) strips the trailing "\n\n" of the prompt, the "." keeps it
export CVEC_PROMPT="$(cat $W/runs/prompt_day.txt; echo .)"; export CVEC_PROMPT="${CVEC_PROMPT%.}"
CVEC_MODEL=$W/models/qwen3.5-2b-q4_k_m.gguf CVEC_DIR=$W/vectors/qwen3.5-2b CVEC_ID=valence \
python -m pytest -v -s unit/test_cvec.py | tee $W/runs/server_tests_q35.log
```

## 8. T8: server sweep

```sh
$FORK/build/bin/llama-server -m $W/models/qwen3.5-2b-q4_k_m.gguf --cvec-dir $W/vectors/qwen3.5-2b -np 4 -c 8192 -t 4 --port 8080 --no-repack &
MODEL=$W/models/Qwen3.5-2B THREADS=4 python $FORK/tools/cvec-live/cvec_sweep.py --url http://127.0.0.1:8080 --out $W/sweep_server_do.json | tee t8_do.log
MODEL=$W/models/Qwen3.5-2B THREADS=4 python $FORK/tools/cvec-live/cvec_sweep.py --url http://127.0.0.1:8080 --out $W/sweep_server_full.json --full-prompt | tee t8_full.log
# informational low-dose sweep
MODEL=$W/models/Qwen3.5-2B THREADS=4 python $FORK/tools/cvec-live/cvec_sweep.py --url http://127.0.0.1:8080 --out $W/sweep_server_low.json --doses -0.3 -0.2 -0.1 0 0.1 0.2 0.3 | tee t8_low.log
kill %1
```

Do not run the HF scripts and llama-server generation at the same time on a small CPU: both spin all cores and
decode slows down by more than 10x.

## 9. T10: throughput

```sh
LLAMA_ARG_REPACK=false python $FORK/tools/cvec-live/cvec_bench.py --server $FORK/build/bin/llama-server -m $W/models/qwen3.5-2b-q4_k_m.gguf \
  --cvec-dir $W/vectors/qwen3.5-2b -t 4 | tee t10.log
```

## 10. AMX multi-sequence bug (upstream, no steering involved)

Only on CPUs with AMX (check `grep -o amx_int8 /proc/cpuinfo`):

```sh
$FORK/build/bin/llama-server -m $W/models/qwen3.5-2b-q4_k_m.gguf -np 4 -c 8192 -t 4 --port 8080 &   # add --no-repack to compare
python - <<'PY'
import requests
from concurrent.futures import ThreadPoolExecutor
P = open("prompt_day.txt").read()
one = lambda _: requests.post("http://127.0.0.1:8080/completion", json={"prompt": P, "n_predict": 12, "temperature": 0, "cache_prompt": False}).json()["content"]
print("solo:", repr(one(0)))
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one, range(4)): print("par :", repr(r))
PY
```
