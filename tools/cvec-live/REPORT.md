# Live control-vector steering in llama.cpp: Qwen3.5-2B test + per-request server fork

Date: 2026-09-26. Commands for every number below are in `REPRO.md`.

## Summary

- **The Qwen3.5 hybrid graph honors control vectors, and they can be switched live.** A zero vector is token-identical
  to no vector, a random vector diverges at token 1 on both Qwen3.5-2B and Qwen3-1.7B, and none -> +v -> -v -> none in
  one process returns to the exact baseline. `llama_set_adapter_cvec` costs 0.03-0.2 ms, with no reload.
- **Layer mapping is confirmed:** HF `layers[L]` output == llama.cpp `il = L` == tensor `direction.<L>` (buffer slot
  `L-1`). At L = 8, KL(HF || llama.cpp) is 0.00006 at il = 8 vs 0.64 / 2.58 at il = 7 / 9, and the steered greedy output
  matches HF token for token.
- **The server fork works:** `--cvec-dir`, a per-request `cvec` field on native and OpenAI routes, `cvec_decode_only`,
  `GET /cvecs`, and co-batching of equal vectors only. On Qwen3.5-2B, 20 of 21 server tests pass (T7, T9, prompt-cache
  rules, OpenAI route). T6 meets every identity/inequality check, but fails "stable over 3 repeats" once: one of 24 outputs
  flipped because of batch-composition numerics, not steering (evidence below).
- **Found and root-caused an upstream bug, not caused by steering:** on CPUs with Intel AMX, Qwen3.5 Q4_K_M
  gives garbage for every sequence after the first when several sequences share a ubatch (stock `master` too). The cause
  is two offset bugs in the AMX matmul; a fix was verified in a scratch build and is not on the branch. Server tests
  run with `--no-repack` (see "Upstream bug" below).
- **Emotion quality (T8) fails with the stand-in axes:** judged valence only moves monotonically for |dose| <= 0.4, and
  perplexity doubles by then. Through the server (decode-only): Spearman of dose means 0.857 (< 0.9), perplexity
  ratio 4.3-5.2 at \|dose\| = 0.8. At \|dose\| <= 0.3 fluency holds (ratio <= 1.22), but the effect is small
  (5.05 -> 5.90 on a 1-9 scale). Held-out probe r = 0.701, just above the 0.7 cutoff, so T8 counts, and it fails.

## Environment

| Item | Value |
| --- | --- |
| CPU | Intel Xeon @ 2.10 GHz, 4 vCPU, Sapphire Rapids class (AVX-512, AVX512_BF16, AMX-BF16/INT8) |
| RAM / GPU | 15 GiB, no swap / none (`nvidia-smi` absent) -> CPU build |
| OS / toolchain | Ubuntu 24.04.4, kernel 6.18.44, gcc 13.3.0, cmake 3.28.3, ninja, Python 3.11.15 |
| llama.cpp base | `master` @ `2145525a4081d66ff1a87cf43ef809f95a85ac0c` (2026-09-26) |
| Fork branch | `claude/ecstatic-hamilton-b0jqyw` on `CesarPetrescu/llama.cpp-volanus` |
| Python packages | torch 2.14.0+cpu, transformers 5.17.0, accelerate 1.15.0, numpy 2.4.6, scipy 1.17.1, huggingface_hub 1.33.0, gguf (fork `gguf-py`, 0.19.0). No flash-linear-attention / causal-conv1d (no GPU): HF used its reference DeltaNet kernels |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` @ `15852e8c`, converted with the fork's `convert_hf_to_gguf.py` to BF16 (3.9 GB) + `llama-quantize` Q4_K_M |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B-GGUF` Q8_0 @ `90862c4b`; HF `Qwen/Qwen3-1.7B` @ `70d244cc` only to measure hnorm |

Settings for all mechanism tests: temperature 0, thinking off (prompt ends with `<think>\n\n</think>\n\n`, as
`enable_thinking=False` renders it), no speculative/MTP decoding (MTP tensors are skipped at load), `cache_prompt: false`.
Prompt: `Tell me about your day in a few sentences.` (22 tokens).

## Results T1-T10

| ID | Result | Evidence |
| --- | --- | --- |
| T1 zero vs none | **PASS** | Qwen3.5-2B Q4_K_M: `first_diff(turn1, turn2) = none (identical, n=64)`. Qwen3-1.7B Q8_0: `identical, n=37` (EOS at 37 in both) |
| T2 random_d3 at mid layer | **PASS** | Qwen3.5-2B (il 12, \|v\| = 18.6 = 3 x hnorm 6.21): diverges at token 1 (`4277 15 15 15 ...` vs `2053 449 ...`). Qwen3-1.7B (il 14, \|v\| = 557 = 3 x 185.7): diverges at token 1 (`220 220 ...`). The graph gap feared in the handoff does not exist |
| T3 none -> +v -> -v -> none | **PASS** | valence x 1.0, one load. turn4 vs turn1 identical (64/64). turn2 != turn1 (first diff 3), turn3 != turn1 (1), turn2 != turn3 (1). Decode-only variant: 3 / 3 / 10, turn4 identical. set calls: 0.146, 0.086, 0.030, 0.084 ms (decode-only run: 0.181, 0.041, 0.066, 0.034, 0.080 ms) |
| T4 switch on at token 32 | **PASS** | `first_diff(turn1, switch) = 35`: tokens 1-34 identical (33 and 34 happen to match), first steered token is 33. random_d3 on Qwen3-1.7B: first diff at exactly 33 |
| T5 HF <-> llama.cpp parity | **PASS** | BF16, valence x 1.0 at L = 8. KL(HF@8 \|\| cpp@7) = 0.64395, **cpp@8 = 0.00006**, cpp@9 = 2.58382. Greedy agreement at il 8: 15/15 (both stop at `<|im_end|>`, "As an AI, I am eager to contribute to the ongoing dialogue."), il 7 / 9: 0. Unsteered reference: KL 0.00001, 64/64 |
| T6 -np 4 concurrent +1 / none / -1 / +1 | **FAIL (stability only)** | Qwen3.5-2B Q4_K_M, valence, 32 tokens, `--no-repack`. Steered-prompt mode: all conditions hold in 3/3 repeats (+1 = +1 = "As an AI, I am eager to provide the answer.", none = solo, -1 starts with `<think>`). Decode-only mode: none = solo, +1 = +1, +1 != -1 != none in 3/3 repeats (+1 "...eager to assist with your inquiries...", -1 "I feel lonely and worthless..."), but -1 in repeat 2 differs from repeats 0-1 at token 10. Log: that request was launched 0.42 s after the other three, so its prefill ran in a different batch. Check: 4 x -1 co-batched give 4 identical outputs equal to repeats 0-1, and a solo -1 gives a third variant (differs at token 11). llama.cpp is not batch-invariant, and this trajectory has a near-tie. Not a steering bug |
| T7 decode-only first token | **PASS** | Doses -1.2 ... +1.2 (7), `cvec_decode_only: true`: first token `40` ("I") for every dose, then outputs differ (e.g. -1.2 `40 1044 8008 38257 ...`, 0 `40 1459 914 599 ...`, +1.2 `40 1044 5354 310 ...`) |
| T8 emotion sweep | **FAIL** | Server decode-only: Spearman(dose means) = 0.857, per-sample 0.445. Perplexity ratio at \|dose\| <= 0.8: 1.68 / 1.35 at -0.4 / +0.4, 4.34 / 5.21 at -0.8 / +0.8. HF sweep: Spearman 0.643, perplexity ratio 8.2-11.8 at \|dose\| = 0.8. Probe r = 0.701 (>= 0.7, so not informational). Tables below |
| T9 bad input | **PASS** | All 400 with a message, on `/completion` and `/v1/completions`: unknown id (`unknown control vector id 'missing'`), `scale` null / `"NaN"` / missing (`each entry must have a string 'id' and a number 'scale'`), 3.5 / -3.5 / 1e39 (`scale of 'valence' must be finite and in [-3, 3]`), duplicate id, non-array, bare `NaN` / `Infinity` (JSON `parse error`), `1e999` (`number overflow`), `cvec` without `--cvec-dir` (`control vectors are disabled, start the server with --cvec-dir`). Startup refused on n_embd 2049 vs 2048 and on `direction.10000` |
| T10 throughput | reported | See table below. `-np 1`: steering costs nothing. `-np 4`: different vectors collapse to single-slot throughput |

Raw logs for T1-T5 were produced by `llama-cvec-live` and `parity.py` (see `REPRO.md`); the tables above quote them.

## Track A: axes (stand-in `emo_steer.py`)

`emo_steer.py` was not attached to the task and was not found on disk. After asking, I wrote a stand-in with the
interface the handoff implies (`MODEL` env; `corpus`, `extract`, `steer`; `out/axes.pt` with `W[k, L]` unit directions
and `hnorm[L]`), plus a `score` command so HF and server sweeps share one judge. `export_cvec.py` reads that
`axes.pt` format; if the real script stores different keys, only the 4 lines that read `axes.pt` need to change.

- Corpus: 22 emotion words with approximate circumplex (valence, arousal) labels x 24 neutral situations x 2 templates
  = 1056 short first-person texts.
- Axes: per layer, mean-pooled layer outputs (token 0 skipped: attention-sink norm outlier), direction = mean(label > 0)
  - mean(label < 0), unit-normalized. `hnorm[L]` = median token norm of the layer output over the corpus.
- Held-out probe: 6 emotion words (annoyed, calm, ecstatic, furious, gloomy, happy) are held out, so the probe must
  generalize to unseen words. A first run that held out situations instead gave r = 0.88 at layer 0 and just decayed with
  depth, which is lexical matching, not an axis; I switched the split.
- Layer choice: best held-out valence r in the middle band [n/4, 3n/4) = [6, 18) -> **L = 8**, r_valence = 0.701,
  r_arousal = 0.666, hnorm = 5.17. `valence.gguf` and `arousal.gguf` = `direction.8`, \|v\| = 5.167 each.

Probe table (Qwen3.5-2B, 768 train / 288 test):

| layer | r_valence | r_arousal | cos(val, aro) | hnorm |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.509 | 0.776 | 0.351 | 2.17 |
| 2 | 0.483 | 0.755 | 0.407 | 3.61 |
| 4 | 0.625 | 0.723 | 0.384 | 4.17 |
| 5 | 0.676 | 0.693 | 0.339 | 4.57 |
| 6 | 0.668 | 0.669 | 0.355 | 5.49 |
| 7 | 0.648 | 0.640 | 0.389 | 5.15 |
| **8** | **0.701** | **0.666** | **0.370** | **5.17** |
| 9 | 0.690 | 0.664 | 0.357 | 5.40 |
| 10 | 0.675 | 0.656 | 0.317 | 6.07 |
| 12 | 0.683 | 0.634 | 0.313 | 6.71 |
| 14 | 0.678 | 0.632 | 0.328 | 10.17 |
| 16 | 0.638 | 0.561 | 0.300 | 14.83 |
| 18 | 0.617 | 0.542 | 0.350 | 20.77 |
| 20 | 0.602 | 0.524 | 0.367 | 25.86 |
| 22 | 0.574 | 0.514 | 0.382 | 31.80 |
| 23 | 0.580 | 0.509 | 0.376 | 157.75 (after final norm) |

(Full 24-row table is printed by `emo_steer.py extract`.)

## Emotion sweeps: HF vs server

Doses are relative: the added vector is `dose x hnorm[8] x W[valence, 8]`. 8 prompts x 2 samples, temperature 0.7,
top-k/top-p/min-p off, 48 new tokens. Judge: the unsteered HF model (BF16) rates each text 1-9, expected value over the
digit log-probs. Perplexity: unsteered HF model on the generated tokens given the prompt; ratio vs dose 0.

Judged valence (mean over 16 texts) and perplexity ratio vs dose 0:

| dose | HF (hook on all positions) | server, steered prompt | server, decode-only (default) |
| ---: | --- | --- | --- |
| -1.2 | 5.34, ppl x8.22 | 5.42, ppl x7.14 | 4.84, ppl x6.53 |
| -0.8 | 4.32, ppl x8.61 | 4.64, ppl x9.40 | 4.56, ppl x4.34 |
| -0.4 | 4.69, ppl x2.29 | 4.80, ppl x2.19 | 4.96, ppl x1.68 |
| 0 | 5.31, ppl 2.11 | 5.61, ppl 2.39 | 5.68, ppl 2.33 |
| +0.4 | 6.64, ppl x2.12 | 6.60, ppl x2.23 | 6.11, ppl x1.35 |
| +0.8 | 6.56, ppl x11.80 | 6.81, ppl x9.31 | 6.38, ppl x5.21 |
| +1.2 | 5.47, ppl x31.62 | 4.65, ppl x29.86 | 6.02, ppl x13.65 |
| Spearman (dose means / per sample) | 0.643 / 0.347 | 0.321 / 0.224 | 0.857 / 0.445 |

Informational low-dose sweep, server decode-only:

| dose | -0.3 | -0.2 | -0.1 | 0 | +0.1 | +0.2 | +0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| judged valence | 5.05 | 5.52 | 5.57 | 5.84 | 5.72 | 5.77 | 5.90 |
| ppl ratio | 1.12 | 1.09 | 1.01 | 1.00 | 0.97 | 1.02 | 1.22 |

Spearman(dose means) = 0.893.

Reading: the HF and steered-prompt server columns agree within sampling noise, as expected from T5. Both collapse
beyond \|dose\| 0.4-0.8 (degenerate text, which the judge scores near 5). Decode-only steering is gentler (lower
perplexity at the same dose) and stays monotonic from -0.8 to +0.8. Example texts: -1.2 decode-only "I don't have a
day, so I feel isolated and lonely", +0.4 HF "My day begins with a detailed examination of your query...", +1.2 HF
"To participate this is my ongoing task. boxed box box box".

## Latency and throughput

- `llama_set_adapter_cvec`: 0.03-0.2 ms per call (copy of `n_embd x n_layer` floats into preallocated tensors), measured
  around every call in the harness.
- Hidden cost: every call sets `sched_need_reserve`, so the **next decode rebuilds the scheduler and reserves graphs**. In the
  harness the first single-token decode after a switch took 85-107 ms vs a 74-82 ms median (Qwen3.5-2B Q4_K_M, 4 threads).
  This is paid on every batch whose vector differs from the previous one, see T10.

T10 (`cvec_bench.py`, Qwen3.5-2B Q4_K_M, 4 threads, `--no-repack`, 64 tokens with `ignore_eos`, decode-only
steering, median of 3):

| -np | config | aggregate tok/s | mean per-request tok/s |
| ---: | --- | ---: | ---: |
| 1 | none | 11.88 | 12.87 |
| 1 | steered (valence 1.0) | 12.14 | 12.99 |
| 4 | all none | 24.70 | 6.61 |
| 4 | all same vector | 18.22 | 9.72 |
| 4 | all different vectors (1.0, -1.0, 0.5, -0.5) | 12.72 | 6.78 |

- Different vectors: one decode call per slot per step, so throughput falls to the single-slot rate (12.7 vs 24.7).
  The per-call switch itself is cheap on CPU (~79 ms per call vs ~84 ms per token at `-np 1`).
- Same vector, decode-only, is also slower (18.2). Follow-up run: same vector with `cvec_decode_only: false` gives 25.6
  tok/s (= no steering), decode-only 19.0 (one run reached 25.55). Cause: a request that arrives while others are already
  generating has an unsteered prefill, which cannot share a decode call with steered generation, so batches alternate.
  This is inherent to one context-wide vector.

## Server fork: what changed

Commits on the branch (base `2145525a`):

1. `cvec-live : add harness ...`: `tools/cvec-live/cvec-live.cpp` (`llama-cvec-live`), flags `--cvec`, `--scale`,
   `--decode-only`, `--switch-at`, `--logits-out`.
2. `server : per-request control vectors`:
   - `common/arg.cpp`, `common/common.h`: `--cvec-dir DIR`, `--cvec-max-scale X` (default 3.0).
   - `server-schema.cpp`: request fields `cvec: [{id, scale}]` and `cvec_decode_only` (default true), shared by
     `/completion`, `/v1/completions` and `/v1/chat/completions` (extra body fields). Validation: array of objects,
     string id, numeric finite scale, \|scale\| <= max, no duplicate ids, `--cvec-dir` must be set. Zero scales are dropped
     so equal vectors have equal keys. `cvec_decode_only: false` forces `cache_prompt: false`.
   - `server-context.cpp`: vectors loaded once at startup with `common_control_vector_load` (strength 1), server refuses
     to start on n_embd mismatch, on a direction past the last layer, or on an empty dir. `server_slot::cvec_active()`
     is the effective key (empty while a decode-only task has not reached `SLOT_STATE_GENERATING`), `can_batch_with()`
     also compares it, and right after `common_set_adapter_lora()` the combined buffer (global baseline + sum of
     scale x direction) is set only if the key changed, `NULL` when both are empty; debug log per apply. Unknown ids are
     rejected at slot launch (400). A slot whose task steered its prompt clears its prompt (KV + checkpoints) on release.
     `GET /cvecs` returns `[{id, layers, n_embd}]`.
   - `server.cpp`: route `/cvecs`; `common_json_error` (malformed JSON, e.g. a bare `NaN` or `1e999`) now maps to 400
     instead of 500 for all routes.
3. `server : add tests ...`: `tools/server/tests/unit/test_cvec.py` + `cvec_dir` / `cvec_max_scale` in `utils.py`.
4. `cvec-live : add scripts ...`: `emo_steer.py` (stand-in), `export_cvec.py`, `parity.py`, `cvec_sweep.py` (T8),
   `cvec_bench.py` (T10).

Implementation notes that matter for production:

- `llama_adapter_cvec::apply()` only writes layers covered by `len`; layers past the end of a shorter buffer **keep their
  old values** while still inside `[il_start, il_end]`. Both the harness and the server always pass a full
  `n_embd x n_layer` buffer for this reason. Anyone calling the API directly with per-file buffers of different lengths
  will get stale steering.
- The global baseline (`--control-vector*`) is re-read, zeroed outside its own layer range, and summed with request vectors,
  and the combined vector is set with range `[1, n_layer)`. So request vectors can use any layer, and the global range is
  still honored.
- Decode-only caveat (documented, not fixed, per the spec): with `cache_prompt: true`, a later request can reuse the KV of
  tokens that an earlier request *generated* under steering (multi-turn history). The prompt part itself was never steered.

## Upstream bug found: AMX + multi-sequence ubatch (not steering)

With 4 identical concurrent requests and **no control vectors at all**, stock `master` (`2145525a`, separate build)
returns 3 garbage outputs out of 4 on Qwen3.5-2B Q4_K_M:

```
solo: "As an artificial intelligence, I don't have a physical day"
par : '眺望远chenesisesisesisesisRENesisRENesisREN'
par : '眺chengezči路交叉口pren品的chengezči路交叉口pren'
par : '眺chengezči路交叉口pren品的chengezči路交叉口pren'
par : "I don't have a physical body or a personal day,"
```

The same load is correct on Qwen3.5-2B BF16, on Qwen3-1.7B Q8_0, and with a single 118-token prompt on Q4_K_M. It is also
correct on Q4_K_M with `--no-repack`, where all four outputs equal the solo output. The weights sit in the `AMX` buffer
(1218 MiB). Only hybrid/recurrent graphs trigger it: they shape activations as `[n_embd, n_seq_tokens, n_seqs]`, so the
matmul gets `ne2 = n_seqs > 1`, and `ggml_backend_amx_mul_mat()` (`ggml/src/ggml-cpu/amx/mmq.cpp`) mishandles batches:

1. `dst_offset` is a byte offset (`ggml_batch_offset` returns `i3 * nb[3] + i2 * nb[2]`) but is added to a `float *`
   (`(float *) dst->data + dst_offset`), so batch k >= 1 writes 4x too far.
2. `src0_offset = ggml_batch_offset(src0, batch_idx, ne2)` does not broadcast the 2-D weight (`src0->ne[2] == 1`): batch
   k >= 1 reads `k * nb[2]` bytes past the weight.

Fixing both (diff below) in a scratch build of stock `master` makes the 4 parallel outputs sane again. They are now
either the solo text or the `--no-repack` text, i.e. normal batch-shape numerics. The F16 AVX path in the same file
(`LAUNCH_TINYGEMM_KERNEL_AVX`) has the same typed-pointer pattern (not tested). Per the guardrails this is **not** on the
branch: it is outside the steering scope and T2 passed. Irrelevant for the GPU target, but any AMX CPU deployment of Qwen3.5
with `-np > 1` is affected. The single-sequence harness runs (T1-T4) use AMX and are unaffected. T6-T10 use
`--no-repack` (`LLAMA_ARG_REPACK=false`).

```diff
--- a/ggml/src/ggml-cpu/amx/mmq.cpp
+++ b/ggml/src/ggml-cpu/amx/mmq.cpp
@@ LAUNCH_TINYGEMM_KERNEL_VNNI
-        (float *) dst->data + dst_offset + nb_start, ldc)
+        (float *) ((char *) dst->data + dst_offset) + nb_start, ldc)
@@ ggml_backend_amx_mul_mat, 3 call sites
-                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
+                    int64_t src0_offset = ggml_batch_offset(src0, (batch_idx % ne2) / (ne2 / src0->ne[2]) + ((batch_idx / ne2) / (dst->ne[3] / src0->ne[3])) * src0->ne[2], src0->ne[2]);
@@ tinygemm_kernel_amx call
-                    (float *) dst->data + dst_offset + mb_start * N + nb_start, ldc);
+                    (float *) ((char *) dst->data + dst_offset) + mb_start * N + nb_start, ldc);
```

## Deviations from the handoff

1. **Branch name**: the session requires pushing to `claude/ecstatic-hamilton-b0jqyw`, not `cvec-per-request`.
2. **`emo_steer.py` was missing**; a stand-in was written (with your OK). Probe split, layer rule and dose sweep are mine.
   Rerun Track A with the real script before trusting T8.
3. **Vectors are not committed**: the fork is public and the guardrails say not to publish vectors. They are handed over
   as files, and `export_cvec.py` + `emo_steer.py` regenerate them (`REPRO.md` sections 4).
4. **random_d3 hnorm** for Qwen3.5-2B comes from a quick forward pass on 4 neutral chat prompts (like Qwen3-1.7B), not from
   `axes.pt`; the vector is only used for T2, where any large norm works.
5. **T5 dtype**: HF ran in FP32 (BF16 weights are exact in FP32) as the reference; llama.cpp ran the BF16 GGUF.
6. **T6-T10 run with `--no-repack`** because of the AMX bug above.
7. **T8/T10 scripts** live in `tools/cvec-live/` instead of `tools/server/tests/`: they need HF transformers and a 2B
   model, which the server test suite does not have. T6, T7, T9 are automated in `test_cvec.py` and run on the tiny
   preset model by default (21 tests, ~4 min) or on Qwen3.5-2B via `CVEC_MODEL`/`CVEC_DIR`/`CVEC_ID`/`CVEC_PROMPT`.
8. **Files outside the spec list**: `common/arg.cpp` + `common/common.h` (the new flags must live there),
   `server-task.{h,cpp}` (task params + `generation_settings` echo), `server-context.h` + `server.cpp` (route), and the
   400 mapping for malformed JSON in `server.cpp` (needed for the NaN/inf part of T9; affects all routes).
9. **No per-apply combined-buffer cache**: building the buffer is a 49k-float loop, cheaper than a cache lookup would save.
10. **`slot.cvec`** is not a separate member: the slot reads `task->params.cvec`, which lives exactly as long as the task.

## Open questions

- **Reserve on every switch.** `llama_set_adapter_cvec` always triggers `sched_reserve`. Only enabling/disabling the
  vector or changing the layer range changes the graph; changing values does not. A libllama change (reserve only on
  topology change, or keep the vector always "on" with zeros) would remove most of the T10 penalty. Out of scope here.
- **GPU path untested.** No GPU in this environment. On CUDA the reserve cost and graph capture interplay may be very
  different; T10 must be re-measured on the target.
- **Dose unit.** `dose x median ||h||` at L = 8 is too strong for this 2B model and these axes. Fluency holds only
  for \|dose\| <= 0.3-0.4, where the valence shift is small (about 0.9 points on 1-9). Either the real `emo_steer.py`
  axes are cleaner, or a 2B model does not have a strong, clean valence direction. Worth re-running on the 27B target.
- **Per-sequence vectors.** Both T10 penalties (different vectors, and decode-only prefill vs steered generation) come
  from the vector being per context. A per-sequence control vector in libllama (select a vector row per token by
  `seq_id`) would let all slots share one decode call. It needs graph changes, so it is out of scope here.
- **AMX fix upstream?** The two-line AMX fix above is verified only on this machine and model.
- **MTP / speculative decoding** with steering is unsupported by design (drafts are not steered). The server does not
  reject `cvec` when speculative decoding is on; it probably should.
- **Qwen3.8-27B** was not tested (no GPU, no weights). Same graph code (`qwen35.cpp`), so T1-T5 should carry over.
