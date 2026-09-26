# Per-request control vectors (llama.cpp-volanus)

This fork lets `llama-server` apply control vectors (steering vectors) **per request**, without reloading the model.
Stock llama.cpp only takes them as startup flags (`--control-vector*`), so they apply to every request and a change
needs a restart.

Everything lives in these places:

| Path | What |
| --- | --- |
| `tools/server/` | the server feature: `--cvec-dir`, `cvec` request field, `GET /cvecs` |
| `tools/cvec-live/cvec-live.cpp` | `llama-cvec-live`, a small harness that switches vectors between decode calls |
| `tools/cvec-live/export_cvec.py` | writes control-vector GGUFs (from `axes.pt`, or zero/random test vectors) |
| `tools/cvec-live/emo_steer.py` | example pipeline: build valence/arousal axes in HF transformers, dose sweep |
| `tools/cvec-live/parity.py`, `cvec_sweep.py`, `cvec_bench.py` | HF parity check, server dose sweep, throughput bench |
| `tools/server/tests/unit/test_cvec.py` | server tests |
| `tools/cvec-live/REPORT.md`, `REPRO.md` | test results on Qwen3.5-2B and the commands behind them |

## How it works

### What a control vector is

A control vector is one `n_embd` vector per layer. llama.cpp adds it to the residual stream at the end of layer `il`,
after the FFN residual (`build_cvec(cur, il)` in `src/models/*.cpp`). This is the same point as the output of
`model.layers[il]` in HF transformers, so HF layer `L` maps to `direction.<L>` in the GGUF (checked with `parity.py`).
Layer 0 cannot be steered. The vector is added as-is, with no scaling at apply time.

libllama already supports changing it at runtime: `llama_set_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)`
copies the data into tensors that are allocated once per context, and `data = NULL` turns it off. It can be called
between any two `llama_decode` calls. The call itself takes 0.03-0.2 ms. It also makes the next decode rebuild the
scheduler graphs, which cost roughly 3-30 ms per switch on CPU.

### What the server does with it

```
startup:   --cvec-dir DIR  ->  load every DIR/*.gguf, id = file stem (valence.gguf -> "valence")
request:   "cvec": [{"id": "valence", "scale": 0.8}]  ->  validated, stored on the task
slot:      active key = the request's {id: scale} map
           (empty while a decode-only request is still processing its prompt)
batching:  slots only share a decode call when their active keys are equal
decode:    before llama_decode, if the batch's key differs from the last one applied:
           set  global --control-vector + sum(scale_i * direction_i)   (or NULL if both are empty)
```

This mirrors the existing per-request LoRA path (`lora` field, `can_batch_with()`, `common_set_adapter_lora()` once per
batch). Because the vector is per context, not per sequence, slots with different vectors run in separate decode calls.
That is correct, but it costs throughput (see Performance).

### Decode-only (default) vs steered prompt

`cvec_decode_only: true` (the default) keeps the prompt unsteered and applies the vector from the first generated
token on. Timeline for one request:

```
prompt tokens  -> decode with NO vector   -> logits -> token 1   (token 1 is the same at every dose)
token 1        -> decode WITH the vector  -> logits -> token 2   (steering starts to show here)
token 2 ...    -> decode WITH the vector  -> ...
```

This keeps the prompt cache valid: cached prompt KV and hybrid-model checkpoints were computed without the vector, so
any later request can reuse them.

`cvec_decode_only: false` steers the prompt too, which is what an HF forward hook does. The server then forces
`cache_prompt: false` for that request and clears the slot's prompt (KV and checkpoints) when the request ends, so no
other request can reuse steered state.

## Quick start

```sh
# build
cmake -B build -DCMAKE_BUILD_TYPE=Release    # add -DGGML_CUDA=ON on NVIDIA
cmake --build build -j --target llama-server llama-cvec-live

# put control vectors in a directory
ls vectors/
#   valence.gguf  arousal.gguf

# start the server
./build/bin/llama-server -m qwen3.5-2b-q4_k_m.gguf --cvec-dir vectors -np 4 -c 8192
```

List what was loaded:

```sh
curl -s localhost:8080/cvecs
# [{"id":"arousal","layers":[8],"n_embd":2048},{"id":"valence","layers":[8],"n_embd":2048}]
```

Native completion:

```sh
curl -s localhost:8080/completion -d '{
  "prompt": "<|im_start|>user\nTell me about your day.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "n_predict": 64,
  "cvec": [{"id": "valence", "scale": 0.8}, {"id": "arousal", "scale": -0.4}]
}'
```

OpenAI-compatible chat (extra body fields):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Tell me about your day in a few sentences."}],
    max_tokens=40,
    extra_body={
        "cvec": [{"id": "valence", "scale": -0.8}],
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(r.choices[0].message.content)
# "I don't have a day, so I can't tell you what's happening. I'm just an AI, and I don't feel like I have a purpose ..."
# same request without cvec:
# "I don't have a physical body or a personal day, so I don't experience time or events in the way humans do. ..."
```

## Reference

### Server flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cvec-dir DIR` | unset | load every `*.gguf` in `DIR` as a selectable vector, id = file stem. The server refuses to start if a file's `n_embd` differs from the model's, if a direction is past the last layer, or if the dir has no `.gguf` files |
| `--cvec-max-scale X` | 3.0 | reject requests with \|scale\| > X |
| `--control-vector`, `--control-vector-scaled`, `--control-vector-layer-range` | stock | still work as a **global baseline** that applies to every request (prompt and generation). The effective vector is always global + request |

### Request fields

Accepted on `/completion`, `/completions`, `/v1/completions`, `/chat/completions` and `/v1/chat/completions` (as extra body fields on the OpenAI routes).

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cvec` | array of `{"id": string, "scale": number}` | none | vectors to add: sum of `scale x direction`, on top of the global baseline. Scale 0 entries are dropped |
| `cvec_decode_only` | bool | `true` | `true`: steer only generated tokens. `false`: steer the prompt too (disables prompt caching for this request) |

The request's settings are echoed in `generation_settings.cvec` and `generation_settings.cvec_decode_only`.

Errors are HTTP 400 with a message:

| Case | Message |
| --- | --- |
| unknown id | `unknown control vector id 'x'` |
| scale missing, null, a string | `each entry must have a string 'id' and a number 'scale'` |
| \|scale\| > max, or not finite | `scale of 'x' must be finite and in [-3, 3]` |
| same id twice | `duplicate id 'x'` |
| `cvec` is not an array | `must be an array of objects with 'id' and 'scale' fields` |
| `cvec` sent without `--cvec-dir` | `control vectors are disabled, start the server with --cvec-dir` |
| bare `NaN`, `Infinity`, `1e999` in the body | JSON `parse error` / `number overflow` (this fork maps malformed JSON to 400 on all routes) |

### `GET /cvecs`

Returns `[{"id", "layers", "n_embd"}]`, where `layers` lists the layers that have a non-zero direction.

## Making vectors

A control-vector file is a GGUF with one F32 1-D tensor of length `n_embd` per steered layer, named `direction.<il>`
(`il >= 1`). This is the format `tools/cvector-generator` writes and `common_control_vector_load` reads. Minimal writer
with the fork's `gguf-py`:

```python
import numpy as np, gguf

vec = np.random.randn(2048).astype(np.float32)        # your direction, already scaled
w = gguf.GGUFWriter("vectors/myvec.gguf", "controlvector")
w.add_string("controlvector.model_hint", "qwen35")
w.add_int32("controlvector.layer_count", 1)
w.add_tensor("direction.8", vec)                      # HF layers[8] output == llama.cpp il 8
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
```

Or use `export_cvec.py`:

```sh
# from an axes file (W [2, n_layers, n_embd] unit directions, hnorm [n_layers], layer)
python tools/cvec-live/export_cvec.py --axes out/axes.pt --out-dir vectors
# zero and random test vectors (random = unit vector x 3 x median hidden norm at the middle layer)
python tools/cvec-live/export_cvec.py --test-vectors --model path/to/hf-model --out-dir vectors-test
```

**Scale units.** The server adds `scale x direction` as-is. `export_cvec.py` bakes `median ||h|| at layer L` into the
direction, so `scale` becomes a relative dose: 1.0 adds a vector as long as a typical hidden state. On Qwen3.5-2B, with
the example valence axis at layer 8, fluency holds up to about \|scale\| 0.3-0.4 and breaks down past 0.8 (see
`REPORT.md`).

`emo_steer.py` is an example pipeline for building such axes (corpus -> per-layer diff-of-means directions -> held-out
probe -> dose sweep): `MODEL=Qwen/Qwen3.5-2B python tools/cvec-live/emo_steer.py corpus|extract|steer`.

## The harness: `llama-cvec-live`

Loads the model once, then runs a fixed schedule: turns none -> +v -> -v -> none (memory cleared per turn, same
prompt, greedy), then one run that turns +v on in the middle of generation. It prints the tokens, where the runs
diverge, and the time of every `llama_set_adapter_cvec` call.

```sh
./build/bin/llama-cvec-live -m model.gguf --cvec vectors/valence.gguf --scale 1.0 -f prompt.txt -n 64 [--decode-only] [--switch-at 32] [--logits-out PREFIX]
```

Read the `first_diff(...)` lines at the end: `turn1 == turn4` shows the vector was cleared correctly, `turn2 != turn1`
shows it had an effect, and `switch` shows the token where the mid-generation switch started to matter.

## Performance

Measured on a 4-core CPU with Qwen3.5-2B Q4_K_M:

| Setup | tok/s |
| --- | ---: |
| `-np 1`, no vector / steered | 11.9 / 12.1 |
| `-np 4`, no vector | 24.7 |
| `-np 4`, all the same vector, steered prompt | 25.6 |
| `-np 4`, all the same vector, decode-only | 18-19 |
| `-np 4`, 4 different vectors | 12.7 |

- Different vectors cannot share a decode call, so each slot decodes alone.
- Decode-only has a smaller cost of the same kind: a request's unsteered prompt cannot share a decode call with other
  slots that are already generating with a vector. It only matters when requests arrive staggered.
- A per-sequence vector in libllama would remove both costs. It is not implemented (it needs graph changes).

## Caveats

- **Intel AMX CPUs**: upstream llama.cpp (not this feature) corrupts outputs of Qwen3.5-style hybrid models when
  several sequences share a batch and the weights use the AMX repack buffer. Use `--no-repack` (or
  `LLAMA_ARG_REPACK=false`) with `-np > 1` on such CPUs. Root cause and a fix are in `REPORT.md`. GPUs are not affected.
- **Speculative / MTP decoding**: drafts are not steered. Do not combine with `cvec`.
- **Multi-turn with decode-only + prompt cache**: the previous turn's reply was generated with the vector, and its
  cached KV is reused by the next turn. The prompt part itself is never steered. This is by design.
- **Exact reproducibility under load**: llama.cpp numerics depend on batch shape, so the same request can give a
  different greedy continuation depending on which other requests share its batch. This is independent of steering,
  but steered outputs near a tie show it more.
- **Direct API users**: `llama_set_adapter_cvec` only overwrites the layers covered by `len`. Layers past the end of a
  shorter buffer keep their old values. Always pass a full `n_embd x n_layer` buffer (the server and harness do).

## Tests

```sh
cd tools/server/tests
export LLAMA_SERVER_BIN_PATH=$PWD/../../../build/bin/llama-server
python -m pytest -v unit/test_cvec.py                  # tiny preset model + random vector, about a minute once models are cached

# on a real model and vector (on AMX CPUs also export LLAMA_ARG_REPACK=false)
# $(cat ...) drops trailing newlines, the "." trick keeps the prompt's final "\n\n"
export CVEC_PROMPT="$(cat prompt.txt; echo .)"; export CVEC_PROMPT="${CVEC_PROMPT%.}"
CVEC_MODEL=qwen3.5-2b-q4_k_m.gguf CVEC_DIR=vectors CVEC_ID=valence python -m pytest -v -s unit/test_cvec.py
```
