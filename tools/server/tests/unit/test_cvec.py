import os
import struct
from pathlib import Path

import numpy as np
import pytest
import requests
from utils import *

# default: tiny preset model + a random control vector written at test time
# set CVEC_MODEL, CVEC_DIR, CVEC_ID (and CVEC_PROMPT) to run against a real model, e.g. Qwen3.5-2B + valence.gguf
CVEC_MODEL = os.environ.get("CVEC_MODEL")
CVEC_DIR = os.environ.get("CVEC_DIR")
CVEC_ID = os.environ.get("CVEC_ID", "rand")
PROMPT = os.environ.get("CVEC_PROMPT", "Once upon a time")
N_PREDICT = 32
DOSES = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]


def write_cvec(path: Path, layer: int, vec: np.ndarray):
    # minimal GGUF v3 file with one F32 tensor "direction.<layer>", as read by common_control_vector_load
    name = f"direction.{layer}".encode()
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    data += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, len(vec), 0, 0)
    data += b"\0" * (-len(data) % 32)
    data += np.asarray(vec, dtype="<f4").tobytes()
    path.write_bytes(data)


def make_server(cvec_dir: str | None, n_slots: int = 1) -> ServerProcess:
    if CVEC_MODEL:
        server = ServerProcess()
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = CVEC_MODEL
        server.n_ctx = 2048 * n_slots
        server.seed = 42
    else:
        server = ServerPreset.tinyllama2()
        server.n_ctx = 512 * n_slots
    server.n_slots = n_slots
    server.cvec_dir = cvec_dir
    return server


@pytest.fixture(scope="module")
def n_embd() -> int:
    server = make_server(None)
    server.start()
    res = server.make_request("GET", "/v1/models")
    server.stop()
    return res.body["data"][0]["meta"]["n_embd"]


@pytest.fixture(scope="module")
def cvec_dir(n_embd, tmp_path_factory) -> str:
    if CVEC_DIR:
        return CVEC_DIR
    d = tmp_path_factory.mktemp("cvec")
    v = np.random.default_rng(0).standard_normal(n_embd)
    write_cvec(d / "rand.gguf", 1, 100.0 * v / np.linalg.norm(v))
    return str(d)


def completion(server: ServerProcess, scale: float | None, decode_only: bool = True, cache_prompt: bool = False, path: str = "/completion"):
    data = {
        "prompt": PROMPT,
        "n_predict": N_PREDICT,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "cvec_decode_only": decode_only,
    }
    if scale is not None:
        data["cvec"] = [{"id": CVEC_ID, "scale": scale}]
    res = server.make_request("POST", path, data=data)
    assert res.status_code == 200, res.body
    return res.body


def test_cvec_list(cvec_dir, n_embd):
    server = make_server(cvec_dir)
    server.start()
    res = server.make_request("GET", "/cvecs")
    assert res.status_code == 200
    entry = next(e for e in res.body if e["id"] == CVEC_ID)
    assert entry["n_embd"] == n_embd
    assert len(entry["layers"]) > 0


# T6: slots with different vectors run in separate decode calls, identical vectors give identical outputs
@pytest.mark.parametrize("decode_only", [True, False])
def test_cvec_concurrent(cvec_dir, decode_only):
    server = make_server(cvec_dir, n_slots=4)
    server.start()
    solo = completion(server, None)["tokens"]
    first = None
    for i in range(3):
        scales = [1.0, None, -1.0, 1.0]
        tasks = [(completion, (server, s, decode_only)) for s in scales]
        pos, none, neg, pos2 = [r["tokens"] for r in parallel_function_calls(tasks)]
        print(f"repeat {i}: +1 {pos}\n  none {none}\n  -1 {neg}\n  +1 {pos2}")
        assert none == solo
        assert pos == pos2
        assert pos != neg and pos != none and neg != none
        if first is None:
            first = (pos, none, neg)
        assert (pos, none, neg) == first


# T7: with cvec_decode_only the first token comes from unsteered prompt logits
def test_cvec_decode_only_first_token(cvec_dir):
    server = make_server(cvec_dir)
    server.start()
    tokens = {dose: completion(server, dose)["tokens"] for dose in DOSES}
    for dose, t in tokens.items():
        print(f"dose {dose:+.1f}: {t}")
    assert len({t[0] for t in tokens.values()}) == 1
    assert any(t != tokens[0.0] for t in tokens.values())


def test_cvec_prompt_cache(cvec_dir):
    server = make_server(cvec_dir)
    server.start()
    solo = completion(server, None, cache_prompt=True)["tokens"]

    # decode-only: the cached prompt was not steered, it can be reused
    completion(server, 1.0, decode_only=True, cache_prompt=True)
    res = completion(server, None, cache_prompt=True)
    assert res["timings"]["cache_n"] > 0
    assert res["tokens"] == solo

    # steered prompt: never cached, never reused
    res = completion(server, 1.0, decode_only=False, cache_prompt=True)
    assert res["timings"]["cache_n"] == 0
    assert res["tokens"] != solo
    res = completion(server, None, cache_prompt=True)
    assert res["tokens"] == solo


def test_cvec_oai_completions(cvec_dir):
    server = make_server(cvec_dir)
    server.start()
    none = completion(server, None, path="/v1/completions")["choices"][0]["text"]
    pos = completion(server, 1.0, path="/v1/completions")["choices"][0]["text"]
    assert none != pos


# T9: bad requests
@pytest.mark.parametrize("cvec,msg", [
    ([{"id": "missing", "scale": 1.0}], "unknown control vector id"),
    ([{"id": CVEC_ID, "scale": None}], "number 'scale'"),
    ([{"id": CVEC_ID, "scale": "NaN"}], "number 'scale'"),
    ([{"id": CVEC_ID}], "number 'scale'"),
    ([{"id": CVEC_ID, "scale": 3.5}], "must be finite and in"),
    ([{"id": CVEC_ID, "scale": -3.5}], "must be finite and in"),
    ([{"id": CVEC_ID, "scale": 1.0}, {"id": CVEC_ID, "scale": 0.5}], "duplicate id"),
    ({"id": CVEC_ID, "scale": 1.0}, "must be an array"),
])
def test_cvec_bad_request(cvec_dir, cvec, msg):
    server = make_server(cvec_dir)
    server.start()
    for path in ["/completion", "/v1/completions"]:
        res = server.make_request("POST", path, data={"prompt": PROMPT, "n_predict": 4, "cvec": cvec})
        assert res.status_code == 400
        assert msg in res.body["error"]["message"]


@pytest.mark.parametrize("scale,msg", [
    ("NaN", "parse error"),
    ("Infinity", "parse error"),
    ("1e999", "number overflow"),
    ("1e39", "must be finite and in"),  # finite double, inf as float
])
def test_cvec_bad_request_non_finite(cvec_dir, scale, msg):
    server = make_server(cvec_dir)
    server.start()
    # raw body, these literals are not valid JSON or overflow
    body = '{"prompt": "hi", "n_predict": 4, "cvec": [{"id": "%s", "scale": %s}]}' % (CVEC_ID, scale)
    res = requests.post(server.make_url("/completion"), data=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 400
    assert msg in res.json()["error"]["message"]


def test_cvec_without_dir():
    server = make_server(None)
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 4, "cvec": [{"id": CVEC_ID, "scale": 1.0}]})
    assert res.status_code == 400
    assert "--cvec-dir" in res.body["error"]["message"]


@pytest.mark.parametrize("delta_n_embd,layer", [(1, 1), (0, 10000)])
def test_cvec_refuse_start(n_embd, tmp_path, delta_n_embd, layer):
    write_cvec(tmp_path / "bad.gguf", layer, np.ones(n_embd + delta_n_embd))
    server = make_server(str(tmp_path))
    with pytest.raises(RuntimeError):
        server.start()
