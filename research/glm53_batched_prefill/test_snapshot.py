import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.glm53_batched_prefill.snapshot_key import canonical_key


def manifest():
    return {
        "schema": 1,
        "model_shards": [{"sha256": "a" * 64, "size_bytes": 110_000_000_000}],
        "tokenizer_sha256": "b" * 64,
        "template_sha256": "c" * 64,
        "runtime_source_sha256": "d" * 64,
        "runtime_binaries": {"probe": "e" * 64, "libllama": "f" * 64},
        "prefix_tokens": list(range(208)),
        "prefix_positions": list(range(208)),
        "state_parameters": {
            "n_ctx": 1024, "n_batch": 8, "n_ubatch": 8, "n_seq_max": 1,
            "type_k": "f16", "type_v": "f16", "flash_attn": False,
            "offload_kqv": False, "n_gpu_layers": 0,
            "rope_scaling_type": 0, "rope_freq_base": 0, "rope_freq_scale": 0,
            "n_threads": 10, "n_threads_batch": 10,
        },
        "render_options": {"reasoning_effort": "low", "enable_thinking": True},
    }


def test_canonical_key_order_is_stable_and_covers_all_dependencies():
    base = manifest()
    expected, _ = canonical_key(base)
    assert canonical_key(dict(reversed(list(base.items()))))[0] == expected
    for field in ("tokenizer_sha256", "template_sha256", "runtime_source_sha256"):
        changed = copy.deepcopy(base)
        changed[field] = "0" * 64
        assert canonical_key(changed)[0] != expected
    mutations = [
        ("model_shards", [{"sha256": "0" * 64, "size_bytes": 110_000_000_000}]),
        ("runtime_binaries", {"probe": "0" * 64, "libllama": "f" * 64}),
        ("prefix_tokens", [999] + base["prefix_tokens"][1:]),
        ("render_options", {"reasoning_effort": "high", "enable_thinking": True}),
        ("state_parameters", {**base["state_parameters"], "n_ubatch": 4}),
    ]
    for field, value in mutations:
        changed = copy.deepcopy(base)
        changed[field] = value
        assert canonical_key(changed)[0] != expected


@pytest.mark.parametrize("mutation", [
    lambda x: x.pop("template_sha256"),
    lambda x: x.update(prefix_positions=list(range(1, 209))),
    lambda x: x.update(prefix_tokens=[]),
    lambda x: x.update(prefix_tokens=[True] * 208),
    lambda x: x["state_parameters"].update(n_seq_max=600),
    lambda x: x["state_parameters"].update(n_ctx=208),
    lambda x: x["state_parameters"].update(n_ubatch=16),
    lambda x: x["state_parameters"].update(rope_freq_base=float("nan")),
    lambda x: x["model_shards"][0].update(sha256="unverified"),
    lambda x: x.update(runtime_binaries={}),
])
def test_incomplete_or_incompatible_key_refused(mutation):
    candidate = manifest()
    mutation(candidate)
    with pytest.raises(ValueError):
        canonical_key(candidate)


def test_native_snapshot_contract_fixture(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("CommonCrypto fixture requires Darwin")
    source = Path(os.environ.get("AIRLLM_PROFILE_RUNTIME_SOURCE", ""))
    if not (source / "include/llama.h").is_file():
        pytest.skip("set AIRLLM_PROFILE_RUNTIME_SOURCE to pinned llama.cpp source")
    root = Path(__file__).parent
    binary = tmp_path / "snapshot-fixture"
    compiled = subprocess.run([
        "clang++", "-std=c++17", "-O2", "-Wno-deprecated-declarations",
        "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
        str(root / "snapshot_fixture.cpp"), "-o", str(binary),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["fixture_passed"] is True
    assert record["native_model_loaded"] is False
