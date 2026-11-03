"""Canonical dependency key for the experimental native prefix snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HASH_FIELDS = (
    "tokenizer_sha256", "template_sha256", "runtime_source_sha256",
)
PARAMETERS = (
    "n_ctx", "n_batch", "n_ubatch", "n_seq_max", "type_k", "type_v",
    "flash_attn", "offload_kqv", "n_gpu_layers", "rope_scaling_type",
    "rope_freq_base", "rope_freq_scale", "n_threads", "n_threads_batch",
)
FIELDS = {
    "schema", "model_shards", "tokenizer_sha256", "template_sha256",
    "runtime_source_sha256", "runtime_binaries", "prefix_tokens",
    "prefix_positions", "state_parameters", "render_options",
}


def _hash(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("dependency identity must be a lowercase SHA256")


def canonical_key(manifest: dict) -> tuple[str, bytes]:
    """Reject incomplete identities; every accepted field contributes to the key.

    Sources must supply actual native-rendered IDs and effective parameters.
    This validates the record, not whether its evidence is truthful or current.
    """
    if set(manifest) != FIELDS or manifest["schema"] != 1:
        raise ValueError("snapshot manifest schema is incomplete or unknown")
    for name in HASH_FIELDS:
        _hash(manifest[name])
    shards = manifest["model_shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("ordered original model shards are required")
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"sha256", "size_bytes"}:
            raise ValueError("each original shard needs SHA256 and byte size")
        _hash(shard["sha256"])
        if type(shard["size_bytes"]) is not int or shard["size_bytes"] <= 0:
            raise ValueError("invalid shard size")
    binaries = manifest["runtime_binaries"]
    if not isinstance(binaries, dict) or not binaries:
        raise ValueError("executable and loaded runtime library identities required")
    for name, value in binaries.items():
        if not isinstance(name, str) or not name:
            raise ValueError("runtime binary names are required")
        _hash(value)
    tokens = manifest["prefix_tokens"]
    if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("nonempty native prefix token IDs are required")
    positions = manifest["prefix_positions"]
    if not isinstance(positions, list) or any(type(x) is not int for x in positions) or positions != list(range(len(tokens))):
        raise ValueError("prefix positions must be contiguous and start at zero")
    params = manifest["state_parameters"]
    if not isinstance(params, dict) or not set(PARAMETERS) <= set(params):
        raise ValueError("effective state parameters are incomplete")
    if type(params["n_seq_max"]) is not int or params["n_seq_max"] != 1:
        raise ValueError("one physical sequence is required")
    for name in ("n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch"):
        if type(params[name]) is not int or params[name] <= 0:
            raise ValueError("invalid positive context or scheduling parameter")
    if params["n_ctx"] <= len(tokens) or params["n_batch"] < params["n_ubatch"]:
        raise ValueError("context must admit a suffix and batch must admit microbatch")
    if not isinstance(manifest["render_options"], dict):
        raise ValueError("native template rendering options are required")
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest(), data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    key, canonical = canonical_key(json.loads(args.manifest.read_text()))
    print(json.dumps({"snapshot_key_sha256": key, "canonical_bytes": len(canonical)}))


if __name__ == "__main__":
    main()
