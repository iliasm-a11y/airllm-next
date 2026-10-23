"""Granite legacy expert banks must be selected as experts, router stays FP32."""
import numpy as np
from safetensors.numpy import save_file

from airllm.nextgen.sidecar import convert_experts, expert_matrices
from airllm.nextgen.storage import OffsetCheckpoint


def test_legacy_granite_experts_are_converted_and_mixed_router_is_excluded(tmp_path):
    root = tmp_path/'model'
    root.mkdir()
    prefix = 'model.layers.0.block_sparse_moe.'
    bank_in, bank_out = prefix+'input_linear.weight', prefix+'output_linear.weight'
    router, attention = prefix+'router.layer.weight', 'model.layers.0.self_attn.q_proj.weight'
    rng = np.random.default_rng(9)
    weights = {bank_in: rng.normal(size=(2, 64, 64)).astype(np.float32),
        bank_out: rng.normal(size=(2, 64, 32)).astype(np.float32),
        router: rng.normal(size=(2, 64)).astype(np.float32),
        attention: rng.normal(size=(64, 64)).astype(np.float32)}
    save_file(weights, root/'model.safetensors')
    with OffsetCheckpoint(root, tmp_path/'meta') as cp:
        assert list(expert_matrices(cp)) == [(bank_in, 0, (64, 64)), (bank_in, 1, (64, 64)),
            (bank_out, 0, (64, 32)), (bank_out, 1, (64, 32))]
        sidecar = convert_experts(cp, {}, tmp_path/'sidecar', group_size=32,
                                  row_chunk=16, scope='mixed', linear_bits=8)
        assert router not in sidecar.entries
        assert sidecar.entries[attention]['bits'] == 8
        for name in (bank_in, bank_out):
            for expert in range(2):
                entry = sidecar.entries[f'{name}#{expert}']
                assert entry['bits'] == 4
                assert entry['relative_l2_error'] < .15
        # Resume uses identical per-matrix precision and content verification.
        resumed = convert_experts(cp, {}, tmp_path/'sidecar', group_size=32,
                                  row_chunk=16, scope='mixed', linear_bits=8)
        assert resumed.entries == sidecar.entries
