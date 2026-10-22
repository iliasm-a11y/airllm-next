"""Measure singleton/block arithmetic on retained, trained BF16 experts.

This is a CPU numerical-contract experiment, not a full language model or a
throughput benchmark. No model is downloaded. The retained pack is read block
by block with its SHA256 checks; it is never accepted as a standalone runtime
checkpoint. Original source identity is not bypassed in LosslessOverlay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from common import ROOT, digest, save
from airllm.nextgen.lossless import RECORD, decode_block


def retained_arrays(directory, max_raw_bytes=32 * 1024**2):
    """Read a bounded research fixture, checking all reconstructed block hashes."""
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    if metadata.get('version') != 1:
        raise ValueError('Unsupported retained research pack')
    entries = metadata['entries']
    total = sum(e['raw_size'] for e in entries.values())
    if not 0 < total <= max_raw_bytes:
        raise MemoryError('Retained expert fixture exceeds experiment allowance')
    arrays = {}
    with (directory / 'blocks.idx').open('rb') as index, (directory / 'weights.bin').open('rb') as weights:
        for key, entry in entries.items():
            if entry['dtype'] != 'BF16' or entry['raw_size'] != 2 * int(np.prod(entry['shape'])):
                raise ValueError('Expected trained BF16 expert matrices')
            raw = bytearray()
            index.seek(entry['index_offset'])
            for _ in range(entry['blocks']):
                record = index.read(RECORD.size)
                if len(record) != RECORD.size:
                    raise EOFError('Truncated retained block index')
                offset, size, raw_size, kind, expected_hash = RECORD.unpack(record)
                expected_size = min(entry['block_bytes'], entry['raw_size'] - len(raw))
                if raw_size != expected_size or not 0 < size <= entry['block_bytes']:
                    raise ValueError('Invalid retained block size')
                weights.seek(offset)
                payload = weights.read(size)
                if len(payload) != size:
                    raise EOFError('Truncated retained expert weights')
                decoded = decode_block(kind, payload, raw_size)
                if hashlib.sha256(decoded).digest() != expected_hash:
                    raise ValueError('Retained expert checksum mismatch')
                raw.extend(decoded)
            if len(raw) != entry['raw_size']:
                raise ValueError('Retained tensor length mismatch')
            arrays[key] = np.frombuffer(raw, dtype='<u2').reshape(entry['shape'])
    return arrays


def compare(a, b):
    """Compare storage bits, distinguishing signed zero from numeric equality."""
    a, b = a.detach().contiguous().cpu(), b.detach().contiguous().cpu()
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError('Incomparable kernel outputs')
    import torch
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Non-finite expert output')
    bits = torch.int16 if a.element_size() == 2 else torch.int32
    different = int((a.view(bits) != b.view(bits)).sum())
    return dict(bitwise_equal=different == 0, different_elements=different,
                elements=a.numel(), max_abs_error=float((a.float() - b.float()).abs().max()))


def run(directory):
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    arrays = retained_arrays(directory)
    groups = sorted({key.rsplit('.', 2)[0] for key in arrays})
    expected = json.loads((ROOT / 'results/trained_parity.json').read_text())
    if len(groups) != 4 or len(arrays) != 12:
        raise ValueError('Expected the four retained trained Ling experts')
    recorded = np.load(ROOT / 'results/trained_expert_outputs.npz')
    rng = np.random.default_rng(20261022)
    inputs = rng.normal(size=(32, 2048)).astype(np.float32)
    inputs[:4] = recorded['inputs']
    trials = []
    weight_hashes = {}
    checked_numpy_reference = []
    for group in groups:
        values = [arrays[f'{group}.{name}.weight'] for name in ['gate_proj', 'up_proj', 'down_proj']]
        for name, value in zip(['gate', 'up', 'down'], values):
            weight_hashes[f'{group}.{name}'] = hashlib.sha256(value.tobytes()).hexdigest()
        # Reproduce the retained FP32 NumPy outputs before changing arithmetic.
        from trained_parity import evaluate, fp32
        previous = next(row for row in expected['results'] if row['expert'] == group)
        numpy_output = evaluate(recorded['inputs'], [fp32(value) for value in values])
        numpy_hash = hashlib.sha256(numpy_output.tobytes()).hexdigest()
        if numpy_hash != previous['output_sha256']:
            raise AssertionError('Retained trained-expert reference was not reproduced')
        checked_numpy_reference.append(dict(expert=group, output_sha256=numpy_hash, bitwise_reproduced=True))
        for dtype in [torch.float32, torch.bfloat16]:
            weights = [torch.from_numpy(value.copy()).view(torch.bfloat16).to(dtype) for value in values]
            x = torch.from_numpy(inputs.copy()).to(dtype)
            # Every BF16 word must survive the PyTorch weight import.
            for original, weight in zip(values, weights):
                if not np.array_equal(weight.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16), original):
                    raise AssertionError('Weight import changed BF16 bits')

            def expert(rows):
                return functional.linear(functional.silu(functional.linear(rows, weights[0]))
                                         * functional.linear(rows, weights[1]), weights[2])

            for length in [1, 2, 4, 8, 16, 32]:
                with torch.inference_mode():
                    singleton = torch.cat([expert(row[None, :]) for row in x[:length]], dim=0)
                    block = expert(x[:length])
                    # A shape-invariant per-row implementation can retain its
                    # own exact contract while sharing the same weight load.
                    rowwise_replay = torch.cat([expert(row[None, :]) for row in x[:length]], dim=0)
                row = dict(expert=group, dtype=str(dtype), positions=length,
                           singleton_vs_block=compare(singleton, block),
                           singleton_vs_same_row_order=compare(singleton, rowwise_replay))
                if not row['singleton_vs_same_row_order']['bitwise_equal']:
                    raise AssertionError('Identical per-row CPU execution is not deterministic')
                trials.append(row)
    # A deterministic arithmetic counterexample to changing expert accumulation
    # order. It is synthetic and is reported separately from the trained data.
    a = np.float32(2**24)
    left = np.float32(np.float32(a + np.float32(1)) - a)
    right = np.float32(np.float32(a - a) + np.float32(1))
    assert left == 0 and right == 1
    result = dict(
        evidence='measured PyTorch CPU arithmetic on four retained trained Ling experts',
        python=platform.python_version(), python_executable=sys.executable,
        torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
        matmul_precision=torch.get_float32_matmul_precision(),
        pack_manifest_sha256=digest(Path(directory) / 'manifest.json'),
        source='Existing retained expert pack; reconstructed blocks verified by SHA256',
        weight_sha256=weight_hashes, numpy_reference_reproduction=checked_numpy_reference,
        trials=trials, block_comparisons=len(trials),
        block_comparisons_with_bit_differences=sum(not t['singleton_vs_block']['bitwise_equal'] for t in trials),
        same_row_order_comparisons_bitwise_equal=all(t['singleton_vs_same_row_order']['bitwise_equal'] for t in trials),
        arithmetic_counterexample=dict(evidence='synthetic float32 scalar identity counterexample',
                                      contributions=[int(a), 1, -int(a)], order_0_1_2=float(left), order_0_2_1=float(right)),
        full_model_logits_or_tokens_compared=False, metal_model_runtime_measured=False,
        inference_speed_measured=False,
        limitations=[
            'Different block outputs do not establish a token change in a full model.',
            'Observed CPU differences do not quantify MLX/Metal differences.',
            'The rowwise control proves its own CPU contract only; no full-model invariance is inferred.',
            'All routes, language quality, full-model KV states, and end-to-end generation remain untested here.',
        ])
    save('numerical_contract.json', result)
    print(json.dumps({k: result[k] for k in ['block_comparisons', 'block_comparisons_with_bit_differences',
                                           'same_row_order_comparisons_bitwise_equal']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-pack', type=Path, required=True)
    run(parser.parse_args().expert_pack)
