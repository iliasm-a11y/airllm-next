#!/usr/bin/env python3
"""Local trained Granite expert samples: original/lossless/affine 4/8 bit.

No download or source mutation. Decode times are warm in-memory CPU medians;
they exclude file I/O, quantization creation, MLX QMM, and model generation.
Gaussian activation probes measure local MLP error, not final-logit accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))


def measure(call, repeats):
    result = call()
    elapsed = []
    for _ in range(repeats):
        begin = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter()-begin)
    return result, {'median_seconds': statistics.median(elapsed),
                    'samples_seconds': elapsed, 'repeats': repeats}


def main():
    import numpy as np
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.lossless import encode_block, decode_block, RECORD
    from airllm.nextgen.quantization import quantize_affine, dequant_affine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 12, 23])
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 7, 31])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    root = Path(args.model_dir).resolve()
    provenance = json.loads((root/'download-provenance.json').read_text())
    rows, probe_rows, codec_counts = [], [], {}
    rng = np.random.default_rng(20261023)
    with OffsetCheckpoint(root, ROOT/'work'/'storage-metadata') as cp:
        for layer in args.layers:
            shared = f'model.layers.{layer}.self_attn.q_proj.weight'
            names = [shared]
            for current, legacy in [('experts.gate_up_proj', 'input_linear.weight'),
                                    ('experts.down_proj', 'output_linear.weight')]:
                prefix = f'model.layers.{layer}.block_sparse_moe.'
                names.append(prefix+(current if prefix+current in cp.tensors else legacy))
            for expert in args.experts:
                matrices = {}
                for key in names:
                    # Shared sample only once per layer.
                    if key == shared and expert != args.experts[0]:
                        continue
                    selected = None if key == shared else expert
                    raw = cp.read_array(key, expert=selected)
                    if cp.tensors[key]['dtype'] != 'BF16':
                        raise ValueError('This trained-checkpoint experiment expects original BF16')
                    original = (raw.astype(np.uint32) << 16).view(np.float32)
                    identity = key if selected is None else f'{key}#{selected}'
                    raw_bytes = raw.tobytes()
                    blocks = [raw_bytes[i:i+1024**2] for i in range(0, len(raw_bytes), 1024**2)]
                    encoded = [(len(block), *encode_block(block)) for block in blocks]
                    for _, codec, _ in encoded:
                        codec_counts[str(codec)] = codec_counts.get(str(codec), 0)+1
                    def lossless_decode():
                        return b''.join(decode_block(codec, payload, size) for size,codec,payload in encoded)
                    decoded_bytes, lossless_time = measure(lossless_decode, args.repeats)
                    if decoded_bytes != raw_bytes:
                        raise AssertionError('Lossless sample did not recover exact source bytes')
                    _, bf16_time = measure(lambda: (raw.astype(np.uint32) << 16).view(np.float32), args.repeats)
                    variants = {}
                    for bits in (4, 8):
                        packed = quantize_affine(original, bits, 64)
                        decoded, decode_time = measure(lambda: dequant_affine(packed), args.repeats)
                        error = decoded.astype(np.float64)-original
                        variants[str(bits)] = {'payload_bytes': packed.nbytes,
                            'relative_weight_l2_error': float(np.linalg.norm(error)/np.linalg.norm(original)),
                            'max_abs_weight_error': float(np.max(np.abs(error))),
                            'decode': decode_time}
                        matrices[(key, bits)] = decoded
                    matrices[(key, 16)] = original
                    rows.append({'identity': identity, 'shape': list(original.shape),
                        'kind': 'shared' if selected is None else 'expert',
                        'source_dtype': 'BF16', 'source_payload_bytes': len(raw_bytes),
                        'source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
                        'original_bf16_to_fp32_decode': bf16_time,
                        'lossless': {'payload_bytes': sum(len(p) for _,_,p in encoded),
                            'index_bytes': len(encoded)*RECORD.size, 'byte_exact': True,
                            'decode_bytes_only': lossless_time}, 'affine': variants})
                gate_key, down_key = names[1:]
                x = rng.normal(0, .1, (32, matrices[(gate_key, 16)].shape[1])).astype(np.float32)
                def mlp(bits):
                    gu = x @ matrices[(gate_key, bits)].T
                    gate, up = np.split(gu, 2, axis=-1)
                    return ((gate/(1+np.exp(-gate)))*up) @ matrices[(down_key, bits)].T
                expected = mlp(16)
                for bits in (4, 8):
                    actual = mlp(bits)
                    for length in (1, 2, 4, 8, 16, 32):
                        error = actual[:length].astype(np.float64)-expected[:length]
                        probe_rows.append({'layer': layer, 'expert': expert, 'tokens': length,
                            'expert_bits': bits, 'relative_mlp_output_l2_error':
                            float(np.linalg.norm(error)/np.linalg.norm(expected[:length])),
                            'max_abs_mlp_output_error': float(np.max(np.abs(error)))})
        reader_stats = dict(cp.stats)
    expert_rows = [r for r in rows if r['kind'] == 'expert']
    shared_rows = [r for r in rows if r['kind'] == 'shared']
    raw_total = sum(r['source_payload_bytes'] for r in rows)
    totals = {'original_bf16_payload_bytes': raw_total,
        'lossless_payload_plus_index_bytes': sum(r['lossless']['payload_bytes']+r['lossless']['index_bytes'] for r in rows),
        'all_sampled_affine4_payload_bytes': sum(r['affine']['4']['payload_bytes'] for r in rows),
        'experts4_shared8_payload_bytes': sum(r['affine']['4']['payload_bytes'] for r in expert_rows)+
            sum(r['affine']['8']['payload_bytes'] for r in shared_rows),
        'sampled_expert_bf16_payload_bytes': sum(r['source_payload_bytes'] for r in expert_rows)}
    result = {'scope': 'sampled trained official Granite expert/attention matrices, local in-memory CPU microbenchmark',
        'model_id': provenance['model_id'], 'revision': provenance['revision'],
        'source_provenance': provenance, 'layers': args.layers, 'experts': args.experts,
        'activation_probe': 'seed=20261023 Gaussian mean0 std0.1, 32 rows; not hidden states from inference',
        'lossless_codec_counts': codec_counts, 'lossless_codec_names': {'0':'raw','1':'zlib','2':'shuffle-zlib','3':'palette16'},
        'group_size': 64, 'totals': totals, 'matrices': rows, 'local_mlp_error_probes': probe_rows,
        'reader_stats': reader_stats, 'platform': platform.platform(), 'python': sys.version,
        'limits': ['No full-model generation with quantized sidecars',
            'Decode timings exclude file I/O and BF16-to-FP32 after lossless reconstruction',
            'Quantized decode timings are NumPy dequantization, not native MLX quantized_matmul',
            'Payload bytes exclude safetensors JSON headers and manifests, except explicit lossless index',
            'Checkpoint fits RAM; no physical external-SSD or giant-model measurement'],
        'source_files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), ROOT/'air_llm/airllm/nextgen/sidecar.py',
                      ROOT/'air_llm/airllm/nextgen/quantization.py', ROOT/'air_llm/airllm/nextgen/lossless.py']}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
