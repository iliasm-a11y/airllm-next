#!/usr/bin/env python3
"""CPU-only comparison of the captured router; no model checkpoint is loaded."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import mlx.core as mx
    torch.set_num_threads(1)
    arrays = np.load(args.capture)
    x, weight = arrays['x'], arrays['weight']
    native_logits, native_probs = arrays['logits'], arrays['probabilities']
    k = arrays['selected'].shape[-1]
    with torch.inference_mode():
        t_logits = F.linear(torch.from_numpy(x.reshape(-1, x.shape[-1])), torch.from_numpy(weight))
        t_probs = F.softmax(t_logits, dim=-1, dtype=torch.float32)
        from_native_logits = F.softmax(torch.from_numpy(native_logits), dim=-1, dtype=torch.float32)
    with mx.stream(mx.cpu):
        cpu_logits = mx.array(x.reshape(-1, x.shape[-1])) @ mx.array(weight).T
        mx_default = np.array(mx.softmax(mx.array(native_logits), axis=-1))
        mx_precise = np.array(mx.softmax(mx.array(native_logits), axis=-1, precise=True))
    variants = {'native_gpu': native_probs, 'torch_on_native_logits': from_native_logits.numpy(),
                'torch_full_cpu_gate': t_probs.numpy(), 'mlx_cpu_default_on_native_logits': mx_default,
                'mlx_cpu_precise_on_native_logits': mx_precise}
    result = {'kind': 'captured_router_cpu_reference_comparison', 'gpu_work': False,
              'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              'versions': {'torch': str(torch.__version__), 'mlx': importlib.metadata.version('mlx')},
              'top_k': k, 'matrix_shape': list(weight.shape), 'rows': []}
    for row in range(len(native_logits)):
        item = {'row': row,
                'native_boundary_logits': {str(e): float(native_logits[row, e]) for e in (49, 206)},
                'torch_full_boundary_logits': {str(e): float(t_logits[row, e]) for e in (49, 206)},
                'mlx_cpu_boundary_logits': {str(e): float(cpu_logits[row, e].item()) for e in (49, 206)},
                'native_vs_torch_full_max_logit_error': float(np.max(np.abs(native_logits[row] - t_logits[row].numpy()))),
                'variants': {}}
        for name, probabilities in variants.items():
            p = torch.from_numpy(np.array(probabilities[row:row+1], dtype=np.float32, copy=True))
            values, ids = torch.topk(p, k, dim=-1)
            scores = values / values.sum(dim=-1, keepdim=True)
            boundary = torch.topk(p, k + 1, dim=-1).values[0]
            item['variants'][name] = {
                'probability_49': float(p[0, 49]), 'probability_206': float(p[0, 206]),
                'probabilities_equal': bool(p[0, 49] == p[0, 206]),
                'boundary_tie': bool(boundary[k-1] == boundary[k]),
                'selected': ids[0].tolist(), 'normalized_scores': scores[0].tolist(),
                'max_probability_difference_vs_native': float(np.max(np.abs(probabilities[row] - native_probs[row])))}
        result['rows'].append(item)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
