#!/usr/bin/env python3
"""Measure actual whole-bank vs selected-slice reads on a local 48 MiB MoE bank."""
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'air_llm'))
import torch
from safetensors.torch import save_file
from airllm.ultra.checkpoint import DirectCheckpoint
from airllm.ultra.cache import AdaptiveCache


def main():
    torch.set_num_threads(1)
    torch.manual_seed(521)
    experts, hidden, intermediate = 128, 128, 256
    with tempfile.TemporaryDirectory(prefix='airllm-experts-') as d:
        d = Path(d)
        tensors = {'experts.gate_up_proj': torch.randn(experts, 2*intermediate, hidden),
                   'experts.down_proj': torch.randn(experts, hidden, intermediate)}
        save_file(tensors, d / 'model.safetensors')
        selected = (3, 7, 11, 23, 31, 43, 59, 83)
        expected = sum(float(tensors['experts.down_proj'][e].sum()) for e in selected)
        del tensors
        cases = []
        for mode in ['full_bank', 'selected', 'selected_cached']:
            with DirectCheckpoint(d, max_open_files=2) as cp:
                cache = AdaptiveCache(4*1024**2)
                times = []
                for repeat in range(8):
                    start = time.perf_counter()
                    check = 0.0
                    if mode == 'full_bank':
                        w = cp.read(cp.keys(), copy=True)
                        check = sum(float(w['experts.down_proj'][e].sum()) for e in selected)
                        del w
                    else:
                        for e in selected:
                            w = cache.get(e) if mode == 'selected_cached' else None
                            if w is None:
                                w = {k: cp.read_expert_slice(k, e, torch.float32) for k in cp.keys()}
                                if mode == 'selected_cached':
                                    cache.put(e, w)
                            check += float(w['experts.down_proj'].sum())
                            del w
                    assert abs(check - expected) < 1e-6
                    times.append(time.perf_counter()-start)
                cases.append({'mode': mode, 'seconds': times,
                              'median_seconds': statistics.median(times),
                              'checkpoint_counters': dict(cp.stats), 'cache_counters': dict(cache.stats),
                              'cache_peak_bytes': cache.peak_bytes})
    report = {'platform': platform.platform(), 'experts': experts, 'active': len(selected),
              'shape': {'hidden': hidden, 'intermediate': intermediate}, 'repeats': 8,
              'all_checksums_match': True, 'runs': cases,
              'limitations': 'Real CPU tensor loads on a warm host filesystem; no SSD/Metal/GLM-5.3 timing.'}
    out = Path(__file__).with_name('results-experts.json')
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
