#!/usr/bin/env python3
"""Exact checkpoint inventory plus explicitly analytical/replayed I/O bounds.

Route replay allocates no weight tensors and measures no runtime/SSD speed.
It uses real exported route ids, production cache policies, decoded-FP32
payload accounting, exact BF16/F32 source sizes, and deterministic access order.
No reload timings are supplied; adaptive modes use their equal-cost fallback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))


def main():
    import numpy as np
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.sidecar import expert_matrices
    from airllm.nextgen.cache import WeightCache
    from airllm.nextgen.adaptive_cache import AdaptiveWeightCache
    from airllm.nextgen.residency_cache import ResidencyWeightCache, RESIDENCY_POLICIES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--trace-json', type=Path)
    parser.add_argument('--cache-mib', type=int, nargs='+', default=[192, 256])
    args = parser.parse_args()
    if any(n < 0 for n in args.cache_mib):
        parser.error('cache budgets must be nonnegative')
    root = Path(args.model_dir).resolve()
    cfg = json.loads((root/'config.json').read_text())
    with OffsetCheckpoint(root, ROOT/'work'/'storage-metadata') as cp:
        matrices = list(expert_matrices(cp))
        banks = {key for key,_,_ in matrices}
        output_key = 'lm_head.weight' if 'lm_head.weight' in cp.tensors else 'model.embed_tokens.weight'
        ordinary = {k:v for k,v in cp.tensors.items() if k not in banks and k not in
                    ('model.embed_tokens.weight', output_key)}
        shared_source = sum(v['bytes'] for v in ordinary.values())
        shared_decoded = sum(math.prod(v['shape'])*4 for v in ordinary.values())
        bank_bytes = sum(cp.tensors[k]['bytes'] for k in banks)
        output_source = cp.tensors[output_key]['bytes']
        experts, topk = cfg['num_local_experts'], cfg['num_experts_per_tok']
        theoretical = []
        for length in (1, 2, 4, 8, 16, 32):
            expected_union = experts*(1-(1-topk/experts)**length)
            union_source = bank_bytes/experts*expected_union
            single_source = bank_bytes/experts*topk
            theoretical.append({'tokens': length,
                'expected_expert_union_per_layer': expected_union,
                'expert_union_lower_bound_per_layer': topk,
                'expert_union_upper_bound_per_layer': min(experts, topk*length),
                'tokenwise_uncached_source_bytes': length*(shared_source+output_source+single_source),
                'union_uncached_expected_source_bytes': shared_source+output_source+union_source,
                'union_expected_source_bytes_per_token': (shared_source+output_source+union_source)/length,
                'perfect_shared_residency_expected_expert_source_bytes_per_token': union_source/length})

        replay = []
        identical_replays = {}
        reused_replays = 0
        if args.trace_json:
            trace = json.loads(args.trace_json.read_text())
            arrays_path = args.trace_json.with_suffix('.npz')
            arrays = np.load(arrays_path)
            # Each union array contains the exact same positions as the paired
            # tokenwise/serial runs; process one array per case/repeat/length.
            for row in trace['records']:
                if row.get('kind') != 'batch' or row.get('variant') != 'union':
                    continue
                routes = arrays[row['array_key']+'_routes']
                if routes.shape != (cfg['num_hidden_layers'], row['batch_size'], topk):
                    raise ValueError(f'Unexpected exported route shape: {routes.shape}')
                if np.any(routes < 0) or np.any(routes >= experts):
                    raise ValueError('Route trace has an invalid expert id')
                route_digest = hashlib.sha256(np.sort(routes, axis=-1).tobytes()).hexdigest()
                actual_union = [len(np.unique(layer)) for layer in routes]
                union_source_bytes = 0
                by_layer = {}
                for layer in range(cfg['num_hidden_layers']):
                    selected = [(key, shape) for key,e,shape in matrices if e == 0 and f'.layers.{layer}.' in key]
                    by_layer[layer] = selected
                    union_source_bytes += sum(cp.tensors[key]['bytes']//experts for key,_ in selected)*actual_union[layer]
                replay.append({'kind': 'observed_union', 'case': row['case'], 'repeat': row['repeat'],
                    'batch_size': row['batch_size'], 'expert_union_per_layer': actual_union,
                    'mean_union_per_layer': float(np.mean(actual_union)),
                    'exact_union_expert_source_bytes': union_source_bytes,
                    'uncached_shared_plus_output_source_bytes': shared_source+output_source,
                    'uncached_total_source_bytes': union_source_bytes+shared_source+output_source,
                    'reported_runtime_logical_bytes': row['io_delta'].get('logical_bytes')})

                def layer_requests(layer, ids, cache, miss):
                    p = f'model.layers.{layer}.'
                    moep = p+'block_sparse_moe.'
                    router = moep+('router.weight' if moep+'router.weight' in cp.tensors else 'router.layer.weight')
                    for key in [p+'input_layernorm.weight', p+'self_attn.q_proj.weight',
                                p+'self_attn.k_proj.weight', p+'self_attn.v_proj.weight',
                                p+'self_attn.o_proj.weight', p+'post_attention_layernorm.weight', router]:
                        miss(key, math.prod(cp.tensors[key]['shape'])*4, cp.tensors[key]['bytes'])
                    references = [f'{key}#{int(expert)}' for expert in ids.flat for key,_ in by_layer[layer]]
                    if isinstance(cache, ResidencyWeightCache):
                        cache.set_batch(references)
                    for expert in np.unique(ids):
                        for key,shape in by_layer[layer]:
                            miss(f'{key}#{int(expert)}', math.prod(shape)*4, cp.tensors[key]['bytes']//experts)
                    if isinstance(cache, ResidencyWeightCache):
                        cache.clear_batch()

                for budget in args.cache_mib:
                    for policy in ('lru', 'slru', 'lfu', 'adaptive', *RESIDENCY_POLICIES):
                        for execution in ('tokenwise', 'union'):
                            signature = (route_digest, budget, policy, execution)
                            if signature in identical_replays:
                                replay.append(dict(identical_replays[signature], case=row['case'],
                                    repeat=row['repeat'], identical_route_replay_reused=True))
                                reused_replays += 1
                                continue
                            cache = (ResidencyWeightCache(budget*1024**2, policy) if policy in RESIDENCY_POLICIES else
                                AdaptiveWeightCache(budget*1024**2) if policy == 'adaptive' else
                                WeightCache(budget*1024**2, policy))
                            source_misses = [0]
                            def miss(key, decoded_size, source_size):
                                if cache.get(key, decoded_size) is None:
                                    source_misses[0] += source_size
                                    cache.put(key, SimpleNamespace(nbytes=decoded_size))
                                if cache.bytes > cache.capacity:
                                    raise AssertionError('Replay exceeded identical cache budget')
                            def output():
                                miss('model.norm.weight', math.prod(cp.tensors['model.norm.weight']['shape'])*4,
                                     cp.tensors['model.norm.weight']['bytes'])
                                n, width = cp.tensors[output_key]['shape']
                                for start in range(0, n, 2048):
                                    stop = min(n, start+2048)
                                    miss(('output', output_key, start, stop), (stop-start)*width*4,
                                         (stop-start)*width*2)
                            if execution == 'union':
                                for layer in range(cfg['num_hidden_layers']):
                                    layer_requests(layer, routes[layer], cache, miss)
                                output()
                            else:
                                for position in range(row['batch_size']):
                                    for layer in range(cfg['num_hidden_layers']):
                                        layer_requests(layer, routes[layer,position:position+1], cache, miss)
                                    output()
                            result_row = {'kind': 'cache_trace_replay', 'case': row['case'],
                                'repeat': row['repeat'], 'batch_size': row['batch_size'], 'cache_mib': budget,
                                'policy': policy, 'execution': execution, 'initial_software_cache': 'empty',
                                'logical_source_miss_bytes': source_misses[0], 'peak_decoded_cache_bytes': cache.peak_bytes,
                                'route_set_ordered_positions_sha256': route_digest,
                                'identical_route_replay_reused': False, 'cache_stats': dict(cache.stats)}
                            replay.append(result_row)
                            identical_replays[signature] = result_row
                print(json.dumps({'replay_case': row['case'], 'repeat': row['repeat'],
                    'batch_size': row['batch_size'], 'completed_rows': len(replay),
                    'identical_replays_reused': reused_replays}), flush=True)
        result = {'scope': 'exact Granite metadata; independent-route analytical model and optional real-route cache replay',
            'checkpoint_inventory': {'shared_nonembedding_nonoutput_source_bytes': shared_source,
                'shared_nonembedding_nonoutput_decoded_fp32_bytes': shared_decoded,
                'shared_source_excludes_embedding_rows': True, 'output_source_bytes_per_forward': output_source,
                'output_decoded_fp32_bytes': math.prod(cp.tensors[output_key]['shape'])*4,
                'all_expert_banks_source_bytes': bank_bytes,
                'expert_pair_source_bytes_per_layer': bank_bytes//(experts*cfg['num_hidden_layers']),
                'num_experts': experts, 'top_k': topk, 'layers': cfg['num_hidden_layers']},
            'analytical_assumption': 'Independent uniform top-k sets across tokens; real route correlation can strongly change union.',
            'theoretical_rows': theoretical, 'replay_rows': replay,
            'identical_route_replays_reused': reused_replays,
            'replay_limits': ['No runtime or SSD speed measurement', 'No recorded load times; equal-cost adaptive fallback',
                'Embedding-row reads and speculative prefetch excluded',
                'Output tiles enabled with 2048-row chunks; replay source precision is BF16, decoded payload FP32',
                'Initial cache empty for each run; no cross-generation warm cache',
                'Actual production order preserved for union and tokenwise; no OS-cache model'],
            'trace_json': str(args.trace_json) if args.trace_json else None,
            'trace_json_sha256': hashlib.sha256(args.trace_json.read_bytes()).hexdigest() if args.trace_json else None,
            'trace_arrays_sha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest() if args.trace_json else None,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['checkpoint_inventory'], indent=2))


if __name__ == '__main__':
    main()
