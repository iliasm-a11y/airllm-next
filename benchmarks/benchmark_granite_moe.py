#!/usr/bin/env python3
"""Pinned trained Granite MoE: official FP32 layers versus bounded native MLX.

No implicit download. The independent reference constructs official Transformers
modules on meta, loads each group from safetensors before its forward, then
releases it back to meta. It never uses AirLLM checkpoint/weight/math code.
Both implementations compute FP32 from the same original BF16 weights. Runtime
comparisons include different loading strategies, so are not kernel speedups.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ['What is the capital of France?', 'Complete the sequence: one, two, three,']
ATOL, RTOL = 1e-4, 5e-4


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'head': head.stdout.strip() if head.returncode == 0 else None,
        'files_sha256': {str(p.relative_to(ROOT)):digest(p) for p in
            sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py'))+[Path(__file__).resolve()]}}


def original_key(name, available, tied_head=False):
    """IBM legacy aliases, identical to official Transformers renaming rules."""
    if name == 'lm_head.weight' and name not in available and tied_head:
        return 'model.embed_tokens.weight'
    aliases = (('experts.gate_up_proj', 'input_linear.weight'),
               ('experts.down_proj', 'output_linear.weight'),
               ('block_sparse_moe.router.weight', 'block_sparse_moe.router.layer.weight'))
    if name in available:
        return name
    for current, legacy in aliases:
        candidate = name.replace(current, legacy)
        if candidate in available:
            return candidate
    raise ValueError(f'Missing reference checkpoint key: {name}')


def build_official_streamed_reference(root):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if cfg.model_type != 'granitemoe' or getattr(cfg, 'quantization_config', None):
        raise ValueError('Reference implements floating granitemoe checkpoints only')
    cfg._attn_implementation = 'eager'
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False, dtype=torch.float32)
    model.eval()
    with safe_open(root/'model.safetensors', framework='pt', device='cpu') as source:
        available = set(source.keys())
    groups = [('model.embed_tokens', model.model.embed_tokens)]
    groups += [(f'model.layers.{i}', layer) for i, layer in enumerate(model.model.layers)]
    groups += [('model.norm', model.model.norm), ('lm_head', model.lm_head)]
    mapping, handles, routes = {}, [], {}
    stats = {'groups_loaded': 0, 'source_encoded_tensor_bytes': 0, 'largest_decoded_group_bytes': 0}
    for prefix, module in groups:
        parameters = [(name, original_key(prefix+'.'+name, available, cfg.tie_word_embeddings))
                      for name, _ in module.named_parameters()]
        mapping.update({prefix+'.'+name: key for name,key in parameters})
        def load(group, inputs, parameters=parameters):
            decoded = 0
            with safe_open(root/'model.safetensors', framework='pt', device='cpu') as source:
                for name, key in parameters:
                    raw = source.get_tensor(key)
                    stats['source_encoded_tensor_bytes'] += raw.numel()*raw.element_size()
                    tensor = raw.to(dtype=torch.float32)
                    decoded += tensor.numel()*tensor.element_size()
                    set_module_tensor_to_device(group, name, 'cpu', value=tensor)
                    del raw, tensor
            stats['groups_loaded'] += 1
            stats['largest_decoded_group_bytes'] = max(stats['largest_decoded_group_bytes'], decoded)
        def unload(group, inputs, output):
            group.to_empty(device='meta')
        handles += [module.register_forward_pre_hook(load), module.register_forward_hook(unload, always_call=True)]
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.block_sparse_moe.router.register_forward_hook(
            lambda module, inputs, output, i=i: routes.__setitem__(i, output[0].detach().numpy().copy())))
    return model, routes, mapping, stats, handles


def worker(args):
    import numpy as np
    import psutil
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT/'air_llm'))
    from airllm.nextgen.physical_io import physical_read_bytes, physical_read_counter_info
    identity = source_identity()
    root = Path(args.model_dir).resolve()
    provenance = json.loads((root/'download-provenance.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    cases, arrays, mapping, handles = [], {}, None, []
    begin = time.perf_counter()
    if args.worker == 'reference':
        model, routes, mapping, reference_stats, handles = build_official_streamed_reference(root)
    else:
        import mlx.core as mx
        from airllm.nextgen.config import NextConfig
        from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
        mx.set_default_device(getattr(mx, args.device))
        cfg = NextConfig(device=args.device, dtype='float32', cache_bytes=args.cache_mib*1024**2,
            io_bytes=32*1024**2, kv_bytes=32*1024**2, workspace_bytes=128*1024**2,
            reserve_bytes=1536*1024**2, max_tensor_bytes=32*1024**2, row_chunk=2048,
            max_context=128, prefill_chunk=64, prefetch=args.worker != 'native_off',
            spec_prefetch=args.worker == 'native_spec', spec_prefetch_bytes=8*1024**2,
            spec_prefetch_experts=1, load_tokenizer=False, use_saved_tuning=False,
            cache_dir=args.metadata_dir, policy='adaptive')
        model = AirLLMNextGraniteMoeModel(root, next_config=cfg)
    construction = time.perf_counter()-begin
    swap_before = psutil.swap_memory().used
    for case, prompt in enumerate(PROMPTS[:args.prompts]):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list):
            ids = ids.input_ids
        if len(ids) > 64:
            raise ValueError('Prompt exceeds single-chunk routing comparison limit')
        repeats = []
        for repeat in range(args.repeats):
            tokens, cache, ttft, before = ids[:], None, None, time.perf_counter()
            physical_before = physical_read_bytes()
            if args.worker != 'reference':
                io_before = dict(model.checkpoint.stats)
            for step in range(args.tokens):
                current = [tokens] if step == 0 else [[tokens[-1]]]
                if args.worker == 'reference':
                    with torch.inference_mode():
                        output = model(torch.tensor(current), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    logits = output.logits.detach().numpy()
                    decisions = routes
                else:
                    output = model(current, past_key_values=cache, logits_to_keep=1)
                    logits = np.asarray(output.logits)
                    decisions = {i: mlp.last_route_ids for i,mlp in enumerate(model.mlps)}
                cache = output.past_key_values
                tokens.append(int(np.argmax(logits[0, -1])))
                arrays[f'c{case}_r{repeat}_s{step}_logits'] = logits.copy()
                for layer, route in decisions.items():
                    arrays[f'c{case}_r{repeat}_s{step}_route{layer}'] = np.sort(route, axis=-1)
                if step == 0:
                    ttft = time.perf_counter()-before
            elapsed = time.perf_counter()-before
            physical_after = physical_read_bytes()
            row = {'repeat': repeat, 'seconds': elapsed, 'ttft_seconds': ttft,
                'decode_tokens_per_second': (args.tokens-1)/(elapsed-ttft) if args.tokens > 1 else None,
                'input_tokens': len(ids), 'new_tokens': len(tokens)-len(ids), 'token_ids': tokens,
                'text': tokenizer.decode(tokens[len(ids):], skip_special_tokens=True),
                'process_physical_read_bytes': physical_after-physical_before
                    if physical_before is not None and physical_after is not None else None,
                'physical_counter': physical_read_counter_info(),
                'physical_counter_scope': 'whole worker process; includes all files, no per-file/device attribution',
                'io_policy': 'safetensors mmap group loading' if args.worker == 'reference' else cfg.io_mode}
            if args.worker != 'reference':
                row['kv_cache'] = {'bytes': cache.bytes, 'peak_bytes': cache.peak_bytes,
                    'budget_bytes': cfg.kv_bytes, 'committed_tokens': cache.position}
            if args.worker != 'reference':
                row['io_delta'] = {k:v-io_before.get(k,0) for k,v in model.checkpoint.stats.items()
                    if isinstance(v,(int,float)) and not isinstance(v,bool)}
                row['metrics'] = model.metrics()
            repeats.append(row)
            print(f'{args.worker}: case {case+1}, repeat {repeat+1}, {elapsed:.3f}s', flush=True)
        cases.append({'prompt': prompt, 'runs': repeats})
    for handle in handles:
        handle.remove()
    if hasattr(model, 'close'):
        model.close(save_profile=False)
    path = Path(args.output)
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    report = {'variant': args.worker, 'device': 'cpu' if args.worker == 'reference' else args.device,
        'compute_dtype': 'float32', 'checkpoint_dtype': 'bfloat16', 'construction_seconds': construction,
        'cases': cases, 'torch': torch.__version__, 'python': sys.version, 'platform': platform.platform(),
        'libraries': {name:importlib.metadata.version(name) for name in ('torch','transformers','mlx','safetensors','numpy','psutil')},
        'host_total_memory_bytes': psutil.virtual_memory().total,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        'swap_used_before_bytes': swap_before, 'swap_used_after_bytes': psutil.swap_memory().used,
        'npz_file': path.with_suffix('.npz').name, 'npz_sha256': digest(path.with_suffix('.npz')),
        'reference_key_mapping': mapping,
        'reference_loading': reference_stats if args.worker == 'reference' else None,
        'source': identity, 'source_unchanged_at_worker_end': identity == source_identity(),
        'checkpoint_identity': {'model_id': provenance['model_id'], 'revision': provenance['revision'],
            'model_config_sha256': digest(root/'config.json'),
            'weight_file_bytes': (root/'model.safetensors').stat().st_size,
            'weight_sha256_verified_by_controller': next(f['sha256'] for f in provenance['files']
                if f['name'] == 'model.safetensors')},
        'arguments': {k:v for k,v in vars(args).items() if k != 'metadata_dir'},
        'native_options': None if args.worker == 'reference' else cfg.as_dict(),
        'os_page_cache': 'uncontrolled; checkpoint hash/download can warm pages; no cold-disk claim',
        'external_ssd_measured': False, 'full_model_payload_exceeds_host_ram': False,
        'comparison_tolerance_predeclared': {'atol': ATOL, 'rtol': RTOL},
        'timing_scope': 'manual exact greedy cached decoding, including routing trace copies and logits readback'}
    path.write_text(json.dumps(report, indent=2)+'\n')


def run_guarded(command, path, args):
    import psutil
    peak = 0
    available_min = psutil.virtual_memory().available
    if available_min < args.reserve_gib*1024**3:
        raise MemoryError('Available system memory below starting reserve')
    with path.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        monitor = psutil.Process(process.pid)
        violation = None
        while process.poll() is None:
            try:
                rss = monitor.memory_info().rss
                available = psutil.virtual_memory().available
                peak, available_min = max(peak, rss), min(available_min, available)
                if rss > args.max_rss_gib*1024**3 or available < args.reserve_gib*1024**3:
                    violation = {'rss_bytes': rss, 'available_bytes': available}
                    process.terminate()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.05)
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
    guard = {'sampled_peak_rss_bytes': peak, 'minimum_available_system_bytes': available_min,
             'rss_limit_bytes': int(args.max_rss_gib*1024**3), 'reserve_bytes': int(args.reserve_gib*1024**3),
             'exit_code': code, 'violation': violation, 'sampling_interval_seconds': .05}
    path.with_suffix('.guard.json').write_text(json.dumps(guard, indent=2)+'\n')
    if code or violation:
        raise RuntimeError(f'Worker failed or memory guard stopped it; see {path.with_suffix(".log")}')
    return guard


def compare(paths):
    import numpy as np
    reports = {v:json.loads(p.read_text()) for v,p in paths.items()}
    if any(not report['source_unchanged_at_worker_end'] for report in reports.values()):
        raise ValueError('Source changed during a worker; rerun with a stable source tree')
    if len({json.dumps(report['source'], sort_keys=True) for report in reports.values()}) != 1:
        raise ValueError('Workers ran with different source identities')
    if len({json.dumps(report['checkpoint_identity'], sort_keys=True) for report in reports.values()}) != 1:
        raise ValueError('Workers ran with different checkpoint identities')
    summary = {}
    with np.load(paths['reference'].with_suffix('.npz')) as expected:
        native_base = None
        for variant, path in paths.items():
            exact_routes, close, errors, bitwise = [], [], [], []
            with np.load(path.with_suffix('.npz')) as actual:
                if set(actual.files) != set(expected.files):
                    raise ValueError('Missing comparison tensors')
                for key in expected.files:
                    a, b = actual[key], expected[key]
                    if '_logits' in key:
                        close.append(bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)))
                        errors.append(float(np.max(np.abs(a-b))))
                    else:
                        exact_routes.append(bool(np.array_equal(a,b)))
                if native_base is not None and variant != 'reference':
                    with np.load(native_base.with_suffix('.npz')) as base:
                        bitwise = [bool(np.array_equal(actual[k],base[k])) for k in actual.files]
            if variant != 'reference' and native_base is None:
                native_base = path
            report = reports[variant]
            sequences = [run['token_ids'] for c in report['cases'] for run in c['runs']]
            reference_sequences = [run['token_ids'] for c in reports['reference']['cases'] for run in c['runs']]
            times = [run['seconds'] for c in report['cases'] for run in c['runs']]
            summary[variant] = {'tokens_match_reference': sequences == reference_sequences,
                'all_topk_expert_sets_match_reference': all(exact_routes),
                'logits_within_predeclared_tolerance': all(close), 'max_abs_logit_error': max(errors),
                'bitwise_equal_first_native_variant': all(bitwise) if bitwise else None,
                'median_generation_seconds': statistics.median(times),
                'process_peak_rss_bytes': report['process_peak_rss_bytes'], 'raw_report': path.name}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prompts', type=int, choices=[1,2], default=2)
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--device', choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--max-rss-gib', type=float, default=5.)
    parser.add_argument('--reserve-gib', type=float, default=1.5)
    parser.add_argument('--variants', nargs='+', choices=['reference','native_off','native','native_spec'],
                        default=['reference','native_off','native','native_spec'])
    parser.add_argument('--worker', choices=['reference','native_off','native','native_spec'], help=argparse.SUPPRESS)
    parser.add_argument('--metadata-dir', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 1 or args.cache_mib < 0 or args.max_rss_gib <= 0 or args.reserve_gib < 0:
        parser.error('Invalid count or memory guard')
    if args.worker:
        return worker(args)
    if 'reference' not in args.variants:
        parser.error('The independent reference is mandatory')
    root, path = Path(args.model_dir).resolve(), Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((root/'download-provenance.json').read_text())
    observed = None
    for entry in provenance['files']:
        file = root/entry['name']
        if file.stat().st_size != entry['bytes'] or digest(file) != entry['sha256']:
            raise ValueError(f'Checkpoint file identity mismatch: {entry["name"]}')
        if entry['name'] == 'model.safetensors':
            observed = entry['sha256']
    if observed is None:
        raise ValueError('Checkpoint provenance omits model.safetensors')
    paths, guards = {}, {}
    for variant in args.variants:
        child = path.with_name(path.stem+'-'+variant+'.json')
        print('running', variant, flush=True)
        with tempfile.TemporaryDirectory(prefix='airllm-granite-') as metadata:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                '--model-dir', str(root), '--output', str(child), '--tokens', str(args.tokens),
                '--repeats', str(args.repeats), '--prompts', str(args.prompts), '--device', args.device,
                '--cache-mib', str(args.cache_mib), '--metadata-dir', metadata]
            guards[variant] = run_guarded(command, child, args)
        paths[variant] = child
    summary = compare(paths)
    result = {'model_id': provenance['model_id'], 'revision': provenance['revision'],
        'checkpoint_bytes': (root/'model.safetensors').stat().st_size, 'checkpoint_sha256': observed,
        'provenance': provenance, 'variant_order': args.variants, 'tokens': args.tokens,
        'repeats': args.repeats, 'prompts': PROMPTS[:args.prompts], 'cache_mib': args.cache_mib,
        'reference': 'official Transformers FP32 layers loaded/unloaded through independent safetensors hooks',
        'tolerance_predeclared': {'atol': ATOL, 'rtol': RTOL}, 'summary': summary, 'guards': guards,
        'os_page_cache': 'uncontrolled; hashing before execution warms checkpoint pages',
        'physical_external_ssd_measured': False, 'giant_model_validated': False,
        'source_files_sha256': {str(p.relative_to(ROOT)):digest(p) for p in
            sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py'))+[Path(__file__).resolve()]}}
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if any(not r['tokens_match_reference'] or not r['all_topk_expert_sets_match_reference'] or
           not r['logits_within_predeclared_tolerance'] or r['bitwise_equal_first_native_variant'] is False
           for r in summary.values()):
        raise SystemExit('Validation mismatch; evidence preserved, no tolerance changed')


if __name__ == '__main__':
    main()
