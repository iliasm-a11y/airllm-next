#!/usr/bin/env python3
"""Paired native inference and bounded storage ablations, with honest cache state.

Fresh processes empty software caches, never the OS page cache. Every policy has
the same model, prompts, dtype, token count and memory budgets. Tiny GLM is an
exactness/performance fixture with random weights, not a giant-model speed claim.
An existing trained local checkpoint can be supplied; nothing is downloaded.
"""
import argparse
from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, process_time


VARIANTS = ('reference', 'adaptive', 'prefetch', 'specprefetch', 'coalesced', 'mmap', 'all')
OPTIONAL_VARIANTS = ('output_pipeline', 'adaptive_pipeline', 'pread_nocache',
                     'adaptive_pipeline_nocache')
MiB = 1024 ** 2


def variant_options(name, io_bytes):
    if name not in (*VARIANTS, *OPTIONAL_VARIANTS, 'baseline'):
        raise ValueError(f'Unknown variant {name}')
    options = {'policy': 'lru', 'prefetch': False, 'spec_prefetch': False,
               'spec_prefetch_bytes': 0, 'spec_prefetch_experts': 1,
               'coalesce_reads': False, 'io_mode': 'pread', 'output_prefetch': False}
    if name in ('adaptive', 'all', 'adaptive_pipeline', 'adaptive_pipeline_nocache'):
        options['policy'] = 'adaptive'
    if name in ('prefetch', 'all'):
        options['prefetch'] = True
    if name in ('specprefetch', 'all'):
        options.update(spec_prefetch=True, spec_prefetch_bytes=io_bytes // 4)
    if name in ('coalesced', 'all'):
        options['coalesce_reads'] = True
    if name in ('mmap', 'all'):
        options['io_mode'] = 'mmap'
    if name in ('output_pipeline', 'adaptive_pipeline', 'adaptive_pipeline_nocache'):
        options['output_prefetch'] = True
    if name in ('pread_nocache', 'adaptive_pipeline_nocache'):
        options['io_mode'] = 'pread_nocache'
    return options


def ordered_jobs(variants, repeats, seed, baseline_source=None):
    if repeats < 1 or not variants or len(set(variants)) != len(variants):
        raise ValueError('Unique variants and positive repeats required')
    if any(name not in (*VARIANTS, *OPTIONAL_VARIANTS) for name in variants):
        raise ValueError('Unknown benchmark variant')
    rng = random.Random(seed)
    jobs = []
    for repeat in range(repeats):
        order = list(variants) + (['baseline'] if baseline_source else [])
        rng.shuffle(order)
        jobs.extend((repeat, name) for name in order)
    return jobs


def cache_state(repeat, prompt_index):
    return ('process_cold_software_weights' if repeat == 0 and prompt_index == 0
            else 'reused_software_weights_os_cache_uncontrolled')


def source_revision(root):
    done = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def file_digest(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(MiB), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot(source_root):
    import psutil
    sys.path.insert(0, str(Path(source_root).resolve() / 'air_llm'))
    from airllm.nextgen.storage import physical_read_bytes
    from airllm.nextgen.physical_io import physical_read_counter_info
    versions = {}
    for package in ('numpy', 'mlx', 'mlx-metal', 'torch', 'transformers', 'safetensors', 'psutil', 'pytest'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hardware = {}
    if sys.platform == 'darwin':
        for name in ('hw.model', 'hw.memsize', 'machdep.cpu.brand_string', 'hw.logicalcpu', 'hw.physicalcpu'):
            result = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True)
            hardware[name] = result.stdout.strip() if result.returncode == 0 else None
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'python_executable': sys.executable,
            'package_versions': versions, 'hardware': hardware,
            'system_memory': dict(psutil.virtual_memory()._asdict()),
            'system_swap': dict(psutil.swap_memory()._asdict()),
            'physical_process_read_bytes_counter': physical_read_bytes(),
            'physical_read_counter_info': physical_read_counter_info(),
            'physical_counter_scope': 'OS process storage read counter, unavailable is null; never a bandwidth result',
            'physical_ssd_bandwidth_certified': False,
            'mlx_enable_tf32_environment': os.environ.get('MLX_ENABLE_TF32'),
            'source_revision': source_revision(source_root)}


def make_tiny(root, source_root):
    # Reuse the repository's deterministic complete native GLM test model.
    spec = importlib.util.spec_from_file_location('io_wall_tiny', source_root / 'tests_ultra/conftest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save_tiny(root, 'glm')


def _optional_memory(mx, name):
    method = getattr(mx, name, None)
    if method is None:
        return None
    try:
        return int(method())
    except (RuntimeError, ValueError):
        return None


def worker(args):
    sys.path.insert(0, str(Path(args.source_root).resolve() / 'air_llm'))
    import numpy as np
    import psutil
    import resource
    import mlx.core as mx
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.storage import physical_read_bytes
    # Baseline checkouts predating phase 2 have no macOS counter metadata.
    try:
        from airllm.nextgen.physical_io import physical_read_counter_info
        counter_info = physical_read_counter_info()
    except ImportError:
        counter_info = {'source': 'legacy Linux /proc/self/io or unavailable',
                        'scope': 'process-wide OS storage counter; unavailable is null',
                        'physical_ssd_bandwidth_certified': False}
    config = json.loads((Path(args.model_dir) / 'config.json').read_text())
    backend = args.backend
    if backend == 'auto':
        backend = ('granite' if config.get('model_type') == 'granitemoe' else
                   'llama' if config.get('model_type') == 'llama' else
                   'qwen' if str(config.get('model_type', '')).startswith('qwen') else 'glm')
    if backend == 'llama':
        from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel as Model
    elif backend == 'qwen':
        from airllm.nextgen.qwen_runtime import ExperimentalQwenModel as Model
    elif backend == 'granite':
        from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel as Model
    else:
        from airllm.nextgen import AirLLMNextModel as Model
    source_hashes = {str(path.relative_to(args.source_root)): file_digest(path)
                     for path in sorted((Path(args.source_root) / 'air_llm/airllm/nextgen').glob('*.py'))}
    if args.fixture_kind == 'tiny_glm':
        inputs = [[1, 7, 8, 9, 10],
                  [1] + [4 + (i % 92) for i in range(args.long_prompt_tokens - 1)]]
        prompt_text = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                                   trust_remote_code=False)
        prompt_text = ['What is the capital of France?', 'Write a Python function that adds two numbers.']
        inputs = [tokenizer.encode(text, add_special_tokens=True) for text in prompt_text]
        if args.extend_trained_prompt:
            filler_text = ' The following context explains a simple programming task using clear examples and ordinary language.'
            filler = tokenizer.encode(filler_text, add_special_tokens=False)
            if not filler or args.long_prompt_tokens < len(inputs[1]):
                raise ValueError('Extended prompt requires nonempty filler and a target at least as long as the base prompt')
            needed = args.long_prompt_tokens - len(inputs[1])
            inputs[1] += (filler * ((needed + len(filler) - 1) // len(filler)))[:needed]
    if max(map(len, inputs)) + args.tokens > args.max_context:
        raise ValueError('Prompt plus fixed generation count exceeds max_context')
    shared = dict(device=args.device, dtype=args.dtype, cache_bytes=args.cache_bytes,
                  io_bytes=args.io_bytes, max_tensor_bytes=args.max_tensor_bytes,
                  kv_bytes=args.kv_bytes, workspace_bytes=args.workspace_bytes,
                  reserve_bytes=args.reserve_bytes, max_context=args.max_context,
                  row_chunk=args.row_chunk, io_workers=args.io_workers,
                  use_saved_tuning=False, load_tokenizer=False, profile_io=args.profile_io)
    options = variant_options(args.baseline_variant if args.worker == 'baseline'
                              else args.worker, args.io_bytes)
    supported = {field.name for field in fields(NextConfig)}
    omitted = {key: value for key, value in (shared | options).items() if key not in supported}
    if args.worker != 'baseline' and omitted:
        raise RuntimeError(f'Current source lacks requested fields: {sorted(omitted)}')
    with tempfile.TemporaryDirectory(prefix='airllm-io-wall-meta-') as metadata:
        cfg = NextConfig(**{key: value for key, value in (shared | options).items()
                            if key in supported}, cache_dir=metadata)
        start = perf_counter()
        with Model(args.model_dir, next_config=cfg) as model:
            startup_seconds = perf_counter() - start
            rows = []
            logits = {}
            process = psutil.Process()
            for repeat in range(args.warm_repeats + 1):
                for index, ids in enumerate(inputs):
                    cp = model.checkpoint
                    before = dict(cp.stats)
                    physical_start = physical_read_bytes()
                    swap_start = psutil.swap_memory().used
                    rss_start = process.memory_info().rss
                    current = np.array([ids], dtype=np.int32)
                    output = list(ids)
                    cache = None
                    step_seconds = []
                    case_logits = []
                    rss_samples = [rss_start]
                    mlx_samples = []
                    start = perf_counter(); cpu_start = process_time()
                    for step in range(args.tokens):
                        step_start = perf_counter()
                        result = model(current, past_key_values=cache, logits_to_keep=1)
                        cache = result.past_key_values
                        last = result.logits[:, -1, :]
                        # Force evaluation of both timing and correctness data.
                        mx.eval(last)
                        values = np.asarray(last).copy()
                        token = int(np.argmax(values[0]))
                        output.append(token)
                        case_logits.append(values)
                        step_seconds.append(perf_counter() - step_start)
                        current = np.array([[token]], dtype=np.int32)
                        rss_samples.append(process.memory_info().rss)
                        mlx_samples.append(_optional_memory(mx, 'get_active_memory'))
                    # Finish speculative I/O before charging a complete request.
                    getattr(model.reader, 'drain', model.reader.io.drain)()
                    seconds = perf_counter() - start
                    cpu_seconds = process_time() - cpu_start
                    physical_end = physical_read_bytes()
                    swap_end = psutil.swap_memory().used
                    name = f'repeat_{repeat}_prompt_{index}'
                    values = np.concatenate(case_logits, axis=0)
                    logits[name] = values
                    delta = {key: value - before.get(key, 0) for key, value in cp.stats.items()
                             if isinstance(value, (int, float))}
                    rows.append({'case': name, 'repeat': repeat, 'prompt_index': index,
                        'cache_state': cache_state(repeat, index),
                        'input_ids': ids, 'output_ids': output, 'new_tokens': args.tokens,
                        'generation_seconds_including_drain_and_sampling': seconds,
                        'cpu_process_seconds_including_all_threads': cpu_seconds,
                        'cpu_process_scope': 'process CPU time; includes Python, IO/decode workers and CPU kernels',
                        'gpu_kernel_seconds': None, 'gpu_wait_seconds': None,
                        'route_readback_seconds': None,
                        'prefill_seconds': step_seconds[0], 'decode_step_seconds': step_seconds[1:],
                        'decode_tokens_per_second': ((args.tokens - 1) / sum(step_seconds[1:])
                                                     if args.tokens > 1 else None),
                        'counter_delta': delta,
                        'physical_process_read_bytes': (physical_end - physical_start
                            if physical_start is not None and physical_end is not None else None),
                        'rss_start_bytes': rss_start, 'rss_peak_sampled_bytes': max(rss_samples),
                        'mlx_active_peak_sampled_bytes': max((n for n in mlx_samples if n is not None), default=None),
                        'swap_system_start_bytes': swap_start, 'swap_system_end_bytes': swap_end,
                        'swap_system_growth_bytes': max(0, swap_end - swap_start),
                        'logits_sha256': sha256(values.tobytes()).hexdigest()})
            profile = getattr(cp, 'io_profile', lambda: None)()
            metrics = model.metrics()
            report = {'variant': args.worker, 'config': cfg.as_dict(), 'omitted_baseline_fields': omitted,
                'source_root': str(Path(args.source_root).resolve()),
                'source_revision': source_revision(args.source_root),
                'baseline_variant': args.baseline_variant if args.worker == 'baseline' else None,
                'source_files_sha256_at_worker_start': source_hashes,
                'startup_seconds': startup_seconds, 'device': model.device, 'backend': backend,
                'physical_read_counter_info': counter_info,
                'fixture_kind': args.fixture_kind, 'prompt_text': prompt_text,
                'trained_prompt_extension': ({'scope': 'synthetic repeated natural-language token context',
                    'second_prompt_target_tokens': args.long_prompt_tokens,
                    'filler_text': filler_text} if args.fixture_kind != 'tiny_glm' and args.extend_trained_prompt else None),
                'rows': rows,
                'io_profile': profile, 'final_metrics': metrics,
                'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                          (1 if sys.platform == 'darwin' else 1024),
                'mlx_peak_memory_bytes': _optional_memory(mx, 'get_peak_memory'),
                'os_page_cache_state': (
                    'F_NOCACHE requested for payload descriptors; existing OS cache was not purged'
                    if cfg.io_mode == 'pread_nocache' else
                    'uncontrolled; a fresh process is not a cold SSD read'),
                'physical_ssd_bandwidth_certified': False}
    final_hashes = {str(path.relative_to(args.source_root)): file_digest(path)
                    for path in sorted((Path(args.source_root) / 'air_llm/airllm/nextgen').glob('*.py'))}
    report['source_files_sha256_at_worker_end'] = final_hashes
    report['source_unchanged_during_worker'] = final_hashes == source_hashes
    report['benchmark_script_sha256'] = file_digest(__file__)
    destination = Path(args.output)
    np.savez_compressed(destination.with_suffix('.npz'), **logits)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    if not report['source_unchanged_during_worker']:
        raise RuntimeError('Native source changed during the worker; diagnostic artifacts retained')


def compare_reports(reports, artifacts, *, atol=4e-6, rtol=2e-4):
    import numpy as np
    reference = next((row for row in reports if row['variant'] == 'baseline'), None)
    if reference is None:
        reference = next((row for row in reports if row['variant'] == 'reference'), None)
    if reference is None:
        raise ValueError('A reference or baseline variant is required for correctness')
    expected = {row['case']: row for row in reference['rows']}
    comparisons = []
    with np.load(artifacts[reference['artifact']]) as ref:
        for report in reports:
            exact = close = sequences = True
            max_error = 0.
            with np.load(artifacts[report['artifact']]) as observed:
                if set(observed.files) != set(ref.files):
                    raise AssertionError('Unequal benchmark cases')
                for row in report['rows']:
                    a, b = observed[row['case']], ref[row['case']]
                    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise AssertionError('Invalid logits shape or nonfinite logits')
                    exact = exact and bool(np.array_equal(a, b))
                    close = close and bool(np.allclose(a, b, atol=atol, rtol=rtol))
                    sequences = sequences and row['output_ids'] == expected[row['case']]['output_ids']
                    max_error = max(max_error, float(np.max(np.abs(a - b))))
            comparisons.append({'artifact': report['artifact'], 'variant': report['variant'],
                                'identical_logits': exact, 'logits_allclose': close,
                                'same_greedy_sequences': sequences, 'max_abs_logit_error': max_error})
    if not all(row['logits_allclose'] and row['same_greedy_sequences'] for row in comparisons):
        raise AssertionError(f'Inference equivalence failed: {comparisons}')
    return comparisons


def storage_comparison(root, repeats, seed):
    spec = importlib.util.spec_from_file_location('io_wall_storage', Path(__file__).with_name('benchmark_ooc.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.make_fixture(root, tensors=32, tensor_bytes=16384)
    return module.run_comparison(root, repeats=repeats, group_size=8,
                                 max_bytes=MiB, max_read_bytes=MiB,
                                 cache_state='warm', seed=seed)


def run_suite(args):
    source = Path(args.source_root).resolve()
    if args.baseline_source:
        # Workers change cwd to this checkout: resolve once in the caller,
        # otherwise a relative source/PYTHONPATH gets interpreted twice.
        args.baseline_source = str(Path(args.baseline_source).resolve(strict=True))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + '-runs')
    work.mkdir(exist_ok=True)
    model = Path(args.model_dir).resolve() if args.model_dir else work / 'tiny-glm'
    fixture_kind = args.fixture_kind or ('trained_local' if args.model_dir else 'tiny_glm')
    if not args.model_dir:
        if model.exists():
            raise FileExistsError('Use a new output name or supply the existing model with --model-dir')
        make_tiny(model, source)
    variants = args.variants.split(',')
    jobs = ordered_jobs(variants, args.repeats, args.seed, args.baseline_source)
    reports, artifacts = [], {}
    for order, (repeat, variant) in enumerate(jobs):
        destination = work / f'{order:03d}-round{repeat}-{variant}.json'
        current_source = args.baseline_source if variant == 'baseline' else source
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                   '--source-root', str(current_source), '--model-dir', str(model),
                   '--fixture-kind', fixture_kind, '--output', str(destination)]
        for name in ('tokens', 'warm_repeats', 'cache_bytes', 'io_bytes', 'max_tensor_bytes',
                     'kv_bytes', 'workspace_bytes', 'reserve_bytes', 'max_context',
                     'row_chunk', 'io_workers', 'device', 'dtype', 'backend', 'long_prompt_tokens',
                     'baseline_variant'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        if args.profile_io:
            command.append('--profile-io')
        if args.extend_trained_prompt:
            command.append('--extend-trained-prompt')
        done = subprocess.run(command, capture_output=True, text=True, timeout=args.worker_timeout,
            cwd=current_source, env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1',
                                         OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(Path(current_source) / 'air_llm')))
        destination.with_suffix('.log').write_text(done.stdout + done.stderr)
        if done.returncode:
            raise RuntimeError(f'Worker {variant} failed: {destination.with_suffix(".log")}')
        report = json.loads(destination.read_text())
        report.update(order=order, round=repeat, artifact=destination.name)
        reports.append(report)
        artifacts[destination.name] = destination.with_suffix('.npz')
    comparisons = compare_reports(reports, artifacts)
    summaries = {}
    for variant in dict.fromkeys(row['variant'] for row in reports):
        runs = [row for row in reports if row['variant'] == variant]
        cold = [row for run in runs for row in run['rows'] if row['cache_state'] == 'process_cold_software_weights']
        warm = [row for run in runs for row in run['rows'] if row['repeat'] > 0]
        summaries[variant] = {'process_cold_median_seconds': statistics.median(row['generation_seconds_including_drain_and_sampling'] for row in cold),
            'warm_median_seconds': statistics.median(row['generation_seconds_including_drain_and_sampling'] for row in warm),
            'warm_median_logical_bytes': statistics.median(row['counter_delta'].get('logical_bytes', 0) for row in warm),
            'warm_median_read_calls': statistics.median(row['counter_delta'].get('read_calls', 0) for row in warm),
            'peak_process_rss_bytes': max(run['process_peak_rss_bytes'] for run in runs),
            'warm_generation_count': len(warm)}
        summaries[variant]['by_prompt'] = {
            str(index): {'input_tokens': len(next(row['input_ids'] for row in warm if row['prompt_index'] == index)),
                         'warm_median_seconds': statistics.median(row['generation_seconds_including_drain_and_sampling'] for row in warm if row['prompt_index'] == index),
                         'warm_median_logical_bytes': statistics.median(row['counter_delta'].get('logical_bytes', 0) for row in warm if row['prompt_index'] == index),
                         'warm_median_read_calls': statistics.median(row['counter_delta'].get('read_calls', 0) for row in warm if row['prompt_index'] == index)}
            for index in sorted({row['prompt_index'] for row in warm})}
    reference = summaries.get('baseline', summaries.get('reference'))
    for summary in summaries.values():
        summary['warm_speedup_vs_reference'] = reference['warm_median_seconds'] / summary['warm_median_seconds']
        for index, prompt in summary['by_prompt'].items():
            prompt['warm_speedup_vs_reference'] = reference['by_prompt'][index]['warm_median_seconds'] / prompt['warm_median_seconds']
    shards = sorted(model.glob('*.safetensors'))
    report = {'schema_version': 1, 'platform': platform.platform(), 'python': sys.version,
        'seed': args.seed, 'source_revision': source_revision(source),
        'baseline_revision': source_revision(args.baseline_source) if args.baseline_source else None,
        'baseline_variant': args.baseline_variant if args.baseline_source else None,
        'model_path': str(model), 'fixture_kind': fixture_kind,
        'model_shards_sha256': {path.name: file_digest(path) for path in shards},
        'model_serialized_bytes': sum(path.stat().st_size for path in shards),
        'scope': 'native full tiny/random GLM or supplied trained local model; no large-model speed claim',
        'protocol': 'randomized fresh worker per variant/round, first run software-cold, later repeated prompts warm; OS cache uncontrolled',
        'profiling_enabled': args.profile_io,
        'baseline_instrumentation_note': 'baseline lacks opt-in interval profiler; use --profile-io only for diagnostic runs' if args.baseline_source else None,
        'correctness': comparisons, 'correctness_atol': 4e-6, 'correctness_rtol': 2e-4,
        'summaries': summaries, 'physical_ssd_bandwidth_certified': False,
        'worker_artifacts': [row['artifact'] for row in reports],
        'current_source_hashes_consistent_across_workers': all(
            row['source_files_sha256_at_worker_start'] == next(
                r['source_files_sha256_at_worker_start'] for r in reports if r['variant'] != 'baseline')
            for row in reports if row['variant'] != 'baseline'),
        'storage': storage_comparison(work / 'storage-fixture', args.repeats, args.seed)}
    output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['current_source_hashes_consistent_across_workers']:
        raise RuntimeError('Native source changed between workers; diagnostic report retained')
    print(json.dumps({'output': str(output), 'summaries': summaries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--baseline-source')
    parser.add_argument('--baseline-variant', choices=(*VARIANTS, *OPTIONAL_VARIANTS),
                        default='reference', help='Options used by the pinned baseline worker')
    parser.add_argument('--model-dir')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--worker', choices=(*VARIANTS, *OPTIONAL_VARIANTS, 'baseline'))
    parser.add_argument('--backend', choices=('auto', 'glm', 'llama', 'qwen', 'granite'), default='auto')
    parser.add_argument('--fixture-kind', choices=('tiny_glm', 'trained_local'))
    parser.add_argument('--long-prompt-tokens', type=int, default=48)
    parser.add_argument('--extend-trained-prompt', action='store_true',
                        help='Extend trained model second prompt to --long-prompt-tokens using repeated local text')
    parser.add_argument('--environment-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--warm-repeats', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--worker-timeout', type=int, default=300)
    parser.add_argument('--profile-io', action='store_true')
    for name, default in (('cache-bytes', 65536), ('io-bytes', 4 * MiB),
                          ('max-tensor-bytes', 4 * MiB), ('kv-bytes', 4 * MiB),
                          ('workspace-bytes', 32 * MiB), ('reserve-bytes', 1024 * MiB),
                          ('max-context', 128), ('row-chunk', 256), ('io-workers', 2)):
        parser.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args()
    if args.environment_only:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(environment_snapshot(args.source_root), indent=2) + '\n')
        return
    for name in ('tokens', 'warm_repeats', 'repeats', 'worker_timeout', 'long_prompt_tokens'):
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if args.worker:
        if not args.model_dir:
            parser.error('--worker requires --model-dir')
        worker(args)
    else:
        run_suite(args)


if __name__ == '__main__':
    main()
