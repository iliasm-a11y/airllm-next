#!/usr/bin/env python3
"""Fresh-worker, guarded research comparisons; no model download or cache purge.

The parent always uses this checkout's supervisor. A worker may import an exact
historical checkout, so unrelated guard fixes do not contaminate engine timing.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def string_buffer_size(value):
    number = int(value)
    if not 0 <= number <= 65536:
        raise argparse.ArgumentTypeError('String buffer must be between 0 and 65536 bytes')
    return number


def router_reference_options(value):
    """Keep unsupported new keywords away from the retained historical engines."""
    if type(value) is not int or value not in (0,16):
        raise ValueError('router_reference_chunk must be 0 or 16')
    return {} if value == 0 else dict(router_reference_chunk=value)


def save(path, value):
    from tempfile import NamedTemporaryFile
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as f:
            temporary = f.name
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def source_manifest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'air_llm/airllm/nextgen').glob('*.py'))}


def engine_manifest(root):
    root = Path(root).resolve()
    sources = source_manifest(root)
    if not sources:
        raise ValueError('Engine root contains no nextgen Python sources: ' + str(root))
    return dict(git_head=subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=sources)


def protocol_manifest(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def protocol_unchanged(expected):
    try:
        return bool(expected) and protocol_manifest(expected) == expected
    except OSError:
        return False


def load_task_snapshot(path):
    raw = Path(path).read_bytes()
    tasks = json.loads(raw)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Research tasks must be a nonempty list')
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Every research task must be an object')
        ident, prompt, count = task.get('id'), task.get('prompt'), task.get('max_new_tokens')
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise ValueError('Research task IDs must be unique nonempty strings')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Research task prompts must be nonempty strings')
        if type(count) is not int or count < 1:
            raise ValueError('Research task max_new_tokens must be a positive integer')
        seen.add(ident)
    return tasks, hashlib.sha256(raw).hexdigest()


def load_tasks(path):
    return load_task_snapshot(path)[0]


def worker(args):
    tasks, tasks_hash = load_task_snapshot(args.tasks)
    protocol = (json.loads(args.expected_protocol_json) if args.expected_protocol_json
                else protocol_manifest([Path(__file__), args.tasks,
                    ROOT / 'benchmarks/hyperstream2_observer.py',
                    ROOT / 'air_llm/airllm/nextgen/request_trace.py']))
    if not isinstance(protocol, dict) or not protocol_unchanged(protocol):
        raise RuntimeError('Research protocol changed before worker startup')
    if protocol.get(str(args.tasks)) != tasks_hash:
        raise RuntimeError('Loaded tasks differ from the frozen protocol')
    initial_engine = engine_manifest(args.engine_root)
    if (args.expected_engine_head is not None and initial_engine['git_head'] != args.expected_engine_head
        or args.expected_engine_sources_json is not None and initial_engine['source_sha256'] != json.loads(args.expected_engine_sources_json)):
        raise RuntimeError('Engine differs from its frozen campaign manifest')
    sys.path.insert(0, str(args.engine_root / 'air_llm'))
    import importlib.metadata
    import numpy as np
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.experiment_guard import system_snapshot
    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.gguf_chat import GGUFChatSession
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
    from airllm.nextgen.physical_io import physical_read_bytes

    stat = args.model.stat()
    cpu_start=resource.getrusage(resource.RUSAGE_SELF)
    sources = initial_engine['source_sha256']
    report = dict(status='started', label=args.label, command=sys.argv,
                  git_head=initial_engine['git_head'], engine_root=str(args.engine_root),
                  source_sha256=sources, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=protocol, tasks_sha256=tasks_hash, tasks_snapshot=tasks,
                  hardware=dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                                processor=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()),
                  versions={k: importlib.metadata.version(k) for k in ('mlx', 'numpy', 'psutil')},
                  model=dict(path=str(args.model), bytes=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns),
                  cache_regime='fresh worker, OS page cache uncontrolled, no purge',
                  physical_ssd_bandwidth_certified=False, before=system_snapshot(), turns=[])
    save(args.output, report)
    start = perf_counter()
    tokenizer_start = perf_counter()
    reader_kwargs = dict(memory_budget_bytes=256 << 20, max_metadata_bytes=192 << 20)
    if args.bulk_metadata:
        reader_kwargs['bulk_metadata'] = True
    if args.string_buffer_bytes:
        reader_kwargs['string_buffer_bytes'] = args.string_buffer_bytes
    with GGUFReader(args.model, **reader_kwargs) as checkpoint:
        tokenizer = GGUFTokenizer(checkpoint.metadata)
        report['tokenizer_io'] = dict(checkpoint.stats)
    report['tokenizer_seconds'] = perf_counter() - tokenizer_start
    options = NextConfig(cache_bytes=64 << 20, io_bytes=64 << 20, kv_bytes=256 << 20,
                         workspace_bytes=128 << 20, reserve_bytes=3 << 30, max_tensor_bytes=16 << 20,
                         row_chunk=args.rows, prefill_chunk=args.prefill_chunk, max_context=1280,
                         policy=args.cache_policy, device='gpu', dtype='float32', prefetch=False,
                         load_tokenizer=False, io_mode=args.io_mode, zero_copy_weights=args.zero_copy_weights)
    kwargs = dict(kernel='metal')
    kwargs.update(router_reference_options(getattr(args,'router_reference_chunk',0)))
    for name in ('expert_batch', 'concat_tiles', 'packed_copy_free', 'packed_direct_io', 'bulk_metadata', 'expert_pipeline'):
        if getattr(args, name):
            kwargs[name] = True
    if args.string_buffer_bytes:
        kwargs['string_buffer_bytes'] = args.string_buffer_bytes
    if args.hyper_prefill:
        kwargs.update(hyper_prefill=True, hyper_prefill_max_tokens=args.hyper_prefill_max_tokens)
    if args.metal_batch_backend != 'off':
        kwargs.update(metal_batch_backend=args.metal_batch_backend,metal_token_tile=args.metal_token_tile)
    if args.grouped_direct_io:
        kwargs['grouped_direct_io']=True
    report['options'] = asdict(options)
    report['backend_options'] = kwargs
    try:
        with NativeGGUFQwenModel(args.model, options, **kwargs) as model:
            report['initialization_seconds'] = perf_counter() - start
            report['model_initialization_seconds'] = report['initialization_seconds'] - report['tokenizer_seconds']
            report['model_config'] = model.config
            observer = None
            trace = None
            if args.numerical_evidence or args.request_trace:
                from hyperstream2_observer import RuntimeObserver
                if args.request_trace:
                    # Import diagnostics from the harness checkout even when the
                    # measured model is the untouched PR #10 checkout.
                    import importlib.util
                    spec = importlib.util.spec_from_file_location('hs2_trace', ROOT / 'air_llm/airllm/nextgen/request_trace.py')
                    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                    trace = module.RequestTrace(max_records=300000)
                observer = RuntimeObserver(model, trace=trace)
                observer.__enter__()
            original = model.forward
            checks = []

            def forward(*a, **kw):
                supplied=np.asarray(a[0] if a else kw['input_ids']).tolist()[0]
                cache=kw.get('past_key_values')
                start_position=0 if cache is None else cache.position
                if observer is not None:
                    observer.phase = 'prefill' if not checks else 'decode'
                    if args.speculative_depth:
                        observer.reset_routes()
                logical = model.checkpoint.stats['logical_bytes']
                disk = physical_read_bytes()
                begin = perf_counter()
                try:
                    result = original(*a, **kw)
                except BaseException:
                    budget = getattr(model.reader, 'packed_budget', None)
                    report['failed_forward_diagnostics'] = dict(
                        cache_bytes=model.reader.cache.bytes, cache_entries=len(model.reader.cache.data),
                        io_reserved_bytes=model.checkpoint._reserved,
                        packed_budget=None if budget is None else dict(live_bytes=budget.live_bytes,
                            capacity_bytes=budget.capacity_bytes, peak_bytes=budget.peak_bytes),
                        metrics=model.metrics())
                    save(args.output, report)
                    raise
                elapsed = perf_counter() - begin
                logits = np.asarray(result.logits)
                routes = {str(i): layer.routes for i, layer in enumerate(model.mlps)}
                after_disk = physical_read_bytes()
                checks.append(dict(logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                                   routes_sha256=hashlib.sha256(json.dumps(routes, sort_keys=True).encode()).hexdigest(),
                                   position=result.past_key_values.position, forward_seconds=elapsed,
                                   logical_bytes=model.checkpoint.stats['logical_bytes'] - logical,
                                   process_storage_read_bytes=None if disk is None or after_disk is None else after_disk - disk))
                checks[-1].update(input_ids=supplied,start_position=start_position,
                    finite_logits_checked=bool(np.isfinite(logits).all()))
                if not checks[-1]['finite_logits_checked']:
                    raise FloatingPointError('Non-finite logits in numerical evidence')
                if observer is not None:
                    checks[-1]['routing_by_layer_position']={layer:{position:ids for position,ids in bank.items()
                        if start_position<=int(position)<result.past_key_values.position}
                        for layer,bank in observer.routing.items()}
                if args.numerical_evidence:
                    vector = np.asarray(logits, dtype='<f4')
                    vector_path = args.output.with_name(args.output.stem + '.step-' + str(len(checks)-1).zfill(3) + '.f32')
                    # One task per numerical worker gives stable unique names.
                    with vector_path.open('xb') as f: f.write(vector.tobytes())
                    checks[-1]['logits_artifact'] = dict(path=vector_path.name,
                        sha256=hashlib.sha256(vector.tobytes()).hexdigest(), dtype='float32', shape=list(vector.shape))
                return result

            model.forward = forward
            for task in tasks:
                checks = []
                chat = GGUFChatSession(model, tokenizer, system_prompt='Tu es un assistant utile. Réponds en français.')
                item = dict(task, status='started', partial_ids=[])
                report['turns'].append(item)
                logical = model.checkpoint.stats['logical_bytes']
                read_calls = model.checkpoint.stats['read_calls']
                disk = physical_read_bytes()
                begin = perf_counter()
                turn_cpu=resource.getrusage(resource.RUSAGE_SELF)

                def emitted(event):
                    item['partial_ids'].append(event['token_id'])
                    item['partial_text'] = tokenizer.decode(item['partial_ids'], skip_special_tokens=True)
                    item['partial_elapsed_seconds'] = perf_counter() - begin
                    save(args.output, report)

                save(args.output, report)
                if args.speculative_depth:
                    messages=chat.messages+[{'role':'user','content':task['prompt']}]
                    prompt_text=tokenizer.apply_chat_template(messages,tokenize=False,
                        add_generation_prompt=True,enable_thinking=False,preserve_thinking=True)
                    full_ids=tokenizer.encode(prompt_text)
                    if len(full_ids)+task['max_new_tokens']>options.max_context:
                        raise MemoryError('Speculative prompt plus output exceeds context limit')
                    from airllm.nextgen.generation import eos_ids
                    stops=sorted(eos_ids(model.config.get('eos_token_id') or tokenizer.eos_token_id,
                        model.config['vocab_size']))
                    full_output=model.generate_speculative([full_ids],max_new_tokens=task['max_new_tokens'],
                        eos_token_id=stops,max_depth=args.speculative_depth,
                        candidate_depths=(args.speculative_depth,))
                    generated=full_output.tolist()[0][len(full_ids):]
                    raw=tokenizer.decode(generated,skip_special_tokens=False)
                    content=generated[:-1] if generated and generated[-1] in stops else generated
                    result=dict(full_input_ids=full_ids,appended_input_ids=full_ids,
                        generated_ids=generated,full_output_ids=full_output.tolist()[0],
                        prompt_text=prompt_text,text=tokenizer.decode(content,skip_special_tokens=True),raw_text=raw,
                        stop_reason='eos' if generated and generated[-1] in stops else 'max_new_tokens',
                        eos_token_ids=stops,token_seconds=None,
                        ttft_seconds=model.stats['ttft_seconds'],total_seconds=model.stats['generation_seconds'],
                        decode_tokens_per_second=model.stats['decode_tokens_per_second'],
                        native_cache_position=model.stats['speculative'].get('final_cache_position'),
                        finite_logits_checked=True,speculative_metrics=model.stats['speculative'],
                        partial_output_streamed=False,
                        generation_parameters=dict(max_new_tokens=task['max_new_tokens'],temperature=0.,
                            do_sample=False,enable_thinking=False,preserve_thinking=True,reuse_kv=False,
                            speculative_depth=args.speculative_depth))
                else:
                    result = chat.reply(task['prompt'], max_new_tokens=task['max_new_tokens'], on_token=emitted)
                item.update(result)
                end_cpu=resource.getrusage(resource.RUSAGE_SELF)
                item['cpu_seconds']=dict(user=end_cpu.ru_utime-turn_cpu.ru_utime,
                    system=end_cpu.ru_stime-turn_cpu.ru_stime,scope='worker process; includes instrumentation')
                item['pure_gpu_seconds']=None
                item.update(status='completed', wall_seconds=perf_counter() - begin, step_checks=checks,
                            logical_read_bytes=model.checkpoint.stats['logical_bytes'] - logical,
                            read_calls=model.checkpoint.stats['read_calls'] - read_calls, metrics=model.metrics())
                if observer is not None:
                    item['routing_sha256'] = observer.routing_sha256()
                    item['routing_by_layer_position'] = observer.routing
                    item['observer_timings'] = observer.summary()
                    if args.speculative_depth:
                        item['routing_scope']='Per-forward banks in step_checks; final bank is not the committed full path'
                if trace is not None:
                    trace_path = args.output.with_suffix('.trace.json')
                    save(trace_path, trace.export())
                    item['request_trace'] = dict(path=trace_path.name,
                        sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
                end_disk = physical_read_bytes()
                item['process_storage_read_bytes'] = None if disk is None or end_disk is None else end_disk - disk
                if getattr(model.reader, 'packed_budget', None) is not None:
                    budget = model.reader.packed_budget
                    item['packed_budget'] = dict(capacity_bytes=budget.capacity_bytes, live_bytes=budget.live_bytes, peak_bytes=budget.peak_bytes)
                print(json.dumps(dict(label=args.label, id=task['id'], text=item['text'],
                                      ttft=item['ttft_seconds'], decode_tok_s=item['decode_tokens_per_second'],
                                      stop_reason=item['stop_reason']), ensure_ascii=False), flush=True)
                save(args.output, report)
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        restoration_error = None
        if 'observer' in locals() and observer is not None:
            try:
                observer.__exit__(None, None, None)
            except BaseException as error:
                restoration_error = error
                report.update(status='failed', observer_restoration_error=f'{type(error).__name__}: {error}')
        report.update(after=system_snapshot(), worker_wall_seconds=perf_counter() - start,
                      source_unchanged=source_manifest(args.engine_root) == sources,
                      protocol_unchanged=protocol_unchanged(protocol),
                      model_stat_unchanged=(args.model.stat().st_size, args.model.stat().st_ino, args.model.stat().st_mtime_ns)
                      == (stat.st_size, stat.st_ino, stat.st_mtime_ns))
        report['git_head_after'] = engine_manifest(args.engine_root)['git_head']
        cpu_end=resource.getrusage(resource.RUSAGE_SELF)
        report['cpu_seconds']=dict(user=cpu_end.ru_utime-cpu_start.ru_utime,
            system=cpu_end.ru_stime-cpu_start.ru_stime,scope='worker process after checkpoint stat')
        report['engine_head_unchanged'] = report['git_head_after'] == report['git_head']
        save(args.output, report)
        if restoration_error is not None:
            raise RuntimeError('Observer restoration failed; evidence invalid') from restoration_error
        if report['status'] == 'completed' and not all(report[key] for key in (
                'source_unchanged', 'protocol_unchanged', 'model_stat_unchanged', 'engine_head_unchanged')):
            report['status'] = 'invalid'
            save(args.output, report)
            raise RuntimeError('Engine, protocol or model identity changed during measurement')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine-root', type=Path, default=ROOT)
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--rows', type=int, default=8192)
    parser.add_argument('--prefill-chunk', type=int, default=16)
    parser.add_argument('--hyper-prefill', action='store_true')
    parser.add_argument('--hyper-prefill-max-tokens', type=int, default=128)
    parser.add_argument('--router-reference-chunk',type=int,choices=[0,16],default=0)
    parser.add_argument('--numerical-evidence', action='store_true')
    parser.add_argument('--request-trace', action='store_true')
    parser.add_argument('--metal-batch-backend',choices=['off','reuse','tensorops'],default='off')
    parser.add_argument('--metal-token-tile',type=int,choices=[2,4,8,16],default=4)
    parser.add_argument('--grouped-direct-io',action='store_true')
    parser.add_argument('--speculative-depth',type=int,choices=[0,2,4,8,16],default=0)
    parser.add_argument('--cache-policy', choices=['adaptive', 'lru'], default='lru')
    parser.add_argument('--io-mode', choices=['pread', 'pread_nocache'], default='pread')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--string-buffer-bytes', type=string_buffer_size, default=0,
                        metavar='0..65536', help='Bounded GGUF string-array read buffer; zero retains legacy parsing')
    for flag in ('expert-batch', 'concat-tiles', 'packed-copy-free', 'packed-direct-io', 'bulk-metadata', 'zero-copy-weights', 'expert-pipeline'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--expected-protocol-json', help=argparse.SUPPRESS)
    parser.add_argument('--expected-engine-head', help=argparse.SUPPRESS)
    parser.add_argument('--expected-engine-sources-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ('model', 'tasks', 'output', 'engine_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    load_tasks(args.tasks)  # Refuse empty/malformed tasks before any model launch.
    if args.prefill_chunk < 1 or args.hyper_prefill_max_tokens < 1:
        parser.error('Prefill sizes must be positive')
    if args.grouped_direct_io and not (args.packed_direct_io and args.expert_batch):
        parser.error('Grouped direct I/O requires direct packed I/O and expert batching')
    if (args.numerical_evidence or args.request_trace) and len(load_tasks(args.tasks)) != 1:
        parser.error('Numerical artifacts and traces require exactly one task per worker')
    if args.worker:
        worker(args)
    else:
        sys.path.insert(0, str(ROOT / 'air_llm'))
        from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, validate_output_paths
        protected = [args.model, args.tasks, Path(__file__)]
        if args.expected_protocol_json:
            protected.extend(map(Path, json.loads(args.expected_protocol_json)))
        protected.extend(args.engine_root / name for name in source_manifest(args.engine_root))
        extras = [args.output.with_suffix('.trace.json')] if args.request_trace else []
        if args.numerical_evidence:
            extras.extend(args.output.with_name(args.output.stem + '.step-' + str(i).zfill(3) + '.f32')
                          for i in range(load_tasks(args.tasks)[0]['max_new_tokens']*4+1
                              if args.speculative_depth else load_tasks(args.tasks)[0]['max_new_tokens']))
        destinations = validate_output_paths(args.output, protected=protected, extra_outputs=extras)
        if any(p.exists() or p.is_symlink() for p in destinations):
            raise FileExistsError('Refusing to overwrite prior experiment evidence')
        args.output.parent.mkdir(exist_ok=True, parents=True)
        supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], args.output,
                  ExperimentLimits(timeout_seconds=args.timeout),
                  env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1'))


if __name__ == '__main__':
    main()
