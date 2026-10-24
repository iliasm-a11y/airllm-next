"""Reproducible logical-agent load; synthetic and trained runs stay separate.

Launch under experiment_guard.supervise; this script never downloads weights.
Each real run uses ONE model object for every logical agent and every scale.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter, sleep

from airllm.nextgen.orchestrator import (
    InferenceResult, LocalOrchestrator, OrchestratorConfig)


class SyntheticBackend:
    identity = 'synthetic-orchestrator-load-v1-NOT-a-trained-model'
    max_concurrency = 4
    concurrent_safe = True
    kv_bytes_per_session = 4096

    def advance(self, request, max_tokens):
        # Deliberate bounded service delay enables observable overlap without
        # CPU/GPU allocations. The resulting throughput is scheduler-only.
        sleep(.001)
        known = list(request.output_ids)
        result = []
        for _ in range(min(max_tokens, request.max_tokens-len(known))):
            if request.stop_event.is_set():
                break
            token = (sum(request.input_ids)+sum(known)+len(known)+1) % 64
            known.append(token)
            result.append(token)
        return InferenceResult(result, len(known) == request.max_tokens)

    def release(self, job_id):
        pass


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ['air_llm/airllm/nextgen/orchestrator.py',
             'air_llm/airllm/nextgen/orchestrator_native.py',
             'air_llm/airllm/nextgen/agent_tools.py',
             'benchmarks/benchmark_orchestrator.py']
    return {path: sha256((root/path).read_bytes()).hexdigest() for path in paths}


def run_scale(backend, count, prompts, tokens, parent, expected=None):
    cfg = OrchestratorConfig(max_agents=max(1024, count), max_jobs=max(1024, count),
        max_concurrency=4, quantum_tokens=min(tokens, 8),
        kv_budget_bytes=backend.kv_bytes_per_session*min(4, backend.max_concurrency))
    begin = perf_counter()
    with LocalOrchestrator(backend, state_dir=parent/f'agents-{count}', config=cfg) as engine:
        job_ids = []
        for index in range(count):
            agent = f'agent-{index:06d}'
            engine.register(agent, token_quota=1024, job_quota=4)
            job_ids.append(engine.submit(agent, prompts[index % len(prompts)],
                max_tokens=tokens, priority=index % 3, eos_token_ids=()))
        admitted = perf_counter()
        admission_status = engine.status()
        while engine.status()['jobs_queued'] and not engine.status()['stopped_reason']:
            engine.step()
        end = perf_counter()
        jobs = [engine.snapshot(job) for job in job_ids]
        status = engine.status()
        successful = all(job['state'] == 'completed' for job in jobs)
        parity = (None if expected is None else all(job['output_ids'] ==
            expected[index % len(expected)] for index, job in enumerate(jobs)))
        if not successful or parity is False:
            raise RuntimeError('Load run did not complete with reference tokens')
        return dict(agents=count, admission_seconds=admitted-begin,
            drain_seconds=end-admitted, total_seconds=end-begin,
            completed_jobs=len(jobs), generated_tokens=sum(len(j['output_ids']) for j in jobs),
            jobs_per_second=len(jobs)/(end-begin),
            tokens_per_second=sum(len(j['output_ids']) for j in jobs)/(end-begin),
            ordinary_serial_tokens_equal=parity, admission_status=admission_status,
            final_status=status, jobs=jobs, config=asdict(cfg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'smollm', 'qwen-gguf'], default='synthetic')
    parser.add_argument('--model', help='Existing local checkpoint; no download')
    parser.add_argument('--counts', default='1,10,100,500,600')
    parser.add_argument('--tokens', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    counts = [int(n) for n in args.counts.split(',')]
    if any(n < 1 or n > 10000 for n in counts) or not 1 <= args.tokens <= 128:
        parser.error('Counts must be 1..10000; tokens must be 1..128')
    out = Path(args.output).resolve()
    if out.exists():
        parser.error('Output already exists; choose a fresh evidence path')
    from airllm.nextgen.experiment_guard import admission, ExperimentLimits, validate_output_paths
    validate_output_paths(out, protected=[__file__]+([args.model] if args.model else []))
    # Run admission before any trained model construction; the independent
    # parent supervisor also monitors resources during each in-flight forward.
    admission(ExperimentLimits())
    initial_sources = source_hashes()
    report = dict(schema_version=1, status='running',
        started_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
        scope=('scheduler simulation; no model inference' if
        args.mode == 'synthetic' else 'trained local text model; short bounded generations'),
        machine=platform.machine(), platform=platform.platform(), python=sys.version,
        source_hashes=initial_sources, requested_tokens=args.tokens,
        os_page_cache_purged=False, physical_ssd_bandwidth_certified=False,
        model_instances=0 if args.mode == 'synthetic' else 1, runs=[])
    model = None
    try:
        with tempfile.TemporaryDirectory(prefix='airllm-orchestration-') as tmp:
            parent = Path(tmp)
            expected = None
            if args.mode == 'synthetic':
                backend = SyntheticBackend()
                prompts = [[1, 3], [5, 2], [9, 4]]
            else:
                if not args.model or not Path(args.model).exists():
                    parser.error('A local model is required')
                from airllm.nextgen import NextConfig
                from airllm.nextgen.orchestrator_native import NativeModelBackend
                options = NextConfig(device='gpu', dtype='float32', cache_bytes=64<<20,
                    io_bytes=64<<20, kv_bytes=256<<20, workspace_bytes=128<<20,
                    reserve_bytes=3<<30, max_tensor_bytes=16<<20, row_chunk=2048,
                    prefill_chunk=16, max_context=256, prefetch=False,
                    load_tokenizer=True, use_saved_tuning=False,
                    cache_dir=str(parent/'metadata'))
                if args.mode == 'smollm':
                    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
                    model = AirLLMNextLlamaModel(args.model, next_config=options)
                    prompts = [model.tokenizer.apply_chat_template(
                        [{'role':'user', 'content':prompt}], tokenize=True,
                        add_generation_prompt=True, return_dict=False)
                        for prompt in ['Answer 2 + 2 briefly.', 'Name one French city.', 'Say hello.']]
                else:
                    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
                    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
                    from airllm.nextgen.gguf import GGUFReader
                    # Same tokenizer allocation contract as throughput_gguf:
                    # the embedded vocabulary/merges exceed metadata-only32MiB.
                    # Parse before constructing the model to avoid overlapping
                    # tokenizer scratch with native inference allocations.
                    with GGUFReader(args.model, memory_budget_bytes=256<<20,
                                    max_metadata_bytes=192<<20,
                                    include_tokenizer_metadata=True,
                                    bulk_metadata=True) as tokenizer_reader:
                        tokenizer = GGUFTokenizer(tokenizer_reader.metadata)
                    del tokenizer_reader
                    model = NativeGGUFQwenModel(args.model, next_config=options, kernel='metal',
                        expert_batch=True, concat_tiles=True, packed_direct_io=True, bulk_metadata=True)
                    prompts = [tokenizer.encode(tokenizer.apply_chat_template(
                        [{'role':'user','content':'Answer 2 + 2 briefly.'}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False, preserve_thinking=True))]
                    model.tokenizer = tokenizer
                backend = NativeModelBackend(model)
                report['native_options'] = options.as_dict()
                report['checkpoint_fingerprint'] = backend.checkpoint_fingerprint
                report['model_name'] = 'SmolLM2-135M-Instruct' if args.mode == 'smollm' else 'Qwen3.6-35B-A3B-GGUF'
                expected = [model.generate([prompt], max_new_tokens=args.tokens,
                    eos_token_id=[]).tolist()[0][len(prompt):] for prompt in prompts]
                report['reference_output_ids'] = expected
                report['reference_output_text'] = [model.tokenizer.decode(t) for t in expected]
            report['backend_identity'] = backend.identity
            report['prompt_ids'] = prompts
            for count in counts:
                report['runs'].append(run_scale(backend, count, prompts, args.tokens, parent, expected))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in report['runs'][-1].items()
                    if k in ('agents','total_seconds','tokens_per_second','ordinary_serial_tokens_equal')}), flush=True)
            if initial_sources != source_hashes():
                raise RuntimeError('Source changed during campaign; evidence is interrupted')
            report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'interrupted'
        report['error'] = type(error).__name__
        report['final_source_hashes'] = source_hashes()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        raise
    finally:
        if model is not None:
            model.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output':str(out), 'runs':len(report['runs'])}))


if __name__ == '__main__':
    main()
