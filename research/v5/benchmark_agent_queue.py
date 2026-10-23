"""Trained-checkpoint job throughput/isolation/restart, not RONIN mission quality.

Run after other timing jobs finish. The checkpoint must already be local. No
network access, OS cache purge, model payload mutation or giant download occurs.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter, time

os.environ.setdefault('MLX_ENABLE_TF32', '0')
from airllm.nextgen.agent_engine import AgentEngine
from airllm.nextgen.config import NextConfig
from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint = Path(args.model).resolve()
    config = NextConfig(device='cpu', dtype='float32', cache_bytes=16*1024**2,
        io_bytes=8*1024**2, kv_bytes=8*1024**2, workspace_bytes=16*1024**2,
        reserve_bytes=0, max_tensor_bytes=8*1024**2, row_chunk=2048,
        prefill_chunk=16, max_context=256, prefetch=False, load_tokenizer=True,
        use_saved_tuning=False, cache_dir=str(Path(args.output).resolve().parent/'agent-benchmark-metadata'))
    prompts = [
        'What is the capital of France? Answer briefly.',
        'Calculate 2 + 2. Answer briefly.',
    ]
    report = {'scope': 'two bounded trained text inference jobs; not autonomous RONIN missions',
              'time_utc': time(), 'platform': platform.platform(), 'python': sys.version,
              'model_path': str(checkpoint), 'compute': 'MLX CPU',
              'os_page_cache_purged': False, 'physical_external_ssd_measured': False,
              'max_new_tokens': 8, 'prompt_count': len(prompts), 'runs': []}
    with tempfile.TemporaryDirectory(prefix='airllm-agent-benchmark-') as temp, \
         AirLLMNextLlamaModel(checkpoint, next_config=config) as model:
        ids = [model.tokenizer.apply_chat_template([{'role':'user', 'content':prompt}],
               tokenize=True, add_generation_prompt=True, return_dict=False) for prompt in prompts]
        report['model_fingerprint'] = model.checkpoint.fingerprint
        report['prompt_ids'] = ids
        report['config'] = config.as_dict()
        model.generate([ids[0]], max_new_tokens=1, eos_token_id=[])
        expected = []
        begin = perf_counter()
        for prompt in ids:
            expected.append(model.generate([prompt], max_new_tokens=8, eos_token_id=[]).tolist()[0][len(prompt):])
        serial_seconds = perf_counter()-begin
        report['runs'].append({'mode':'ordinary_serial', 'seconds':serial_seconds,
            'completed_jobs':len(ids), 'inference_jobs_per_hour_short_run':len(ids)*3600/serial_seconds,
            'output_ids':expected})
        for quantum in (8, 1):
            state = Path(temp)/f'quantum-{quantum}'
            begin = perf_counter()
            with AgentEngine(model, state_dir=state, quantum_tokens=quantum) as engine:
                job_ids = [engine.submit(prompt, max_tokens=8, eos_token_id=[], agent_id=f'RONIN-{i}')
                           for i, prompt in enumerate(ids)]
                while engine.status()['queued']:
                    engine.step()
                jobs = [engine.snapshot(job_id) for job_id in job_ids]
                status = engine.status()
            elapsed = perf_counter()-begin
            actual = [job['output_ids'] for job in jobs]
            if actual != expected or any(job['state'] != 'completed' for job in jobs):
                raise AssertionError('Agent queue changed ordinary greedy tokens')
            report['runs'].append({'mode':f'priority_round_robin_quantum_{quantum}',
                'seconds':elapsed, 'completed_jobs':len(ids),
                'inference_jobs_per_hour_short_run':len(ids)*3600/elapsed,
                'ordinary_tokens_equal':True, 'jobs':jobs, 'engine':status,
                'output_text':[model.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in actual]})
        state = Path(temp)/'restart'
        begin = perf_counter()
        with AgentEngine(model, state_dir=state, quantum_tokens=2) as engine:
            job_id = engine.submit(ids[0], max_tokens=8, eos_token_id=[], agent_id='RONIN-restart')
            first = engine.step()
            assert len(first['output_ids']) == 2
            first_engine_status = engine.status()
        with AgentEngine(model, state_dir=state, quantum_tokens=2) as engine:
            assert engine.snapshot(job_id)['state'] == 'paused'
            engine.resume(job_id)
            while engine.status()['queued']:
                engine.step()
            job = engine.snapshot(job_id)
            if job['state'] != 'completed' or job['output_ids'] != expected[0]:
                raise AssertionError('Durable resume changed ordinary greedy tokens')
            report['restart'] = {'seconds':perf_counter()-begin, 'ordinary_tokens_equal':True,
                                 'job':job, 'engine':engine.status(), 'pre_restart_engine':first_engine_status}
        report['model_metrics'] = model.metrics()
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['process_peak_rss_bytes'] = raw if sys.platform == 'darwin' else raw*1024
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output':str(output), 'queue_tokens_equal':True,
                      'restart_tokens_equal':True,
                      'runs':[{'mode':r['mode'], 'seconds':r['seconds']} for r in report['runs']]}))


if __name__ == '__main__':
    main()
