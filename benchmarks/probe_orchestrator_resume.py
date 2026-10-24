"""Real SmolLM2 replay after a complete supervised worker process exit."""
import argparse
import json
import os
from pathlib import Path
import sys


def worker(args):
    from airllm.nextgen import NextConfig, LocalOrchestrator, OrchestratorConfig, NativeModelBackend
    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
    from airllm.nextgen.experiment_guard import admission, ExperimentLimits
    admission(ExperimentLimits())
    state = Path(args.output).resolve()/'state'
    options = NextConfig(device='gpu', dtype='float32', cache_bytes=64<<20,
        io_bytes=64<<20, kv_bytes=256<<20, workspace_bytes=128<<20,
        reserve_bytes=3<<30, max_tensor_bytes=16<<20, row_chunk=2048,
        prefill_chunk=16, max_context=256, prefetch=False, load_tokenizer=True,
        use_saved_tuning=False, cache_dir=str(state.parent/'metadata'))
    cfg = OrchestratorConfig(kv_budget_bytes=options.kv_bytes, quantum_tokens=1)
    with AirLLMNextLlamaModel(args.model, next_config=options) as model:
        backend = NativeModelBackend(model)
        with LocalOrchestrator(backend, state, cfg) as engine:
            if args.worker_phase == 'suspend':
                prompts = [model.tokenizer.apply_chat_template(
                    [{'role':'user','content':p}], tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                    for p in ['Answer 2 + 2 briefly.', 'Name one French city.']]
                expected = [model.generate([p], max_new_tokens=4, eos_token_id=[]).tolist()[0][len(p):]
                            for p in prompts]
                job_ids = []
                for i, prompt in enumerate(prompts):
                    agent = f'agent-{i}'
                    engine.register(agent)
                    job_ids.append(engine.submit(agent, prompt, max_tokens=4))
                engine.step()
                engine.step()
                engine.suspend_all()
                payload = dict(expected=expected, job_ids=job_ids,
                    jobs=[engine.snapshot(j) for j in job_ids],
                    status=engine.status(), model_identity=backend.identity,
                    worker_pid=os.getpid())
            else:
                previous = json.loads((state.parent/'suspend.json').read_text())
                assert all(engine.snapshot(j)['state'] == 'paused' for j in previous['job_ids'])
                assert backend.identity == previous['model_identity']
                engine.clear_pressure()
                engine.resume_all()
                while engine.status()['jobs_queued']:
                    engine.step()
                jobs = [engine.snapshot(j) for j in previous['job_ids']]
                assert [j['output_ids'] for j in jobs] == previous['expected']
                assert all(j['state'] == 'completed' for j in jobs)
                payload = dict(jobs=jobs, ordinary_serial_tokens_equal=True,
                    status=engine.status(), model_identity=backend.identity,
                    output_text=[model.tokenizer.decode(j['output_ids']) for j in jobs],
                    worker_pid=os.getpid(), previous_worker_pid=previous['worker_pid'])
    (state.parent/f'{args.worker_phase}.json').write_text(json.dumps(payload, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, help='Fresh evidence directory')
    parser.add_argument('--worker-phase', choices=['suspend', 'resume'])
    args = parser.parse_args()
    if args.worker_phase:
        return worker(args)
    from airllm.nextgen.experiment_guard import supervise, ExperimentLimits
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MLX_ENABLE_TF32='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for phase in ('suspend', 'resume'):
        supervise([sys.executable, str(Path(__file__).resolve()), '--model', args.model,
                   '--output', str(folder), '--worker-phase', phase],
                  folder/f'{phase}.json', ExperimentLimits(), env=env)
    report = json.loads((folder/'resume.json').read_text())
    assert report['worker_pid'] != report['previous_worker_pid']
    print(json.dumps({'fresh_process_resume':True, 'ordinary_serial_tokens_equal':True}))


if __name__ == '__main__':
    main()
