#!/usr/bin/env python3
"""Supervised replay that captures the original router's failing local arrays.

Instrumentation is confined to the child: a wrapper calls the unchanged router
and reads its traceback locals only after its original ValueError is raised.
It never substitutes routing decisions or resumes after the captured failure.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def worker(args):
    import numpy as np
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.experiment_guard import system_snapshot
    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
    from airllm.nextgen.gguf_chat import GGUFChatSession
    from airllm.nextgen import gguf_qwen, qwen_moe

    reference = json.loads(args.reference.read_text())
    reference_turn = next(turn for turn in reference['turns'] if turn['id'] == args.turn)
    tasks = json.loads(args.tasks.read_text())
    session_spec = next(session for session in tasks
                        if any(turn['id'] == args.turn for turn in session['turns']))
    spec = next(turn for turn in session_spec['turns'] if turn['id'] == args.turn)
    if spec['prompt'] != reference_turn['prompt'] or spec['max_new_tokens'] != reference_turn['max_new_tokens']:
        raise ValueError('Replay prompt or output cap differs from the failed reference turn')
    if len(session_spec['turns']) != 1:
        raise ValueError('This focused replay only supports a standalone conversation')
    sources = reference['identity']['sources']
    current = {name: digest(ROOT / name) for name in sources}
    changed = {name: {'recorded': expected, 'current': current[name]}
               for name, expected in sources.items()
               if current[name] != expected and name.startswith('air_llm/')}
    if changed:
        raise ValueError('Source files differ from failed execution: ' + json.dumps(changed))
    original = gguf_qwen.qwen_route
    if original is not qwen_moe.qwen_route:
        raise ValueError('Unexpected pre-existing router instrumentation')
    report = {
        'kind': 'native_router_failure_traceback_capture', 'status': 'started',
        'reference_file': str(args.reference), 'reference_sha256': digest(args.reference),
        'turn_id': args.turn, 'reference_error': reference_turn.get('error'),
        'reference_generated_ids': reference_turn.get('partial_generated_ids', []),
        'source_hashes': current, 'sources_match_reference': True,
        'harness_source_changes': {name: {'recorded': expected, 'current': current[name]}
                                  for name, expected in sources.items()
                                  if current[name] != expected and not name.startswith('air_llm/')},
        'instrumentation_file': str(Path(__file__).resolve()),
        'instrumentation_sha256': digest(Path(__file__).resolve()),
        'method': 'Call unchanged original qwen_route; on its ValueError capture original traceback frame locals; rethrow without continuing inference.',
        'prompt': spec['prompt'], 'system': session_spec.get('system'),
        'max_new_tokens': spec['max_new_tokens'], 'options': reference['options'],
        'before': system_snapshot(), 'partial_generated_ids': [], 'partial_text': '',
        'router_calls': 0, 'capture': None,
    }
    save(args.output, report)
    begin = perf_counter()

    def traced_router(mx, x, weight, config):
        report['router_calls'] += 1
        try:
            return original(mx, x, weight, config)
        except ValueError as error:
            if 'boundary tie' not in str(error):
                raise
            tb = error.__traceback__
            original_locals = None
            frame_details = []
            while tb is not None:
                frame = tb.tb_frame
                if frame.f_code is original.__code__:
                    original_locals = frame.f_locals
                frame_details.append({'function': frame.f_code.co_name, 'line': tb.tb_lineno,
                                      'file': frame.f_code.co_filename})
                tb = tb.tb_next
            if original_locals is None:
                raise RuntimeError('Could not find original router traceback frame') from error
            # Capture actual lazy arrays retained by the failing original call,
            # not a recomputation under a possibly different matmul schedule.
            arrays = {'x': x, 'weight': weight}
            for name in ('logits', 'probabilities', 'order', 'selected', 'scores', 'next_score'):
                if name in original_locals:
                    arrays[name] = original_locals[name]
            conservative_bytes = sum(math.prod(value.shape) * 8 for value in arrays.values())
            if conservative_bytes > 16 << 20:
                raise MemoryError('Router capture exceeds the fixed 16 MiB export allowance') from error
            host = {name: np.asarray(value).copy() for name, value in arrays.items()}
            array_bytes = sum(value.nbytes for value in host.values())
            if array_bytes > 16 << 20:
                raise MemoryError('Router arrays exceed the fixed 16 MiB export allowance') from error
            # Find native mixer/forward context in the live caller stack.
            caller = sys._getframe(1)
            context = {}
            try:
                while caller is not None:
                    owner = caller.f_locals.get('self')
                    if isinstance(owner, gguf_qwen.GGUFQwenSparseMLP):
                        context['layer'] = owner.layer
                        context['weight_prefix'] = owner.p
                    if caller.f_code.co_name == 'forward' and 'cache' in caller.f_locals:
                        cache = caller.f_locals['cache']
                        context['cache_position_before_current_chunk'] = cache.position
                        if 'ids' in caller.f_locals:
                            context['forward_input_ids'] = np.asarray(caller.f_locals['ids']).tolist()
                        context['prefill_chunk_start'] = caller.f_locals.get('start')
                    caller = caller.f_back
            finally:
                del caller
            top_k = config['num_experts_per_tok']
            order = host['order']
            rows = np.nonzero(host['scores'][:, -1] == host['next_score'][:, 0])[0]
            boundaries = []
            for row in rows.tolist():
                selected_id = int(order[row, top_k - 1])
                excluded_id = int(order[row, top_k])
                boundaries.append({
                    'flat_row': row, 'selected_boundary_expert': selected_id,
                    'excluded_boundary_expert': excluded_id,
                    'selected_logit': float(host['logits'][row, selected_id]),
                    'excluded_logit': float(host['logits'][row, excluded_id]),
                    'logits_equal': bool(host['logits'][row, selected_id] == host['logits'][row, excluded_id]),
                    'selected_probability': float(host['probabilities'][row, selected_id]),
                    'excluded_probability': float(host['probabilities'][row, excluded_id]),
                    'probabilities_equal': bool(host['probabilities'][row, selected_id] == host['probabilities'][row, excluded_id]),
                })
            np.savez_compressed(args.output.with_suffix('.npz'), **host)
            report['capture'] = {
                'error': str(error), **context, 'generated_token_count_before_failure': len(report['partial_generated_ids']),
                'num_experts': config['num_experts'], 'top_k': top_k, 'tie_boundaries': boundaries,
                'arrays': {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                  'nbytes': value.nbytes, 'all_finite': bool(np.isfinite(value).all())}
                           for name, value in host.items()},
                'total_uncompressed_array_bytes': array_bytes,
                'npz_file': str(args.output.with_suffix('.npz')),
                'npz_sha256': digest(args.output.with_suffix('.npz')),
                'original_traceback': frame_details,
            }
            save(args.output, report)
            raise

    gguf_qwen.qwen_route = traced_router
    qwen_moe.qwen_route = traced_router
    try:
        with GGUFReader(args.model, memory_budget_bytes=256 << 20,
                        max_metadata_bytes=192 << 20) as cp:
            tokenizer = GGUFTokenizer(cp.metadata)
        del cp
        with gguf_qwen.NativeGGUFQwenModel(args.model, NextConfig(**reference['options']), kernel='metal') as model:
            report['config'] = model.config
            chat = GGUFChatSession(model, tokenizer, system_prompt=session_spec.get('system'), reuse_kv=True)
            def emitted(event):
                report['partial_generated_ids'].append(event['token_id'])
                report['partial_text'] = tokenizer.decode(report['partial_generated_ids'], skip_special_tokens=True)
                report['partial_elapsed_seconds'] = perf_counter() - begin
                save(args.output, report)
            try:
                result = chat.reply(spec['prompt'], max_new_tokens=spec['max_new_tokens'], on_token=emitted)
                report['result'] = result
                report['status'] = 'completed_without_reference_failure'
            finally:
                report['metrics'] = model.metrics()
    except BaseException as error:
        report['status'] = 'captured_reference_failure' if report['capture'] else 'failed_without_capture'
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        # An observed router failure is the requested diagnostic outcome.
        if report['capture'] is None:
            raise
    finally:
        gguf_qwen.qwen_route = original
        qwen_moe.qwen_route = original
        actual = report['partial_generated_ids']
        expected = report['reference_generated_ids']
        common = min(len(actual), len(expected))
        report['comparison'] = {'generated_ids_exact_equal': actual == expected,
                                'common_prefix_exact_equal': actual[:common] == expected[:common],
                                'common_prefix_length': common, 'actual_count': len(actual),
                                'reference_count': len(expected)}
        report['after'] = system_snapshot()
        report['elapsed_seconds'] = perf_counter() - begin
        report['source_hashes_after'] = {name: digest(ROOT / name) for name in current}
        report['source_files_unchanged_during_replay'] = report['source_hashes_after'] == current
        save(args.output, report)
    print(json.dumps({'status': report['status'], 'comparison': report['comparison'],
                      'capture': report['capture']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=ROOT / 'research/real_world/tasks.json')
    parser.add_argument('--turn', default='D2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, validate_output_paths
    validate_output_paths(args.output, protected=[args.model, args.reference, args.tasks],
                          extra_outputs=[args.output.with_suffix('.npz')])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
    else:
        limits = ExperimentLimits(admitted_payload_bytes=784 << 20, timeout_seconds=args.timeout)
        supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                  args.output, limits,
                  env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))


if __name__ == '__main__':
    main()
