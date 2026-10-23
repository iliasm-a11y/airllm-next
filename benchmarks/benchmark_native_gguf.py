#!/usr/bin/env python3
"""Supervised, progressive native GGUF experiments; no implicit model download."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + '\n')


def identity():
    files = list((ROOT / 'air_llm/airllm/nextgen').glob('*.py')) + [Path(__file__)]
    versions = {}
    for name in ('mlx', 'numpy', 'psutil', 'transformers', 'tokenizers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'platform': platform.platform(), 'python': sys.version, 'versions': versions,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}


def worker(args):
    from airllm.nextgen.experiment_guard import system_snapshot
    from airllm.nextgen.physical_io import physical_read_bytes, physical_read_counter_info
    from airllm.nextgen.gguf import GGUFReader
    begin = perf_counter()
    result = {'kind': 'native_airllm_gguf_research', 'phase': args.phase, 'status': 'started',
              'identity': identity(), 'before': system_snapshot(), 'arguments': vars(args) | {'output': str(args.output)},
              'os_cache': 'Uncontrolled; F_NOCACHE is an fd policy, never a cache purge or NAND measurement.',
              'gpu_command_seconds': None, 'energy_joules': None}
    result['arguments']['model'] = str(args.model)
    save(args.output, result)
    try:
        if args.phase in ('inspect', 'tokenizer', 'kernel'):
            reader = GGUFReader(args.model, memory_budget_bytes=256 << 20, max_metadata_bytes=192 << 20,
                                include_tokenizer_metadata=args.phase == 'tokenizer')
            with reader:
                result['file_bytes'] = reader.file_size
                result['tensor_count'] = len(reader.tensors)
                result['architecture'] = reader.metadata.get('general.architecture')
                if args.phase == 'inspect':
                    result['metadata'] = reader.metadata
                    result['tensors'] = {n: {'shape': d.shape, 'quant_type': int(d.quant_type), 'offset': d.offset,
                                             'nbytes': d.nbytes} for n, d in reader.tensors.items()}
                elif args.phase == 'tokenizer':
                    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
                    tokenizer = GGUFTokenizer(reader.metadata)
                    result['prompt_ids'] = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}])
                    result['prompt_text'] = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=False)
                    result['encoded_prompt'] = tokenizer.encode(args.prompt)
                    result['vocab_size'] = tokenizer.vocab_size
                    if args.reference_tokenizer:
                        from transformers import AutoTokenizer
                        reference = AutoTokenizer.from_pretrained(args.reference_tokenizer, local_files_only=True)
                        texts = [args.prompt, 'Été e\u0301lan 中文 مرحبا 123456 3.14159',
                                 '<think>abc</think><tool_call>{"a":1}</tool_call>', 'Hello\n\n  world!\r\n']
                        comparisons = []
                        for text in texts:
                            own, ref = tokenizer.encode(text), reference.encode(text, add_special_tokens=False)
                            comparisons.append({'text': text, 'ids': own, 'reference_ids': ref, 'equal': own == ref})
                        own = result['prompt_ids']
                        ref = reference.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=True,
                                                             add_generation_prompt=True, enable_thinking=False)
                        if hasattr(ref, 'get'):
                            ref = ref['input_ids']
                        comparisons.append({'kind': 'chat_template', 'ids': own, 'reference_ids': list(ref), 'equal': own == list(ref)})
                        result['reference_tokenizer'] = comparisons
                        if not all(c['equal'] for c in comparisons):
                            raise AssertionError('GGUF tokenizer does not match independent official tokenizer')
                else:
                    import mlx.core as mx
                    from airllm.nextgen.gguf_metal import packed_matmul
                    rng = np.random.default_rng(20261023)
                    comparisons = []
                    for name, expert in [('blk.0.ffn_gate_exps.weight', 7), ('blk.0.ffn_down_exps.weight', 7),
                                         ('output.weight', None), ('blk.3.attn_q.weight', None)]:
                        d = reader.tensors[name]
                        count = min(args.rows, d.shape[-2])
                        raw = reader.read_packed_rows(name, 0, count, expert=expert)
                        x = rng.normal(size=(1, d.width)).astype(np.float32)
                        host = reader.read_rows(name, 0, count, expert=expert)
                        expected = x @ host.T
                        inputs = mx.array(x)
                        actual = np.asarray(packed_matmul(raw, inputs, int(d.quant_type)))
                        delta = float(np.max(np.abs(actual - expected)))
                        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=5e-5)
                        timings = {'numpy_decode_matmul': [], 'metal_packed': []}
                        for _ in range(args.repeats):
                            t = perf_counter(); decoded = reader.read_rows(name, 0, count, expert=expert)
                            _ = x @ decoded.T
                            timings['numpy_decode_matmul'].append(perf_counter() - t)
                            del decoded
                            t = perf_counter(); packed_matmul(raw, inputs, int(d.quant_type))
                            timings['metal_packed'].append(perf_counter() - t)
                        comparisons.append({'tensor': name, 'expert': expert, 'rows': count,
                                             'maximum_absolute_error': delta, 'timings_seconds': timings,
                                             'scope': 'CPU includes read+decode+matmul; Metal reuses packed host tile, includes import+kernel'})
                        del raw, host, expected, actual
                    result['comparisons'] = comparisons
                result['io'] = dict(reader.stats)
        else:
            from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
            from airllm.nextgen.config import NextConfig
            options = NextConfig(cache_bytes=args.cache_mib << 20, io_bytes=64 << 20, kv_bytes=192 << 20,
                workspace_bytes=128 << 20, reserve_bytes=3 << 30, max_tensor_bytes=16 << 20,
                row_chunk=args.rows, prefill_chunk=args.prefill_chunk, max_context=args.max_context,
                policy=args.policy, device='gpu', dtype='float32', prefetch=False, load_tokenizer=False,
                io_mode=args.io_mode)
            t = perf_counter()
            with NativeGGUFQwenModel(args.model, options, kernel=args.kernel,cache_decay_interval=args.cache_decay) as model:
                result['initialization_seconds'] = perf_counter() - t
                result['config'] = model.config
                result['options'] = asdict(options)
                save(args.output, result)
                if args.phase == 'generate':
                    ids = [int(t) for t in args.token_ids.split(',')]
                    result['input_ids'] = ids
                    logical = model.checkpoint.stats['logical_bytes']
                    physical = physical_read_bytes(); cpu = process_time(); t = perf_counter()
                    cache = None; current = np.array([ids], np.int32); tokens = []; latencies = []
                    for step in range(args.tokens):
                        start = perf_counter()
                        out = model.forward(current, past_key_values=cache, logits_to_keep=1)
                        cache = out.past_key_values
                        logits = np.asarray(out.logits[0, -1])
                        if not np.all(np.isfinite(logits)):
                            raise AssertionError('Nonfinite logits')
                        token = int(np.argmax(logits)); tokens.append(token); latencies.append(perf_counter() - start)
                        result['generated_ids'] = tokens
                        result['token_seconds'] = latencies
                        result['logit_top10'] = [{'id': int(i), 'value': float(logits[i])} for i in np.argsort(-logits)[:10]]
                        result['last_logits_sha256'] = hashlib.sha256(logits.tobytes()).hexdigest()
                        result['metrics'] = model.metrics(); save(args.output, result)
                        print(json.dumps({'step': step + 1, 'token': token, 'seconds': latencies[-1],
                                          'rss_bytes': psutil.Process().memory_info().rss}), flush=True)
                        if token in (248046, 248044):
                            break
                        current = np.array([[token]], np.int32)
                    elapsed = perf_counter() - t
                    storage_end = physical_read_bytes()
                    result.update(generation_seconds=elapsed, generation_tokens_per_second=len(tokens) / elapsed,
                                  ttft_seconds=latencies[0] if latencies else None,
                                  decode_tokens_per_second=(len(tokens) - 1) / sum(latencies[1:]) if len(tokens) > 1 else None,
                                  cpu_seconds=process_time() - cpu,
                                  logical_bytes=model.checkpoint.stats['logical_bytes'] - logical,
                                  process_storage_read_bytes=storage_end - physical if physical is not None and storage_end is not None else None,
                                  storage_counter=physical_read_counter_info(), kv_peak_bytes=cache.peak_bytes if cache else 0)
                    if args.logits_output:
                        np.save(args.logits_output, logits)
                result['metrics'] = model.metrics()
        result['status'] = 'completed'
    except BaseException as error:
        result['status'] = 'failed'; result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['wall_seconds'] = perf_counter() - begin
        result['after'] = system_snapshot()
        result['rss_bytes_at_end'] = psutil.Process().memory_info().rss
        save(args.output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['inspect', 'tokenizer', 'kernel', 'init', 'generate'], default='inspect')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--tokens', type=int, default=1)
    p.add_argument('--token-ids', default='')
    p.add_argument('--prompt', default='Explique en deux phrases pourquoi le ciel est bleu.')
    p.add_argument('--reference-tokenizer')
    p.add_argument('--kernel', choices=['reference', 'metal'], default='reference')
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cache-mib', type=int, default=256)
    p.add_argument('--cache-decay', type=int)
    p.add_argument('--policy', choices=['adaptive', 'lru', 'slru', 'cost'], default='adaptive')
    p.add_argument('--prefill-chunk', type=int, default=16)
    p.add_argument('--max-context', type=int, default=256)
    p.add_argument('--io-mode', choices=['pread', 'pread_nocache'], default='pread')
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--logits-output')
    args = p.parse_args()
    from airllm.nextgen.experiment_guard import validate_output_paths
    extra = []
    if args.logits_output:
        extra = [args.logits_output if args.logits_output.endswith('.npy') else args.logits_output + '.npy']
    validate_output_paths(args.output, protected=[args.model], extra_outputs=extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.tokens < 1 or args.rows < 1 or args.repeats < 1 or args.cache_mib < 0 or (args.cache_decay is not None and args.cache_decay<1):
        p.error('Positive token/row/repetition counts and nonnegative cache are required')
    if args.worker:
        worker(args)
    else:
        from airllm.nextgen.experiment_guard import ExperimentLimits, supervise
        limits = ExperimentLimits(admitted_payload_bytes=max(768, args.cache_mib + 448) << 20,
                                  timeout_seconds=args.timeout)
        supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], args.output,
                  limits, env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))


if __name__ == '__main__':
    main()
