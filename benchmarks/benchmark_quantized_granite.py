#!/usr/bin/env python3
"""Explicit lossy full-Granite ablation; never modifies the original checkpoint.

Expert Q4 and mixed expert Q4 / eligible linear Q8 use the existing resumable
sidecar converter and NativeWeights QMM. Generation timings are uninstrumented.
The optional extra linear profile synchronizes every floating and packed linear
operation; its dependency-inclusive wall times are not GPU kernel timings.
Teacher forcing always follows the original model's generated continuation.
These few prompts do not validate general corpus quality or giant models.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
MiB = 1024**2
PROMPTS = [
    'What is the capital of France? Answer in one sentence.',
    'Complete the sequence: one, two, three, four, five,',
    'Return only Python code for a function add(a, b) that returns their sum.',
    'Explique en deux phrases pourquoi le ciel est bleu.',
]
VARIANTS = ('original', 'expert4', 'mixed')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    paths = sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py')) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def numeric_delta(after, before):
    return {key: value-before.get(key, 0) for key, value in after.items()
            if type(value) in (int, float)}


def compare_logits(reference, actual, labels=None):
    """Stable float64 KL(reference || variant), without an exactness threshold."""
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or reference.ndim < 2 or not reference.size:
        raise ValueError('Logits must have equal nonempty shapes with a vocabulary axis')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite model logits')
    def log_softmax(values):
        centered = values-values.max(axis=-1, keepdims=True)
        return centered-np.log(np.exp(centered).sum(axis=-1, keepdims=True))
    ref_log, actual_log = log_softmax(reference), log_softmax(actual)
    kl = np.sum(np.exp(ref_log)*(ref_log-actual_log), axis=-1)
    error = actual-reference
    result = dict(positions=int(np.prod(reference.shape[:-1])),
        max_abs_logit_error=float(np.abs(error).max()), mean_squared_logit_error=float(np.mean(error**2)),
        mean_kl_reference_to_variant=float(np.mean(kl)), max_kl_reference_to_variant=float(np.max(kl)),
        top1_agreement=float(np.mean(reference.argmax(axis=-1) == actual.argmax(axis=-1))))
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != reference.shape[:-1] or np.any(labels < 0) or np.any(labels >= reference.shape[-1]):
            raise ValueError('Teacher-forced labels do not match the prediction positions')
        ref_nll = -np.take_along_axis(ref_log, labels[..., None], axis=-1)
        actual_nll = -np.take_along_axis(actual_log, labels[..., None], axis=-1)
        result.update(reference_mean_continuation_nll=float(np.mean(ref_nll)),
            variant_mean_continuation_nll=float(np.mean(actual_nll)),
            mean_continuation_nll_delta=float(np.mean(actual_nll-ref_nll)))
    return result


def generation_summary(tokens, prompt_length, reference_tokens, seconds, eos_ids):
    """Count actual work, including a terminal EOS, and reject post-EOS work."""
    continuation = tokens[prompt_length:]
    reference = reference_tokens[prompt_length:]
    eos_positions = [index for index, token in enumerate(continuation) if token in eos_ids]
    if not continuation or seconds <= 0:
        raise ValueError('Generation must produce tokens in a positive measured interval')
    if eos_positions and eos_positions != [len(continuation)-1]:
        raise ValueError('Generation continued beyond the first EOS')
    differences = [index for index, (left, right) in enumerate(zip(continuation, reference)) if left != right]
    first = differences[0] if differences else min(len(continuation), len(reference)) if len(continuation) != len(reference) else None
    return dict(new_tokens=len(continuation), tokens_per_second=len(continuation)/seconds,
        tokens=tokens, matches_original_tokens=tokens == reference_tokens,
        first_continuation_divergence=first, original_reference_new_tokens=len(reference),
        continuation_length_difference=len(continuation)-len(reference),
        eos_count=len(eos_positions), tokens_after_first_eos=0,
        stop_reason='eos' if eos_positions else 'max_new_tokens')


def expected_entries(cp, scope):
    from airllm.nextgen.sidecar import expert_matrices, is_expert_tensor
    matrices = list(expert_matrices(cp))
    if scope == 'mixed':
        for key, meta in cp.tensors.items():
            if len(meta['shape']) != 2 or not key.endswith('.weight'):
                continue
            if is_expert_tensor(key) or any(word in key for word in
                    ('embed_tokens', 'indexer.', 'indexers_proj', 'mlp.gate.',
                     'block_sparse_moe.router.', 'kv_b_proj')):
                continue
            matrices.append((key, None, tuple(meta['shape'])))
    return {key if expert is None else f'{key}#{expert}':
            (key, expert, shape, 4 if is_expert_tensor(key) else 8)
            for key, expert, shape in matrices}


def audit_sidecar(cp, sidecar, scope):
    expected = expected_entries(cp, scope)
    if set(sidecar.entries) != set(expected):
        raise ValueError(f'Incomplete/wrong-scope {scope} sidecar: expected {len(expected)}, got {len(sidecar.entries)} matrices')
    sidecar.verify(require_digest=True)
    replaced, histogram = 0, {}
    for ident, (key, expert, shape, bits) in expected.items():
        entry = sidecar.entries[ident]
        if tuple(entry['shape']) != shape or entry.get('bits', sidecar.bits) != bits:
            raise ValueError(f'Wrong packed matrix format: {ident}')
        histogram[str(bits)] = histogram.get(str(bits), 0)+1
        meta = cp.tensors[key]
        replaced += meta['bytes'] if expert is None else meta['bytes']//meta['shape'][0]
    payload = sum(entry['nbytes'] for entry in sidecar.entries.values())
    original_payload = sum(meta['bytes'] for meta in cp.tensors.values())
    return dict(scope=scope, complete=True, matrices=len(expected), matrix_bits=histogram,
        affine_group_size=sidecar.group_size,
        original_model_payload_bytes=original_payload, replacement_source_payload_bytes=replaced,
        packed_payload_bytes=payload, effective_model_payload_bytes=original_payload-replaced+payload,
        sidecar_file_bytes=sum((sidecar.root/entry['file']).stat().st_size for entry in sidecar.entries.values()),
        manifest_sha256=digest(sidecar.path), all_entry_sha256_verified=True,
        max_matrix_relative_l2_error=max((entry['relative_l2_error'] for entry in sidecar.entries.values()), default=0),
        max_matrix_abs_weight_error=max((entry['max_abs_error'] for entry in sidecar.entries.values()), default=0),
        max_conversion_tile_bytes=max((entry['peak_conversion_tile_bytes'] for entry in sidecar.entries.values()), default=0))


def snapshot(model):
    return dict(io=dict(model.checkpoint.stats), reader=dict(model.reader.stats),
                sidecar=dict(model.reader.sidecar.stats) if model.reader.sidecar else {})


def clear_weights(model):
    model.reader.drain()
    model.reader.cache.clear()
    model.reader.compact_cache.clear()


def measure(model, operation):
    from airllm.nextgen.physical_io import physical_read_bytes
    clear_weights(model)
    before, physical_before = snapshot(model), physical_read_bytes()
    started = time.perf_counter()
    output = operation()
    model.mx.eval(output)
    elapsed = time.perf_counter()-started
    after, physical_after = snapshot(model), physical_read_bytes()
    io = numeric_delta(after['io'], before['io'])
    row = dict(seconds=elapsed, io_delta=io,
        reader_delta=numeric_delta(after['reader'], before['reader']),
        sidecar_delta=numeric_delta(after['sidecar'], before['sidecar']),
        source_file_payload_bytes=io.get('logical_bytes', 0)-io.get('quantized_logical_bytes', 0),
        quantized_payload_bytes=io.get('quantized_logical_bytes', 0),
        process_physical_read_bytes=physical_after-physical_before
            if physical_before is not None and physical_after is not None else None,
        cache=model.metrics()['cache'], io_scheduler=model.metrics()['prefetch'])
    return output, row


@contextmanager
def linear_profile(model):
    """Extra pass only: each wall interval includes input graph work and eval."""
    from airllm.nextgen.quantization import PackedMatrix
    native = model.reader.reader
    original = native.linear_weight
    counters = dict(packed_calls=0, floating_calls=0, packed_wall_seconds=0., floating_wall_seconds=0.)
    def measured(x, weight):
        kind = 'packed' if isinstance(weight, PackedMatrix) else 'floating'
        started = time.perf_counter()
        value = original(x, weight)
        model.mx.eval(value)
        counters[kind+'_wall_seconds'] += time.perf_counter()-started
        counters[kind+'_calls'] += 1
        return value
    native.linear_weight = measured
    try:
        yield counters
    finally:
        native.linear_weight = original


def configuration(args, metadata, variant):
    from airllm.nextgen.config import NextConfig
    quantized = None if variant == 'original' else str(args.pack_root/f'quantized-granite-{variant}')
    return NextConfig(device=args.device, dtype='float32', cache_bytes=args.cache_mib*MiB,
        io_bytes=32*MiB, kv_bytes=32*MiB, workspace_bytes=256*MiB, reserve_bytes=1536*MiB,
        max_tensor_bytes=32*MiB, row_chunk=2048, prefill_chunk=64, max_context=512,
        policy='adaptive', prefetch=False, spec_prefetch=False, load_tokenizer=False,
        output_tile_cache=True, use_saved_tuning=False, cache_dir=str(metadata), quantized_dir=quantized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--pack-root', type=Path, default=ROOT/'work')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['convert', 'benchmark', 'all'], default='benchmark')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--prompts', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    parser.add_argument('--conversion-row-chunk', type=int, default=128)
    parser.add_argument('--profile-linear', action='store_true', help='Extra synchronized profiling pass; excluded from primary throughput')
    args = parser.parse_args()
    if not 1 <= args.prompts <= len(PROMPTS) or min(args.tokens, args.repeats, args.conversion_row_chunk) < 1 or args.cache_mib < 0:
        parser.error('Use 1..4 prompts, positive tokens/repeats/row chunk and a nonnegative cache')
    if 'original' not in args.variants:
        parser.error('original is required to define the teacher-forced baseline continuation')
    args.model_dir, args.pack_root, args.output = args.model_dir.resolve(), args.pack_root.resolve(), args.output.resolve()
    if args.pack_root.is_relative_to(args.model_dir) or args.output.is_relative_to(args.model_dir):
        parser.error('Sidecars and reports must be outside the original model directory')
    import psutil
    from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel, normalize_granite_config
    from airllm.nextgen.sidecar import QuantizedSidecar, convert_experts
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.physical_io import physical_read_counter_info
    from airllm.nextgen.tuning import telemetry
    config = normalize_granite_config(json.loads((args.model_dir/'config.json').read_text()))
    provenance = json.loads((args.model_dir/'download-provenance.json').read_text())
    original_hashes = {entry['name']: digest(args.model_dir/entry['name']) for entry in provenance['files']}
    if any(original_hashes[entry['name']] != entry['sha256'] for entry in provenance['files']):
        raise ValueError('Original checkpoint provenance mismatch')
    source_hashes, records, arrays = source_identity(), [], {}
    eos = config.get('eos_token_id')
    eos = [] if eos is None else [eos] if type(eos) is int else list(eos)
    tokenizer_hashes = {name: digest(args.model_dir/name) for name in
        ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json')
        if (args.model_dir/name).is_file()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, experiment='lossy-Granite-full-model-ablation',
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        checkpoint=provenance, original_files_sha256=original_hashes,
        tokenizer_files_sha256=tokenizer_hashes, source_sha256=source_hashes,
        host=platform.platform(), host_memory_bytes=psutil.virtual_memory().total,
        original_weights='Original BF16 checkpoint decoded to FP32; no FP32-rounded copy saved',
        precision_variants=dict(original='original BF16 values in FP32 arithmetic',
            expert4='all routed expert matrices affine Q4, others original',
            mixed='routed experts affine Q4, eligible attention/linear matrices Q8; router/embedding/norm original'),
        general_corpus_quality_validated=False, giant_model_validated=False, physical_external_ssd_measured=False,
        exactness_threshold=None, quality_scope='KL/top1/NLL on baseline-generated continuations of the listed prompts only',
        os_cache='Original SHA256 and sidecar verification warm file pages; OS cache is not purged',
        cache_state='Weight caches cleared before every measurement; identical configured budgets, no fixed per-layer quota',
        primary_timing_scope='Uninstrumented generation wall time including prefill; excludes construction, conversion and teacher forcing',
        eos_policy=dict(stop_at_first_eos=True, eos_token_ids=eos, max_new_tokens=args.tokens,
            terminal_eos_included_in_new_tokens=True, post_eos_generation=False),
        generation_length_comparison='Variants may stop at different lengths; report actual new_tokens/seconds. Raw wall-time ratios do not compare equal work.',
        teacher_forcing_reference='Original EOS-stopped continuation and logits prepared before primary timed variant runs; never supplied to generation.',
        variant_order_policy='All variants including original shuffled deterministically for every case/repeat',
        linear_profile_scope='Optional extra pass: synchronized per-linear dependency-inclusive wall time, neither pure QMM kernel time nor primary throughput',
        dequantization_accounting='reader dequant_seconds counts explicit source FP8 decoding; affine sidecar scale/bias work is integrated into QMM and is not separately timed',
        physical_counter=physical_read_counter_info(), sidecars={}, records=records)
    def save():
        report.update(source_unchanged=source_hashes == source_identity(), telemetry=telemetry())
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='quantized-granite-') as metadata:
        with OffsetCheckpoint(args.model_dir, metadata, max_read_bytes=32*MiB) as cp:
            report['base_fingerprint'] = cp.fingerprint
            for variant in args.variants:
                if variant == 'original':
                    continue
                scope = 'experts' if variant == 'expert4' else 'mixed'
                destination = args.pack_root/f'quantized-granite-{variant}'
                started = time.perf_counter()
                if args.phase in ('convert', 'all'):
                    print(json.dumps(dict(conversion_started=variant, destination=str(destination))), flush=True)
                    sidecar = convert_experts(cp, config, destination, bits=4, group_size=args.group_size,
                        row_chunk=args.conversion_row_chunk, scope=scope, linear_bits=8)
                else:
                    sidecar = QuantizedSidecar(destination, cp.fingerprint)
                    if sidecar.group_size != args.group_size:
                        raise ValueError('Sidecar group size differs from --group-size')
                summary = audit_sidecar(cp, sidecar, scope)
                summary.update(destination=str(destination), conversion_or_audit_seconds=time.perf_counter()-started,
                               integrity_read_bytes=sidecar.stats['integrity_bytes'])
                report['sidecars'][variant] = summary
                save()
                print(json.dumps(dict(sidecar_ready=variant, matrices=summary['matrices'], seconds=summary['conversion_or_audit_seconds'])), flush=True)
        if args.phase in ('benchmark', 'all'):
            import mlx.core as mx
            from transformers import AutoTokenizer
            mx.set_default_device(getattr(mx, args.device))
            report['libraries'] = {name: importlib.metadata.version(name) for name in ('mlx', 'transformers', 'numpy', 'safetensors')}
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
            report['prompts'] = PROMPTS[:args.prompts]
            references = {}
            report['reference_preparation'] = []
            # Prepare quality references outside the primary timing campaign.
            # Timed original runs participate in the same shuffle as variants.
            for case, prompt in enumerate(PROMPTS[:args.prompts]):
                ids = tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                cfg = configuration(args, metadata, 'original')
                model = AirLLMNextGraniteMoeModel(args.model_dir, next_config=cfg)
                try:
                    generated, preparation = measure(model, lambda: model.generate([ids], max_new_tokens=args.tokens))
                    baseline_tokens = generated.tolist()[0]
                    baseline_new_tokens = len(baseline_tokens)-len(ids)
                    logits, forced = measure(model, lambda: model.forward(
                        [baseline_tokens[:-1]], logits_to_keep=baseline_new_tokens).logits)
                    baseline_logits = np.asarray(logits).copy()
                    references[case] = (ids, baseline_tokens, baseline_logits)
                    report['reference_preparation'].append(dict(case=case,
                        excluded_from_primary_timing=True, generation=preparation,
                        new_tokens=baseline_new_tokens, teacher_forced=forced,
                        generation_summary=generation_summary(baseline_tokens, len(ids), baseline_tokens,
                            preparation['seconds'], eos)))
                    arrays[f'c{case}_original_reference_tokens'] = np.asarray(baseline_tokens, dtype=np.int32)
                    arrays[f'c{case}_original_reference_logits'] = baseline_logits
                    save()
                finally:
                    model.close(save_profile=False)
                    mx.clear_cache()
            for repeat in range(args.repeats):
                for case, prompt in enumerate(PROMPTS[:args.prompts]):
                    ids, baseline_tokens, baseline_logits = references[case]
                    baseline_new_tokens = len(baseline_tokens)-len(ids)
                    variants = list(args.variants)
                    random.Random(20261023+repeat*len(PROMPTS)+case).shuffle(variants)
                    for variant in variants:
                        cfg = configuration(args, metadata, variant)
                        started = time.perf_counter()
                        model = AirLLMNextGraniteMoeModel(args.model_dir, next_config=cfg)
                        construction = time.perf_counter()-started
                        try:
                            generated, generation = measure(model, lambda: model.generate([ids], max_new_tokens=args.tokens))
                            tokens = generated.tolist()[0]
                            logits, forced = measure(model, lambda: model.forward(
                                [baseline_tokens[:-1]], logits_to_keep=baseline_new_tokens).logits)
                            host_logits = np.asarray(logits).copy()
                            quality = compare_logits(baseline_logits, host_logits, [baseline_tokens[len(ids):]])
                            generation.update(generation_summary(tokens, len(ids), baseline_tokens,
                                generation['seconds'], eos),
                                text=tokenizer.decode(tokens[len(ids):], skip_special_tokens=True))
                            row = dict(repeat=repeat, case=case, variant=variant, prompt=prompt, options=cfg.as_dict(),
                                construction_seconds=construction, generation=generation, teacher_forced=forced, quality=quality)
                            if args.profile_linear:
                                with linear_profile(model) as profile:
                                    profiled, measured = measure(model, lambda: model.generate([ids], max_new_tokens=args.tokens))
                                row['linear_profile'] = dict(profile, generation_seconds=measured['seconds'],
                                    tokens_match_uninstrumented=profiled.tolist()[0] == tokens,
                                    new_tokens=len(profiled.tolist()[0])-len(ids),
                                    reader_delta=measured['reader_delta'])
                            records.append(row)
                            stem = f'r{repeat}c{case}{variant}'
                            arrays[stem+'_teacher_forced_logits'] = host_logits
                            arrays[stem+'_generated_tokens'] = np.asarray(tokens, dtype=np.int32)
                            save()
                            print(json.dumps(dict(case=case, repeat=repeat, variant=variant,
                                seconds=generation['seconds'], top1_agreement=quality['top1_agreement'],
                                max_abs_logit_error=quality['max_abs_logit_error'])), flush=True)
                        finally:
                            model.close(save_profile=False)
                            mx.clear_cache()
            np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
            report['raw_arrays_sha256'] = digest(args.output.with_suffix('.npz'))
    report['original_files_unchanged'] = all(digest(args.model_dir/name) == value for name, value in original_hashes.items())
    report['tokenizer_files_unchanged'] = all(digest(args.model_dir/name) == value for name, value in tokenizer_hashes.items())
    save()
    if not report['source_unchanged'] or not report['original_files_unchanged'] or not report['tokenizer_files_unchanged']:
        raise SystemExit('Source/original checkpoint changed during experiment; evidence preserved')


if __name__ == '__main__':
    main()
