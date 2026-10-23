#!/usr/bin/env python3
"""Paired, opt-in E1/E2 empty-draft retry experiment on original Granite.

The four prompts, variants, 8-token probe interval and 1% preparation gate are
declared before generation. E1 and E2 share one source version and differ only
in retry_empty_drafts. All variants use ordinary prefetch, original weights,
normal EOS, and equal configured payload budgets. Probe costs and completion
of outstanding prefetch belong to measured generation time. No speedup or
general quality conclusion is inferred from these four selected prompts.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
sys.path.insert(0, str(ROOT/'benchmarks'))
import benchmark_multitoken_moe as base

CASES = (
    ('known_red_blue', 'Repeat exactly this pattern several times: red blue green red blue green'),
    ('new_copper_sand_olive', 'Repeat the cycle "cuivre | sable | olive" eight times, using the same separators and no introduction.'),
    ('four_line_copy', 'Copy the following sentence on four separate lines, with no introduction:\nThe amber train crosses the quiet valley.'),
    ('negative_metal_wood_explanation', 'Explain in a few sentences why a metal spoon feels colder than a wooden spoon when both are at the same room temperature.'),
)
VARIANTS = ('autoregressive', 'fixed8', 'adaptive_e1', 'adaptive_e2')
DEPTHS = (2, 4, 8, 16, 32)
SEED = 20261023
TOKEN_CAP = 64
RETRY_INTERVAL = 8
RETRY_COST_FRACTION = .01


def sources():
    return base.sources() | {str(Path(__file__).resolve().relative_to(ROOT)): base.digest(__file__)}


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def scheduled_blocks(prompts, repeats, seed=SEED):
    """Keep each case/repeat paired; randomize blocks and all four variants."""
    rng = random.Random(seed)
    blocks = [(repeat, case) for repeat in range(repeats) for case in range(prompts)]
    rng.shuffle(blocks)
    result = []
    for repeat, case in blocks:
        order = list(VARIANTS)
        rng.shuffle(order)
        result.append(dict(repeat=repeat, case=case, variants=order))
    return result


def validate_tokens(prompt, tokens, cap, eos):
    continuation = tokens[len(prompt):]
    if tokens[:len(prompt)] != prompt or not 1 <= len(continuation) <= cap:
        raise ValueError('Generation did not preserve the prompt or exceeded the token cap')
    eos_positions = [i for i, token in enumerate(continuation) if token in eos]
    if eos_positions and eos_positions != [len(continuation)-1]:
        raise ValueError('Generation continued after its first EOS')
    if not eos_positions and len(continuation) != cap:
        raise ValueError('Generation stopped before EOS or the declared token cap')
    return dict(new_tokens=len(continuation), eos_count=len(eos_positions), tokens_after_first_eos=0,
                stop_reason='eos' if eos_positions else 'max_new_tokens', tokens_sha256=token_digest(tokens))


def measure_generation(model, ids, variant, cap, eos):
    """Measure one fresh-cache run, including draft work and pending read drain."""
    from airllm.nextgen.physical_io import physical_read_bytes
    from airllm.nextgen.prompt_draft import PromptSpanDraft
    from airllm.nextgen.storage import encoded_read_bytes
    base.clear_weights(model)
    captured = {}
    original_session = model.new_session
    def capture_session():
        cache = original_session()
        captured['cache'] = cache
        return cache
    model.new_session = capture_session
    draft = None if variant == 'autoregressive' else PromptSpanDraft(min_match=2, max_match=32)
    before = base.native_snapshot(model)
    encoded_before = encoded_read_bytes(model.checkpoint)
    physical_before = physical_read_bytes()
    started = time.perf_counter()
    try:
        if draft is None:
            output = model.generate([ids], max_new_tokens=cap, eos_token_id=None)
        else:
            adaptive = variant.startswith('adaptive_')
            output = model.generate_speculative([ids], draft, max_new_tokens=cap, eos_token_id=None,
                max_depth=32 if adaptive else 8, adaptive=adaptive,
                candidate_depths=DEPTHS if adaptive else None,
                retry_empty_drafts=variant == 'adaptive_e2')
        values = output.tolist()[0]
        # Count completion of unused lookahead that this run actually issued.
        # Draining after the timer would hide its service time and race byte
        # snapshots; this identical rule applies to AR and every draft variant.
        model.reader.drain()
        elapsed = time.perf_counter()-started
        physical_after = physical_read_bytes()
        encoded_after = encoded_read_bytes(model.checkpoint)
        after = base.native_snapshot(model)
        metrics = model.metrics()
        cache = captured['cache']
        token_info = validate_tokens(ids, values, cap, eos)
        budget = model.options
        guards = dict(
            cache_bytes=0 <= model.reader.cache.bytes <= budget.cache_bytes,
            cache_peak_bytes=model.reader.cache.peak_bytes <= budget.cache_bytes,
            io_peak_bytes=model.reader.io.peak_bytes <= budget.io_bytes,
            io_fully_drained=model.reader.io.bytes == 0,
            kv_bytes=0 <= cache.bytes <= budget.kv_bytes,
            kv_peak_bytes=cache.peak_bytes <= budget.kv_bytes,
            workspace_reservation=after['experts']['workspace_peak_bytes'] <= budget.workspace_bytes,
            cache_position=cache.position == len(values)-1,
            prefetch_enabled=model.reader.prefetch_enabled is True,
            original_weights=model.reader.sidecar is None and budget.lossless_dir is None,
        )
        io_delta = base.delta(after['io'], before['io'])
        return dict(seconds=elapsed, tokens=values, tokens_per_second=token_info['new_tokens']/elapsed,
            **token_info, guards=guards, runtime=metrics['runtime'], cache=metrics['cache'],
            compact_cache=metrics['compact_cache'], io_scheduler=metrics['prefetch'],
            weights=metrics['weights'], reader_delta=base.delta(after['loads'], before['loads']),
            expert_delta=base.delta(after['experts'], before['experts']), io_delta=io_delta,
            encoded_file_bytes=encoded_after-encoded_before,
            logical_bytes_per_generated_token=io_delta.get('logical_bytes', 0)/token_info['new_tokens'],
            process_physical_read_bytes=physical_after-physical_before
                if physical_before is not None and physical_after is not None else None,
            kv=dict(bytes=cache.bytes, peak_bytes=cache.peak_bytes, budget_bytes=budget.kv_bytes,
                    committed_tokens=cache.position, limit=cache.limit),
            expert_workspace_reservation_peak_bytes=after['experts']['workspace_peak_bytes'])
    finally:
        model.new_session = original_session


def paired_comparisons(records):
    groups = {}
    for row in records:
        groups.setdefault((row['repeat'], row['case']), {})[row['variant']] = row
    pairs = []
    for (repeat, case), variants in sorted(groups.items()):
        if set(variants) != set(VARIANTS):
            continue
        ar = variants['autoregressive']
        for variant in VARIANTS:
            row = variants[variant]
            pairs.append(dict(repeat=repeat, case=case, variant=variant,
                paired_speedup=ar['seconds']/row['seconds'], seconds=row['seconds'],
                paired_ar_seconds=ar['seconds'], tokens_equal_to_paired_ar=row['tokens'] == ar['tokens'],
                logical_bytes_ratio_to_ar=row['io_delta'].get('logical_bytes', 0)/ar['io_delta']['logical_bytes']
                    if ar['io_delta'].get('logical_bytes') else None))
    return pairs


def summarize(pairs):
    groups = {}
    for row in pairs:
        groups.setdefault((row['case'], row['variant']), []).append(row)
    return [dict(case=case, variant=variant, repetitions=len(rows),
        paired_speedup_median=statistics.median(row['paired_speedup'] for row in rows),
        paired_speedup_min=min(row['paired_speedup'] for row in rows),
        paired_speedup_max=max(row['paired_speedup'] for row in rows),
        all_tokens_equal=all(row['tokens_equal_to_paired_ar'] for row in rows))
        for (case, variant), rows in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--prompts', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=TOKEN_CAP)
    parser.add_argument('--io-mode', choices=['pread', 'pread_nocache'], default='pread')
    args = parser.parse_args()
    if not 1 <= args.prompts <= len(CASES) or args.repeats < 1 or not 1 <= args.tokens <= TOKEN_CAP:
        parser.error('Use 1..4 prompts, positive repeats and a token cap from 1 to 64')
    args.model_dir, args.output = args.model_dir.resolve(), args.output.resolve()
    args.cache_mib = 256  # Fixed across variants; not an experimental degree of freedom.
    if args.output.is_relative_to(args.model_dir):
        parser.error('Reports must be outside the original checkpoint directory')
    import mlx.core as mx
    import psutil
    from transformers import AutoTokenizer
    from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
    from airllm.nextgen.generation import eos_ids
    from airllm.nextgen.physical_io import physical_read_counter_info
    from airllm.nextgen.tuning import telemetry
    if 'retry_empty_drafts' not in inspect.signature(AirLLMNextGraniteMoeModel.generate_speculative).parameters:
        raise SystemExit('The opt-in E2 patch must be applied before this benchmark; no model inference started')
    mx.set_default_device(getattr(mx, args.device))
    identity = sources()
    provenance = json.loads((args.model_dir/'download-provenance.json').read_text())
    for entry in provenance['files']:
        if base.digest(args.model_dir/entry['name']) != entry['sha256']:
            raise ValueError('Original checkpoint identity mismatch')
    config = json.loads((args.model_dir/'config.json').read_text())
    eos = set(eos_ids(config.get('eos_token_id'), config['vocab_size']))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    prompts = [tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
        add_generation_prompt=True, return_dict=False) for _,text in CASES[:args.prompts]]
    order = scheduled_blocks(args.prompts, args.repeats)
    records, references, failures = [], [], []
    report = dict(schema_version=1, experiment='adaptive-empty-draft-retry-E1-E2',
        arguments={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
        primary_campaign=True, eos_policy='model_config_stop', source_sha256=identity, checkpoint=provenance,
        original_files_sha256={entry['name']: entry['sha256'] for entry in provenance['files']},
        host=platform.platform(), host_memory_bytes=psutil.virtual_memory().total,
        libraries={name: importlib.metadata.version(name) for name in ('mlx', 'torch', 'transformers', 'numpy', 'safetensors')},
        prompt_set=[dict(case=index, name=name, text=text, token_ids=prompts[index])
                    for index,(name,text) in enumerate(CASES[:args.prompts])],
        selection_scope='One known positive red/blue control, two new repetition probes and one negative; no general workload claim',
        variants=dict(autoregressive=dict(prefetch=True), fixed8=dict(prefetch=True, depth=8, adaptive=False, retry_empty_drafts=False),
            adaptive_e1=dict(prefetch=True, depths=DEPTHS, adaptive=True, retry_empty_drafts=False),
            adaptive_e2=dict(prefetch=True, depths=DEPTHS, adaptive=True, retry_empty_drafts=True)),
        predeclared_retry=dict(interval_generated_tokens=RETRY_INTERVAL, preparation_cost_fraction_below=RETRY_COST_FRACTION,
            transport='both observed draft logical and encoded deltas must be known zero'),
        random_seed=SEED, paired_block_order=order, records=records, references_outside_comparison=references, failures=failures,
        timing_scope='Generation, probes, output synchronization and completion/cancellation of outstanding prefetch; construction/reference validation excluded',
        os_cache='Checkpoint SHA256 verification warms pages; OS cache not purged; decoded weight cache starts empty each trial',
        physical_counter=physical_read_counter_info(), physical_external_ssd_measured=False, giant_model_validated=False,
        general_workload_speedup_validated=False, quality_corpus_validated=False,
        memory_budget_scope='Configured retained tensor/IO/KV/workspace payload allowances; Python metadata and allocator residency are separately observed, not a hard process RSS bound')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report.update(source_unchanged=identity == sources(), telemetry=telemetry(),
            paired_comparisons=paired_comparisons(records))
        report['summary'] = summarize(report['paired_comparisons'])
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    def run(ids, variant, cfg):
        available = psutil.virtual_memory().available
        if available < cfg.reserve_bytes+cfg.workspace_bytes:
            raise MemoryError('Configured system reserve/workspace guard reached before trial')
        started = time.perf_counter()
        model = base.construct(args.model_dir, cfg)
        construction = time.perf_counter()-started
        try:
            row = measure_generation(model, ids, variant, args.tokens, eos)
            row.update(construction_seconds=construction, options=cfg.as_dict())
            return row
        finally:
            base.close(model)
    with tempfile.TemporaryDirectory(prefix='adaptive-retry-') as metadata:
        cfg = base.configuration(args, metadata, prefetch=True)
        report['common_options'] = cfg.as_dict()
        for case, ids in enumerate(prompts):
            row = run(ids, 'autoregressive', cfg)
            row.update(case=case, case_name=CASES[case][0], role='correctness_reference_outside_primary_comparison')
            references.append(row)
            if not all(row['guards'].values()):
                failures.append(dict(case=case, reason='reference budget/cache guard', guards=row['guards']))
            save()
        for block in order:
            for position, variant in enumerate(block['variants']):
                case, repeat = block['case'], block['repeat']
                row = run(prompts[case], variant, cfg)
                row.update(case=case, repeat=repeat, variant=variant, case_name=CASES[case][0], paired_order=position,
                    tokens_match_reference=row['tokens'] == references[case]['tokens'])
                if not row['tokens_match_reference']:
                    failures.append(dict(case=case, repeat=repeat, variant=variant, reason='target greedy tokens differ'))
                if not all(row['guards'].values()):
                    failures.append(dict(case=case, repeat=repeat, variant=variant, reason='budget/cache guard', guards=row['guards']))
                speculative = row['runtime'].get('speculative')
                if speculative is not None:
                    declared = (speculative.get('retry_empty_drafts') is (variant == 'adaptive_e2')
                        and speculative.get('empty_retry_interval_tokens') == RETRY_INTERVAL
                        and speculative.get('empty_retry_cost_fraction') == RETRY_COST_FRACTION)
                    if not declared:
                        failures.append(dict(case=case, repeat=repeat, variant=variant, reason='retry opt-in/threshold declaration mismatch'))
                records.append(row)
                save()
                print(json.dumps(dict(case=case, repeat=repeat, variant=variant, seconds=row['seconds'],
                    new_tokens=row['new_tokens'], tokens_match_reference=row['tokens_match_reference'],
                    probes=speculative.get('empty_retry_probes') if speculative else None)), flush=True)
    report['original_files_unchanged'] = all(base.digest(args.model_dir/entry['name']) == entry['sha256'] for entry in provenance['files'])
    save()
    if not report['source_unchanged'] or not report['original_files_unchanged'] or failures:
        raise SystemExit('Source, identity, token or budget guard failed; experiment evidence preserved')


if __name__ == '__main__':
    main()
