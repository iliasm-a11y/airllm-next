#!/usr/bin/env python3
"""Summarize guarded MoE campaign evidence without pooling incompatible runs.

Examples:
    python benchmarks/synthesize_multitoken.py --output evidence-multitoken/synthesis.json
    python benchmarks/synthesize_multitoken.py evidence-multitoken/generation-cpu-256.json

Only exact case+repeat pairs within the same evidence file are compared.
Sequence seeds never enter gains. Generation rates include prefill and drafting;
controlled batch rates count verified positions, not emitted generation tokens.
Unknown transport stays unknown. OS-accounted reads are not NAND/device bytes.
The source manifest recorded for a historical run need not match today's code.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


PHASES = ('batch', 'generation', 'cache', 'combination')
BUDGET_FIELDS = ('cache_bytes', 'io_bytes', 'kv_bytes', 'workspace_bytes')
ROOT = Path(__file__).resolve().parents[1]


def finite(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def known_sum(values):
    values = list(values)
    return sum(values) if values and all(finite(x) for x in values) else None


def ratio(numerator, denominator):
    return numerator / denominator if finite(numerator) and finite(denominator, positive=True) else None


def median(values):
    values = [x for x in values if x is not None]
    return statistics.median(values) if values else None


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity(report):
    encoded = json.dumps(report['source_sha256'], sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def variant_key(row, phase):
    if phase == 'batch':
        return (row.get('variant'), row.get('batch_size'))
    if row.get('kind') == 'baseline':
        return ('autoregressive',)
    if phase == 'cache':
        return (row.get('variant'),)
    return (row.get('variant'), row.get('depth'), row.get('adaptive'),
            row.get('prefetch'), row.get('spec_prefetch'))


def row_key(row, phase):
    return (row.get('case'), row.get('repeat'), *variant_key(row, phase))


def expected_keys(report):
    """Detect interrupted evidence from the benchmark's declared task grid."""
    args, phase = report.get('arguments', {}), report['phase']
    prompts, repeats = args.get('prompts'), args.get('repeats')
    if type(prompts) is not int or prompts < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError('Missing valid prompts/repeats in arguments')
    keys = set()
    for case in range(prompts):
        for repeat in range(repeats):
            if phase == 'batch':
                for size in args['batch_sizes']:
                    for variant in ('tokenwise', 'serial_positions', 'union'):
                        keys.add((case, repeat, variant, size))
            elif phase == 'cache':
                keys.update((case, repeat, policy) for policy in args['policies'])
            else:
                keys.add((case, repeat, 'autoregressive'))
                choices = ([(depth, False, False, False) for depth in args['depths']]
                           + [(32, True, False, False)] if phase == 'generation' else
                           [(4, False, False, False), (4, False, True, False),
                            (4, False, True, True), (32, True, True, False)])
                for draft in args['drafts']:
                    for choice in choices:
                        keys.add((case, repeat, draft, *choice))
    return keys


def campaign_rows(report):
    phase = report['phase']
    kinds = {'batch': {'batch'}, 'cache': {'cache'},
             'generation': {'baseline', 'speculation'},
             'combination': {'baseline', 'speculation'}}[phase]
    # Legacy initial AR rows lack repeat; they are sequence seeds, not controls.
    return [row for row in report.get('records', []) if row.get('kind') in kinds
            and type(row.get('repeat')) is int]


def guards(report, path):
    errors, warnings = [], []
    if report.get('source_unchanged') is not True:
        errors.append('source_unchanged must be true')
    manifest = report.get('source_sha256')
    if not isinstance(manifest, dict) or not manifest or any(
            not isinstance(v, str) or len(v) != 64
            or any(c not in '0123456789abcdef' for c in v) for v in manifest.values()):
        errors.append('Missing valid source SHA256 manifest')
    if report.get('failures') != []:
        errors.append('Numerical failures present, or failures list missing')
    if report.get('completed') is False:
        errors.append('Campaign explicitly incomplete')
    phase = report.get('phase')
    if phase not in PHASES:
        return ['Not a recognized campaign report'], warnings
    rows = campaign_rows(report)
    counts = Counter(row_key(row, phase) for row in rows)
    if any(count != 1 for count in counts.values()):
        errors.append('Duplicate case/repeat/variant keys')
    try:
        expected = expected_keys(report)
        missing, extra = expected - counts.keys(), counts.keys() - expected
        if missing:
            errors.append(f'Incomplete task grid: {len(missing)} missing records')
        if extra:
            errors.append(f'Unexpected task grid: {len(extra)} extra records')
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f'Cannot establish campaign completion: {error}')
    if phase == 'batch':
        comparison_keys = Counter()
        for comparison in report.get('comparisons', []):
            key = (comparison.get('case'), comparison.get('repeat'),
                   comparison.get('variant'), comparison.get('batch_size'))
            comparison_keys[key] += 1
            if not all(comparison.get(name) is True for name in
                       ('logits_close', 'argmax_equal', 'route_sets_equal')):
                errors.append(f'Invalid batch exactness comparison: {key}')
        if any(comparison_keys[key] != 1 for key in counts):
            errors.append('Missing or duplicate batch exactness comparisons')
        arrays_path = path.with_suffix('.npz')
        if not arrays_path.is_file() or report.get('raw_arrays_sha256') != digest(arrays_path):
            errors.append('Missing or mismatched raw batch array hash')
    else:
        if any(row.get('tokens_match_baseline') is not True for row in rows):
            errors.append('Missing or false token exactness flag')
        if any(row.get('variant') == 'smollm' for row in rows):
            provenance = report.get('draft_checkpoint', {}).get('files_sha256')
            if not isinstance(provenance, dict) or not provenance:
                errors.append('Missing trained draft checkpoint hashes')
    if report.get('eos_policy') is None:
        warnings.append('EOS policy not recorded; horizon semantics remain unknown')
        if report.get('primary_campaign') is True and phase != 'batch':
            errors.append('Primary generation/cache evidence requires a declared EOS policy')
    if report.get('physical_external_ssd_measured') is not True:
        warnings.append('External SSD not validated by this run')
    if report.get('process_peak_rss_bytes') is not None:
        warnings.append('Process peak RSS is campaign-wide, not a variant memory peak')
    return list(dict.fromkeys(errors)), warnings


def budget(row, report):
    target = row.get('target_options', row.get('options', report.get('baseline_options', {})))
    draft = row.get('draft_options', {})
    if any(not finite(target.get(field)) or (draft and not finite(draft.get(field)))
           for field in BUDGET_FIELDS):
        return None
    return {field: target[field] + draft.get(field, 0) for field in BUDGET_FIELDS}


def observation(row, report):
    """Return additive observations; never average already normalized byte rates."""
    phase = report['phase']
    units = row.get('positions') if phase == 'batch' else row.get('new_tokens')
    seconds = row.get('seconds')
    if type(units) is not int or units < 1 or not finite(seconds, positive=True):
        raise ValueError('Invalid actual token/position count or wall time')
    runtime = row.get('runtime', {})
    speculation = runtime.get('speculative', {})
    if phase == 'batch':
        if units != row.get('batch_size'):
            raise ValueError('Actual verified positions differ from batch_size')
    else:
        tokens = row.get('tokens')
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError('Missing actual generated token sequence')
        if runtime.get('tokens_generated') != units:
            raise ValueError('Actual output count differs from runtime counter')
        if len(tokens) - runtime.get('prefill_tokens', len(tokens) + 1) != units:
            raise ValueError('Output length does not match prompt and emitted count')
    target_logical = row.get('io_delta', {}).get('logical_bytes')
    target_encoded = (speculation.get('encoded_file_bytes') if speculation else
                      runtime.get('encoded_file_bytes_per_token_including_prefill'))
    if not speculation and finite(target_encoded):
        target_encoded *= units
    if phase == 'batch':
        # These campaign phases use original checkpoints. No invented relation
        # between logical bytes and future packed/quantized file bytes is used.
        target_encoded = None
    if row.get('kind') == 'speculation':
        total_logical = speculation.get('total_logical_bytes')
        total_encoded = speculation.get('total_encoded_file_bytes')
    else:
        total_logical, total_encoded = target_logical, target_encoded
    experts = row.get('expert_delta', {})
    draft_runtime = row.get('draft_metrics', {}).get('runtime', {})
    target_kv, draft_kv = row.get('kv_peak_bytes'), draft_runtime.get('kv_peak_bytes')
    if not row.get('draft_options'):
        draft_kv = 0
    result = dict(units=units, seconds=seconds,
        target_logical_bytes=target_logical, total_logical_bytes=total_logical,
        target_encoded_file_bytes=target_encoded, total_encoded_file_bytes=total_encoded,
        os_accounted_storage_read_bytes=row.get('physical_read_bytes'),
        proposed=speculation.get('proposals'), accepted=speculation.get('accepted_draft_tokens'),
        draft_seconds=speculation.get('draft_seconds'),
        verification_seconds=speculation.get('verification_seconds'),
        verifier_passes=speculation.get('rounds'),
        target_forward_calls=speculation.get('target_forward_calls'),
        target_forward_positions=speculation.get('target_forward_positions'),
        rollback_tokens=speculation.get('rollback_tokens'),
        layer_batches=experts.get('batches'), distinct_expert_sum=experts.get('distinct_expert_sum'),
        target_kv_peak_bytes=target_kv,
        sum_target_draft_individual_kv_peaks=known_sum([target_kv, draft_kv]),
        disabled_reason=speculation.get('disabled_reason'))
    for name, value in result.items():
        if name not in ('disabled_reason',) and value is not None and not finite(value):
            raise ValueError(f'Invalid negative/non-finite observation: {name}')
    if speculation:
        if speculation.get('tokens_generated') != units:
            raise ValueError('Speculative output count differs from actual count')
        if not finite(result['proposed']) or not finite(result['accepted']) or result['accepted'] > result['proposed']:
            raise ValueError('Invalid proposal/acceptance counts')
        if finite(total_logical) and finite(target_logical) and total_logical < target_logical:
            raise ValueError('Total target+draft bytes below target bytes')
    return result


def pooled(observations):
    units, seconds = sum(row['units'] for row in observations), sum(row['seconds'] for row in observations)
    result = dict(samples=len(observations), actual_units=units, total_wall_seconds=seconds,
                  pooled_units_per_second=units / seconds,
                  median_sample_units_per_second=median(row['units'] / row['seconds'] for row in observations))
    for field in ('target_logical_bytes', 'total_logical_bytes', 'target_encoded_file_bytes',
                  'total_encoded_file_bytes', 'os_accounted_storage_read_bytes'):
        total = known_sum(row[field] for row in observations)
        result[field] = total
        result[field + '_per_actual_unit'] = ratio(total, units)
        result[field + '_known_samples'] = sum(row[field] is not None for row in observations)
    for field in ('proposed', 'accepted', 'draft_seconds', 'verification_seconds', 'verifier_passes',
                  'target_forward_calls', 'target_forward_positions', 'rollback_tokens',
                  'layer_batches', 'distinct_expert_sum'):
        result[field] = known_sum(row[field] for row in observations)
    result['acceptance_rate_weighted_by_proposals'] = ratio(result['accepted'], result['proposed'])
    result['mean_distinct_experts_per_layer_batch'] = ratio(result['distinct_expert_sum'], result['layer_batches'])
    result['target_kv_peak_bytes_max'] = max((row['target_kv_peak_bytes'] for row in observations
                                             if row['target_kv_peak_bytes'] is not None), default=None)
    result['sum_individual_kv_peaks_max_not_simultaneous_peak'] = max(
        (row['sum_target_draft_individual_kv_peaks'] for row in observations
         if row['sum_target_draft_individual_kv_peaks'] is not None), default=None)
    result['disabled_reasons'] = dict(Counter(row['disabled_reason'] for row in observations if row['disabled_reason']))
    return result


def compare_pairs(pairs):
    """Paired median is primary; pooled totals are also reported explicitly."""
    variants, baselines, details = [], [], []
    for row, baseline, candidate, reference in pairs:
        variants.append(candidate)
        baselines.append(reference)
        speedup = (candidate['units'] / candidate['seconds']) / (reference['units'] / reference['seconds'])
        detail = dict(case=row['case'], repeat=row['repeat'], throughput_speedup=speedup,
                      variant_units=candidate['units'], baseline_units=reference['units'])
        for field in ('target_logical_bytes', 'total_logical_bytes', 'total_encoded_file_bytes',
                      'os_accounted_storage_read_bytes'):
            candidate_rate = ratio(candidate[field], candidate['units'])
            baseline_rate = ratio(reference[field], reference['units'])
            relative = ratio(candidate_rate, baseline_rate)
            detail[field + '_per_unit_reduction'] = 1 - relative if relative is not None else None
        details.append(detail)
    result = dict(pairs=len(pairs), case_count=len({row['case'] for row, _, _, _ in pairs}),
                  repeat_count=len({row['repeat'] for row, _, _, _ in pairs}),
                  median_paired_throughput_speedup=median(row['throughput_speedup'] for row in details),
                  min_paired_throughput_speedup=min(row['throughput_speedup'] for row in details),
                  max_paired_throughput_speedup=max(row['throughput_speedup'] for row in details),
                  variant=pooled(variants), baseline=pooled(baselines), paired_observations=details)
    result['pooled_throughput_speedup'] = result['variant']['pooled_units_per_second'] / result['baseline']['pooled_units_per_second']
    for field in ('target_logical_bytes', 'total_logical_bytes', 'total_encoded_file_bytes',
                  'os_accounted_storage_read_bytes'):
        result['median_paired_' + field + '_per_unit_reduction'] = median(
            row[field + '_per_unit_reduction'] for row in details)
    return result


def summarize_report(report, path):
    errors, warnings = guards(report, path)
    header = dict(file=str(path), phase=report.get('phase'), accepted=not errors,
                  errors=errors, warnings=warnings, groups=[])
    if errors:
        return header
    header.update(source_manifest_sha256=source_identity(report),
        arguments=report['arguments'], checkpoint=report.get('checkpoint'),
        draft_checkpoint=report.get('draft_checkpoint'), precision=report.get('precision'),
        eos_policy=report.get('eos_policy', 'unknown'), os_cache=report.get('os_cache'),
        primary_campaign=report.get('primary_campaign'),
        controlled_input=report.get('batch_control_text', report.get('controlled_input', report.get('batch_input'))),
        physical_counter=report.get('physical_counter'),
        process_peak_rss_bytes_campaign=report.get('process_peak_rss_bytes'),
        physical_external_ssd_measured=report.get('physical_external_ssd_measured', False),
        giant_model_validated=report.get('giant_model_validated', False))
    phase, rows = report['phase'], campaign_rows(report)
    indexed = {row_key(row, phase): row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        if phase == 'batch':
            references = (['tokenwise', 'serial_positions'] if row['variant'] == 'union' else
                          ['tokenwise'] if row['variant'] == 'serial_positions' else [])
            reference_keys = [(row['case'], row['repeat'], ref, row['batch_size']) for ref in references]
        elif phase == 'cache':
            reference_keys = [(row['case'], row['repeat'], 'adaptive')] if row['variant'] != 'adaptive' else []
        else:
            reference_keys = [(row['case'], row['repeat'], 'autoregressive')] if row['kind'] == 'speculation' else []
        for reference_key in reference_keys:
            baseline = indexed.get(reference_key)
            if baseline is None:
                errors.append(f'Missing paired reference: {reference_key}')
                continue
            try:
                if phase != 'batch' and row['tokens'] != baseline['tokens']:
                    raise ValueError('Paired output token sequences differ')
                candidate_budget, reference_budget = budget(row, report), budget(baseline, report)
                if candidate_budget is None or candidate_budget != reference_budget:
                    raise ValueError('Unknown or unequal paired target+draft budgets')
                candidate, reference = observation(row, report), observation(baseline, report)
                if candidate['units'] != reference['units']:
                    raise ValueError('Paired actual output/position counts differ')
                key = (variant_key(row, phase), variant_key(baseline, phase))
                groups[key].append((row, baseline, candidate, reference))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f'Invalid pair {row_key(row, phase)}: {error}')
    # Fail closed: no gains from a report with even one invalid pair.
    if errors:
        header['accepted'] = False
        header['errors'] = list(dict.fromkeys(errors))
        return header
    for (variant, reference), pairs in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(variant_key=list(variant), baseline_key=list(reference),
                    actual_unit='verified_causal_position' if phase == 'batch' else 'emitted_token',
                    time_scope='controlled_verification_only' if phase == 'batch' else 'full_generation_including_prefill_and_draft',
                    budget_total_bytes=budget(pairs[0][0], report))
        item.update(compare_pairs(pairs))
        header['groups'].append(item)
    if not groups:
        header['accepted'] = False
        errors.append('No matched comparison groups')
    return header


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='*')
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'evidence-multitoken')
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence-multitoken' / 'synthesis.json')
    parser.add_argument('--primary-only', action='store_true',
                        help='Skip reports unless primary_campaign is explicitly true')
    args = parser.parse_args(argv)
    paths = sorted(set(args.inputs or [path for phase in PHASES
                                      for path in args.evidence_dir.glob(phase + '-*.json')
                                      if not path.name.endswith('.guard.json')]))
    reports = []
    for path in paths:
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict):
                raise ValueError('Evidence must be a JSON object')
            if args.primary_only and report.get('primary_campaign') is not True:
                reports.append(dict(file=str(path), accepted=False, skipped=True,
                                    reason='primary_campaign is not explicitly true', groups=[]))
                continue
            reports.append(summarize_report(report, path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            reports.append(dict(file=str(path), accepted=False, errors=[str(error)], groups=[]))
    result = dict(schema_version=1, reports=reports,
        accepted_reports=sum(report['accepted'] for report in reports),
        excluded_reports=sum(not report['accepted'] and not report.get('skipped') for report in reports),
        skipped_reports=sum(bool(report.get('skipped')) for report in reports),
        notes=['No pooling across evidence files, devices, source manifests, budgets or EOS policies.',
               'Paired median throughput ratios use identical case+repeat controls.',
               'Byte totals are summed before dividing by actual emitted tokens or verified positions.',
               'Unknown byte counters remain null, including zero-denominator reductions.',
               'Batch position throughput is not a generation speed claim.',
               'Source guards verify recorded-run consistency, not equality to the current checkout.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), accepted_reports=result['accepted_reports'],
                          excluded_reports=result['excluded_reports'],
                          skipped_reports=result['skipped_reports'],
                          groups=sum(len(report['groups']) for report in reports))))
    return 0 if result['accepted_reports'] and not result['excluded_reports'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
