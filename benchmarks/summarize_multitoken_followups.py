#!/usr/bin/env python3
"""Summarize existing E2/F JSON only; no model inference or weight reads."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard_summary(maps):
    missing = sum(not isinstance(row, dict) or not row for row in maps)
    failed = [{key: value for key,value in row.items() if value is not True}
              for row in maps if isinstance(row, dict) and row and not all(value is True for value in row.values())]
    return dict(measurements=len(maps), missing_guard_maps=missing, failures=failed,
        all_guards=False if failed else True if maps and not missing else None)


def provenance(path, report):
    guard_path = path.with_suffix('.guard.json')
    guard = read(guard_path) if guard_path.is_file() else None
    return dict(path=str(path), sha256=digest(path), experiment=report.get('experiment'),
        device=report.get('arguments', {}).get('device'),
        source_unchanged=report.get('source_unchanged'), original_files_unchanged=report.get('original_files_unchanged'),
        tokenizer_files_unchanged=report.get('tokenizer_files_unchanged'), sidecar_inputs_unchanged=report.get('sidecar_inputs_unchanged'),
        process_guard_exit_code=guard.get('exit_code') if guard else None,
        process_guard_path=str(guard_path) if guard else None)


def rates(generation, io=None):
    count, seconds = generation['new_tokens'], generation['seconds']
    if type(count) is not int or count < 1 or seconds <= 0:
        raise ValueError('Summary requires actual positive generated token counts and elapsed time')
    logical = (io if io is not None else generation.get('io_delta', {})).get('logical_bytes')
    return dict(new_tokens=count, seconds=seconds, tokens_per_second=count/seconds,
        logical_bytes=logical, logical_bytes_per_generated_token=logical/count if logical is not None else None)


def totals(rows):
    count = sum(row['new_tokens'] for row in rows)
    seconds = sum(row['seconds'] for row in rows)
    known = all(row['logical_bytes'] is not None for row in rows)
    logical = sum(row['logical_bytes'] for row in rows) if known else None
    return dict(runs=len(rows), actual_new_tokens=count, generation_seconds=seconds,
        aggregate_tokens_per_second=count/seconds if seconds else None,
        logical_bytes=logical, logical_bytes_per_generated_token=logical/count if logical is not None and count else None,
        median_tokens_per_second=statistics.median(row['tokens_per_second'] for row in rows))


def summarize_e2(path):
    report = read(path)
    records = report.get('records', [])
    pairs, by_case = {}, {}
    for row in records:
        pairs.setdefault((row['case'], row['repeat']), {})[row['variant']] = row
        by_case.setdefault((row['case'], row['variant']), []).append(row)
    rows = []
    for (case, variant), samples in sorted(by_case.items()):
        comparisons = []
        for sample in samples:
            ar = pairs[(case, sample['repeat'])].get('autoregressive')
            if ar is not None:
                equal = sample['tokens'] == ar['tokens']
                comparisons.append(dict(repeat=sample['repeat'], identical_tokens=equal,
                    paired_speedup=ar['seconds']/sample['seconds'] if equal else None))
        speedups = [row['paired_speedup'] for row in comparisons if row['paired_speedup'] is not None]
        speculative = [row.get('runtime', {}).get('speculative', {}) for row in samples]
        row = dict(case=case, case_name=samples[0].get('case_name'), variant=variant,
            **totals([rates(sample) for sample in samples]), paired= comparisons,
            paired_speedup_median=statistics.median(speedups) if speedups else None,
            paired_speedup_min=min(speedups) if speedups else None,
            paired_speedup_max=max(speedups) if speedups else None,
            tokens_equal_to_reference=all(sample.get('tokens_match_reference') is True for sample in samples),
            guards=guard_summary([sample.get('guards') for sample in samples]),
            verified_proposals=sum(value.get('proposals', 0) for value in speculative),
            accepted_draft_tokens=sum(value.get('accepted_draft_tokens', 0) for value in speculative),
            probes=sum(value.get('empty_retry_probes', 0) for value in speculative),
            reactivations=sum(value.get('empty_retry_reactivations', 0) for value in speculative),
            probe_seconds=sum(value.get('empty_retry_probe_seconds', 0.) for value in speculative),
            abandoned_probe_candidates=sum(value.get('empty_retry_abandoned_candidate_tokens', 0) for value in speculative))
        rows.append(row)
    ar_by_case = {row['case']: row for row in rows if row['variant'] == 'autoregressive'}
    for row in rows:
        original = ar_by_case.get(row['case'], {}).get('logical_bytes_per_generated_token')
        row['logical_bytes_per_token_ratio_to_ar'] = row['logical_bytes_per_generated_token']/original if original else None
    return dict(input=provenance(path, report), records=len(records),
        expected_records=report.get('arguments', {}).get('prompts', 0)*report.get('arguments', {}).get('repeats', 0)*4,
        failures=report.get('failures'), all_guards=guard_summary([row.get('guards') for row in records]), rows=rows)


def summarize_f(path):
    report = read(path)
    records = report.get('records', [])
    grouped, paired, guards = {}, {}, []
    for row in records:
        grouped.setdefault(row['variant'], []).append(row)
        paired.setdefault((row['case'], row['repeat']), {})[row['variant']] = row
        for kind in ('generation', 'teacher_forced', 'linear_profile'):
            if kind in row:
                guards.append(row[kind].get('guards'))
    rows = []
    for variant, samples in sorted(grouped.items()):
        comparisons = []
        for sample in samples:
            ar = paired[(sample['case'], sample['repeat'])].get('original')
            if ar is None:
                continue
            actual, reference = rates(sample['generation']), rates(ar['generation'])
            comparisons.append(dict(case=sample['case'], repeat=sample['repeat'],
                actual_new_tokens=actual['new_tokens'], original_new_tokens=reference['new_tokens'],
                identical_tokens=sample['generation']['tokens'] == ar['generation']['tokens'],
                observed_tokens_per_second_ratio_to_original=actual['tokens_per_second']/reference['tokens_per_second'],
                logical_bytes_per_token_ratio_to_original=actual['logical_bytes_per_generated_token']/reference['logical_bytes_per_generated_token']
                    if actual['logical_bytes_per_generated_token'] is not None and reference['logical_bytes_per_generated_token'] else None))
        positions = sum(sample['quality']['positions'] for sample in samples)
        rows.append(dict(variant=variant, **totals([rates(sample['generation']) for sample in samples]),
            paired_rate_comparisons=comparisons,
            teacher_forced_positions=positions,
            mean_kl_reference_to_variant=sum(sample['quality']['mean_kl_reference_to_variant']*sample['quality']['positions'] for sample in samples)/positions,
            top1_agreement=sum(sample['quality']['top1_agreement']*sample['quality']['positions'] for sample in samples)/positions,
            max_abs_logit_error=max(sample['quality']['max_abs_logit_error'] for sample in samples),
            all_profile_tokens_match=all(sample['linear_profile']['tokens_match_uninstrumented'] for sample in samples if 'linear_profile' in sample)
                if any('linear_profile' in sample for sample in samples) else None))
    return dict(input=provenance(path, report), records=len(records),
        profile_linear_enabled=report.get('arguments', {}).get('profile_linear', False),
        budget_guards_passed=report.get('budget_guards_passed'), budget_guard_failures=report.get('budget_guard_failures'),
        guards=guard_summary(guards), rows=rows,
        comparison_scope='Actual generated tokens/second and logical bytes/generated-token; lossy variants may generate different content and lengths. No unequal-work wall-time speedup is computed.',
        quality_scope=report.get('quality_scope'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e2', type=Path, default=ROOT/'evidence-multitoken/adaptive-retry-metal.json')
    parser.add_argument('--f', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, default=ROOT/'evidence-multitoken/repeated-and-f-summary.json')
    args = parser.parse_args()
    f_paths = args.f
    if not f_paths:
        for path in sorted((ROOT/'evidence-multitoken').glob('quantized*.json')):
            if path.name.endswith('.guard.json'):
                continue
            report = read(path)
            if report.get('experiment') == 'lossy-Granite-full-model-ablation' and report.get('records'):
                f_paths.append(path)
    result = dict(schema_version=1, analysis_only=True, model_inference_run=False,
        e2=summarize_e2(args.e2), f=[summarize_f(path) for path in f_paths],
        limits=['Four selected E2 prompts do not establish general workload acceleration',
            'F generation diverges under lossy weights; token rate is descriptive, not equal-answer speedup',
            'F continuation KL/top1 are not general corpus quality',
            'Missing historical memory/sidecar guards remain unknown, not a passing result'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), e2_rows=len(result['e2']['rows']), f_reports=len(result['f']))))


if __name__ == '__main__':
    main()
