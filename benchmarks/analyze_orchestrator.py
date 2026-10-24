"""Validate load invariants and summarize saved observations without inference."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, max(0, int((len(ordered)-1)*q)))]


def analyze(report):
    if report['status'] != 'completed':
        raise ValueError('Interrupted campaign cannot be presented as completed')
    rows = []
    for run in report['runs']:
        count = run['agents']
        admitted, final = run['admission_status'], run['final_status']
        assert admitted['registered_agents'] == count
        assert admitted['waiting_agents'] == count
        assert final['jobs_completed'] == count
        assert final['inferences_in_flight'] == 0
        assert final['kv_reserved_bytes'] == 0
        assert final['stopped_reason'] is None
        assert final['peak_simultaneous_inferences'] <= final['effective_capacity']
        assert all(j['state'] == 'completed' for j in run['jobs'])
        assert sum(len(j['output_ids']) for j in run['jobs']) == run['generated_tokens']
        if report['mode'] != 'synthetic':
            assert run['ordinary_serial_tokens_equal'] is True
            assert final['peak_simultaneous_inferences'] == 1
        ttft = [j['metrics']['ttft_seconds'] for j in run['jobs']
                if j['metrics']['ttft_seconds'] is not None]
        completion = [j['metrics']['completion_seconds'] for j in run['jobs']]
        resources = final['resource_metrics']
        rows.append(dict(mode=report['mode'], agents=count,
            seconds=run['total_seconds'], admission_seconds=run['admission_seconds'],
            drain_seconds=run['drain_seconds'], jobs_per_second=run['jobs_per_second'],
            tokens_per_second=run['tokens_per_second'],
            peak_inferences=final['peak_simultaneous_inferences'],
            transcript_reserved_mib=final['transcript_reserved_bytes']/1024**2,
            sampled_rss_mib=resources['peak_rss_bytes']/1024**2,
            swap_growth_mib=resources['maximum_swap_growth_bytes']/1024**2,
            ttft_p50_seconds=statistics.median(ttft) if ttft else None,
            ttft_p95_seconds=percentile(ttft, .95),
            completion_p50_seconds=statistics.median(completion),
            completion_p95_seconds=percentile(completion, .95)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in analyze(json.loads(Path(path).read_text()))]
    output = Path(args.output).resolve()
    if output in [Path(path).resolve() for path in args.inputs]:
        parser.error('Output must differ from evidence inputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
