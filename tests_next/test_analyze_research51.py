"""Synthetic analyzer fixtures only; no benchmark, model or engine is executed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('analyze_research51', Path(__file__).resolve().parents[1] / 'benchmarks/analyze_research51.py')
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(folder):
    """Twelve tiny fabricated workers, unmistakably marked as test fixtures."""
    h = '1' * 64
    roots = {variant: '/SYNTHETIC_FIXTURE_ONLY/' + variant for variant in ('baseline', 'candidate')}
    heads = {'baseline': 'a' * 40, 'candidate': 'b' * 40}
    harness = '/SYNTHETIC_FIXTURE_ONLY/benchmark.py'
    manifests = {roots[v]: {'git_head': heads[v], 'source_sha256': {'fixture.py': h}} for v in roots}
    protocol = {harness: h, **{'/SYNTHETIC_FIXTURE_ONLY/' + task + '.json': h for task in ('task_a', 'task_b', 'task_c')}}
    campaign = dict(fixture=True, status='completed', plan=[], runs=[], engine_manifests=manifests,
                    engine_manifests_after=copy.deepcopy(manifests), protocol_sha256=protocol,
                    protocol_unchanged=True, engines_unchanged=True, tasks_by_job={})
    for task in ('task_a', 'task_b', 'task_c'):
        for repeat in (0, 1):
            for variant in ('baseline', 'candidate'):
                candidate = variant == 'candidate'
                ident = f'{task}.r{repeat}.{variant}'
                tasks_path = '/SYNTHETIC_FIXTURE_ONLY/' + task + '.json'
                task_spec = {'id': task, 'prompt': 'SYNTHETIC FIXTURE ONLY. This is not a model benchmark.', 'max_new_tokens': 8}
                controls = ['--packed-direct-io'] if candidate else []
                output = folder / (ident + '.json')
                command = ['SYNTHETIC_PYTHON', harness, '--tasks', tasks_path, '--engine-root', roots[variant],
                           '--output', str(output), '--label', ident, '--expected-protocol-json', json.dumps(protocol),
                           '--expected-engine-head', heads[variant], '--expected-engine-sources-json',
                           json.dumps(manifests[roots[variant]]['source_sha256']), *controls]
                campaign['plan'].append(dict(id=ident, args=controls, tasks=tasks_path, engine_root=roots[variant]))
                campaign['runs'].append(dict(id=ident, command=command, status='completed', exit_code=0,
                                             guard_file=str(output.with_suffix('.guard.json'))))
                campaign['tasks_by_job'][ident] = [task_spec]
                turn = dict(task_spec, status='completed', stop_reason='eos', generated_ids=[10, 99], partial_ids=[10, 99],
                            full_input_ids=[1, 2], appended_input_ids=[1, 2], full_output_ids=[1, 2, 10, 99], eos_token_ids=[99],
                            reused_prefix_tokens=0, native_cache_position=3, finite_logits_checked=True,
                            generation_parameters=dict(temperature=0., do_sample=False, max_new_tokens=8),
                            text='SYNTHETIC FIXTURE ONLY', raw_text='SYNTHETIC FIXTURE ONLY<EOS>',
                            token_seconds=[1., .5] if candidate else [2., 1.],
                            decode_tokens_per_second=.375 if candidate else .25,
                            ttft_seconds=1. if candidate else 2., total_seconds=1.5 if candidate else 3.,
                            wall_seconds=1.6 if candidate else 3.1, logical_read_bytes=300, read_calls=2,
                            process_storage_read_bytes=None, step_checks=[
                                dict(position=2, logits_sha256=h, routes_sha256=h, forward_seconds=.9 if candidate else 1.9,
                                     logical_bytes=100, process_storage_read_bytes=None),
                                dict(position=3, logits_sha256=h, routes_sha256=h, forward_seconds=.4 if candidate else .9,
                                     logical_bytes=200, process_storage_read_bytes=None)],
                            metrics={'allocator': {'peak_bytes': 100}, 'runtime': {'context_payload_peak_bytes': 200},
                                     'cache': {'hits': 1, 'misses': 1, 'peak_bytes': 300, 'load_observations': 1},
                                     'weights': {'loads': 1, 'read_dequant_seconds': .1, 'packed_read_seconds': .2},
                                     'io': {'peak_reserved_bytes': 400, 'read_calls': 4}})
                options = dict(reserve_bytes=3 << 30, zero_copy_weights=candidate, dtype='float32')
                worker = dict(status='completed', label=ident, source_unchanged=True, protocol_unchanged=True,
                              model_stat_unchanged=True, engine_head_unchanged=True,
                              git_head=heads[variant], git_head_after=heads[variant],
                              source_sha256=manifests[roots[variant]]['source_sha256'], protocol_sha256=protocol,
                              harness_sha256=h, tasks_sha256=h, tasks_snapshot=[task_spec], engine_root=roots[variant],
                              command=command[1:] + ['--worker'], options=options, turns=[turn],
                              backend_options=dict(kernel='metal', packed_direct_io=candidate),
                              model=dict(path='/SYNTHETIC_FIXTURE_ONLY/NO_MODEL_EXISTS.gguf', bytes=1000, inode=1, mtime_ns=1),
                              hardware={'fixture': True}, versions={'fixture': True}, cache_regime='SYNTHETIC FIXTURE ONLY',
                              physical_ssd_bandwidth_certified=False, initialization_seconds=.1, tokenizer_seconds=.05,
                              worker_wall_seconds=3. if candidate else 4.)
                snapshot = dict(available_bytes=4 << 30, physical_bytes=16 << 30, swap_used_bytes=0, pressure_level=1)
                guard = dict(command=command + ['--worker'], exit_code=0, violation=None, sample_count=10,
                             limits=dict(rss_bytes=2 << 30, reserve_bytes=3 << 30, swap_growth_bytes=128 << 20,
                                         admitted_payload_bytes=768 << 20, timeout_seconds=600., interval_seconds=.05, max_processes=32),
                             sampled_peak_rss_bytes=64 << 20, minimum_available_bytes=4 << 30,
                             maximum_swap_growth_bytes=0, peak_pressure_level=1, before=snapshot, after=snapshot,
                             elapsed_seconds=4. if candidate else 5.)
                save(output, worker)
                save(output.with_suffix('.guard.json'), guard)
    save(folder / 'campaign.json', campaign)
    return campaign


class Research51AnalyzerTests(unittest.TestCase):
    def test_primary_streaming_and_forward_rates_remain_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            summary, rows, _ = ANALYZER.analyze_campaign(folder)
        self.assertEqual(summary['pair_count'], 6)
        self.assertEqual(summary['paired_decode_speedup']['median'], 1.5)
        self.assertEqual(summary['paired_forward_only_decode_speedup']['median'], 2.)
        self.assertEqual(summary['aggregate']['baseline']['decode_tok_s'], .25)
        self.assertEqual(summary['aggregate']['baseline']['forward_only_decode_tps'], 1.)
        self.assertEqual(summary['aggregate']['candidate']['end_to_end_tok_s'], .5)
        self.assertEqual(summary['aggregate']['baseline']['generated_tokens_including_eos'], 12)
        self.assertIsNone(summary['aggregate']['baseline']['process_storage_read_bytes'])
        self.assertIsNone(summary['aggregate']['baseline']['decode_process_storage_read_bytes_per_token'])
        self.assertTrue(all(row['expert_load_seconds_separately_measured'] is None for row in rows))

    def test_failure_controls_refuse_unsafe_or_changed_evidence(self):
        cases = [('campaign.json', 'status', 'started'),
                 ('task_a.r0.baseline.guard.json', 'violation', 'system_swap_growth'),
                 ('task_a.r0.baseline.guard.json', 'exit_code', 1),
                 ('task_a.r0.baseline.json', 'source_unchanged', False),
                 ('task_a.r0.baseline.json', 'protocol_unchanged', False)]
        for name, key, value in cases:
            with self.subTest(name=name, key=key), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                fixture(folder)
                path = folder / name
                document = json.loads(path.read_text())
                document[key] = value
                save(path, document)
                with self.assertRaises(ValueError):
                    ANALYZER.analyze_campaign(folder)

    def test_logit_or_route_drift_and_missing_steps_fail(self):
        for key in ('logits_sha256', 'routes_sha256', 'position', 'missing_step'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                fixture(folder)
                path = folder / 'task_a.r0.candidate.json'
                document = json.loads(path.read_text())
                steps = document['turns'][0]['step_checks']
                if key == 'missing_step':
                    steps.pop()
                else:
                    steps[0][key] = '2' * 64 if key.endswith('sha256') else 999
                save(path, document)
                with self.assertRaises(ValueError):
                    ANALYZER.analyze_campaign(folder)

    def test_incomplete_pair_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            campaign = fixture(folder)
            campaign['runs'].pop()
            save(folder / 'campaign.json', campaign)
            with self.assertRaises(ValueError):
                ANALYZER.analyze_campaign(folder)

    def test_outputs_are_complete_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            summary, rows, raw = ANALYZER.analyze_campaign(folder)
            output = folder / 'analysis'
            ANALYZER.write_outputs(output, summary, rows, raw)
            self.assertTrue(all((output / name).is_file() for name in ('summary.json', 'benchmarks.csv', 'responses.md')))
            text = (output / 'responses.md').read_text()
            self.assertIn('FIXTURES SYNTHÉTIQUES', text)
            self.assertEqual(text.count('### Sortie brute intégrale'), 12)
            with self.assertRaises(ValueError):
                ANALYZER.write_outputs(output, summary, rows, raw)

    def test_linear_iqr_definition(self):
        result = ANALYZER.describe([1., 2., 3., 4.])
        self.assertEqual((result['median'], result['q1'], result['q3'], result['iqr']), (2.5, 1.75, 3.25, 1.5))


if __name__ == '__main__':
    unittest.main()
