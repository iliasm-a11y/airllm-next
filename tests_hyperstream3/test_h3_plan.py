"""Stdlib-only plan validation: no engine, guard or inference is executed."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import plan_hyperstream3 as planner
from campaign_research51 import validate_job_args


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.roots = []
        for name in ('h1', 'h2', 'candidate'):
            root = self.folder / name
            required = list(planner.CORE_FILES)
            if name != 'h1':
                required += ['air_llm/airllm/nextgen/hyperprefill.py']
            if name == 'h2':
                required += ['research/hyperstream2/context.json']
            if name == 'candidate':
                required += ['air_llm/airllm/nextgen/gguf_m5.py',
                             'air_llm/airllm/nextgen/gguf_speculative.py',
                             'air_llm/airllm/nextgen/gguf_packed_io.py', planner.LOOKUP_TASK]
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith('.json'):
                    path.write_text(json.dumps([dict(id='synthetic', prompt='Synthetic copy workload',
                                                    max_new_tokens=128 if relative == planner.LOOKUP_TASK else 32)]))
                else:
                    path.write_text('# synthetic planner fixture; never imported\n')
            self.roots.append(root)
        self.output = self.folder / 'plans'

    def test_counterbalanced_ids_and_engine_roots(self):
        jobs = planner.paired_jobs(self.roots[0], self.roots[2], pairs=3)
        self.assertEqual([job['id'] for job in jobs],
            ['baseline-p00', 'candidate-p00', 'candidate-p01', 'baseline-p01',
             'baseline-p02', 'candidate-p02'])
        for job in jobs:
            expected = self.roots[0] if job['id'].startswith('baseline') else self.roots[2]
            self.assertEqual(job['engine_root'], str(expected.resolve()))
            self.assertEqual(set(job), {'id', 'engine_root', 'args'})
            self.assertEqual(job['args'], list(planner.COMMON))

    def test_every_ablation_is_separate_and_campaign_compatible(self):
        experiments = planner.build_experiments(*self.roots, pairs=3)
        self.assertEqual(len(experiments), 16)
        for experiment in experiments:
            self.assertEqual(len(experiment['plan']), 6)
            self.assertEqual(len({job['id'] for job in experiment['plan']}), 6)
            for job in experiment['plan']:
                self.assertEqual(validate_job_args(job['args']), job['args'])
                for flag in planner.COMMON:
                    self.assertIn(flag, job['args'])

    def test_fixed_prefill_trials_have_no_adaptive_policy(self):
        experiments = {trial['name']: trial for trial in planner.build_experiments(*self.roots)}
        for chunk in (16, 32, 64, 128):
            trial = experiments[f'prefill{chunk}_vs16']
            for job in trial['plan']:
                self.assertNotIn('--hyper-prefill', job['args'])
                self.assertEqual(job['engine_root'], str(self.roots[1].resolve()))
                value = job['args'][job['args'].index('--prefill-chunk') + 1]
                self.assertEqual(value, '16' if job['id'].startswith('baseline') else str(chunk))
        adaptive = experiments['h1_vs_h2']['plan']
        self.assertNotIn('--hyper-prefill', adaptive[0]['args'])
        self.assertIn('--hyper-prefill', adaptive[1]['args'])

    def test_isolated_new_feature_flags_do_not_leak_to_baselines(self):
        experiments = planner.build_experiments(*self.roots)
        feature_flags = {'--metal-batch-backend', '--metal-token-tile',
                         '--grouped-direct-io', '--speculative-depth'}
        for experiment in experiments:
            for job in experiment['plan']:
                if job['id'].startswith('baseline'):
                    self.assertFalse(feature_flags & set(job['args']))
        indexed = {experiment['name']: experiment for experiment in experiments}
        for tile in (2, 4, 8, 16):
            candidate = indexed[f'reuse_tile{tile}_vs_h2']['plan'][1]
            self.assertEqual(candidate['args'][-4:],
                             ['--metal-batch-backend', 'reuse', '--metal-token-tile', str(tile)])
        for depth in (2, 4, 8, 16):
            candidate = indexed[f'speculation_depth{depth}_vs_h2']['plan'][1]
            self.assertEqual(candidate['args'][-2:], ['--speculative-depth', str(depth)])

    def test_combination_is_last_unvalidated_and_has_all_prior_requirements(self):
        experiments = planner.build_experiments(*self.roots)
        combo = experiments[-1]
        self.assertEqual(combo['name'], 'combined_tensorops_grouped_spec8')
        self.assertEqual(combo['status'], 'unvalidated_requires_prior_pass')
        self.assertTrue(combo['speculative'])
        self.assertEqual(set(combo['requires_prior_pass']),
                         {'h1_vs_h2', 'tensorops_vs_h2', 'grouped_transport_vs_h2',
                          'speculation_depth8_vs_h2'})
        self.assertTrue(set(combo['requires_prior_pass']) <= {trial['name'] for trial in experiments[:-1]})
        self.assertIn('Do not execute', combo['notes'])

    def test_speculative_pairs_share_the_same_frozen_lookup_task(self):
        expected = str((self.roots[2] / planner.LOOKUP_TASK).resolve())
        experiments = planner.build_experiments(*self.roots)
        for trial in experiments:
            self.assertEqual(trial['tasks_kind'],
                             'lookup_synthetic_copy' if trial['speculative'] else 'historical_364')
            for job in trial['plan']:
                if trial['speculative']:
                    self.assertEqual(job['tasks'], expected)
                else:
                    self.assertNotIn('tasks', job)
        actual = json.loads((ROOT / planner.LOOKUP_TASK).read_text())
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['max_new_tokens'], 128)
        self.assertLess(len(actual[0]['prompt'].encode('utf-8')), 900)
        self.assertEqual(actual[0]['prompt'].count('The small blue boat waits near the old stone bridge.'), 8)

    def test_missing_or_invalid_lookup_workload_refuses_before_output(self):
        path = self.roots[2] / planner.LOOKUP_TASK
        valid = dict(id='one', prompt='copy', max_new_tokens=128)
        cases = [[], [valid, valid], [dict(valid, max_new_tokens=127)],
                 [dict(valid, max_new_tokens=True)], [dict(valid, prompt='')],
                 [dict(valid, prompt='x' * 901)]]
        for value in cases:
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    planner.write_plans(*self.roots, self.output)
                self.assertFalse(self.output.exists())
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            planner.write_plans(*self.roots, self.output)
        self.assertFalse(self.output.exists())

    def test_output_is_exclusive_and_hashes_inspected_sources_without_launch(self):
        with patch('subprocess.Popen', side_effect=AssertionError('No process may launch')):
            index = planner.write_plans(*self.roots, self.output, pairs=2)
        self.assertEqual(index['status'], 'plans_only_no_inference_launched')
        self.assertEqual(len(list(self.output.glob('*.json'))), 17)
        self.assertTrue((self.output / 'SCHEMA.md').is_file())
        for experiment in index['experiments']:
            plan = json.loads((self.output / experiment['plan_file']).read_text())
            self.assertIsInstance(plan, list)
            self.assertEqual(len(plan), 4)
            self.assertNotIn('plan', experiment)
        original = (self.output / 'index.json').read_bytes()
        with self.assertRaises(FileExistsError):
            planner.write_plans(*self.roots, self.output)
        self.assertEqual((self.output / 'index.json').read_bytes(), original)
        for value in index['inspected_source_sha256']['candidate'].values():
            self.assertEqual(len(value), 64)
        self.assertEqual(index['tasks_recommendation']['lookup_synthetic_copy'],
                         str((self.roots[2] / planner.LOOKUP_TASK).resolve()))
        self.assertEqual(index['tasks_recommendation']['historical_364'],
                         str((self.roots[1] / 'research/hyperstream2/context.json').resolve()))
        self.assertEqual(set(index['task_sha256']), {'historical_364', 'lookup_synthetic_copy'})

    def test_broken_output_symlink_cannot_create_its_target(self):
        target = self.folder / 'absent-target'
        self.output.symlink_to(target)
        with self.assertRaises(FileExistsError):
            planner.write_plans(*self.roots, self.output)
        self.assertFalse(target.exists())
        self.assertTrue(self.output.is_symlink())

    def test_invalid_roots_and_missing_files_refuse_before_output_creation(self):
        missing = self.roots[2] / 'air_llm/airllm/nextgen/gguf_m5.py'
        missing.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            planner.write_plans(*self.roots, self.output)
        self.assertFalse(self.output.exists())
        with self.assertRaises(FileNotFoundError):
            planner.validate_root(self.folder / 'absent')
        with self.assertRaises(ValueError):
            planner.validate_root(self.roots[0] / planner.CORE_FILES[0])

    def test_invalid_pairs_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                planner.build_experiments(*self.roots, pairs=value)

    def test_cli_runs_without_site_packages_and_only_writes_plans(self):
        command = [sys.executable, '-S', str(ROOT / 'benchmarks/plan_hyperstream3.py'),
                   '--h1-root', str(self.roots[0]), '--h2-root', str(self.roots[1]),
                   '--candidate', str(self.roots[2]), '--output-dir', str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'plans_only_no_inference_launched')
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('File exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
