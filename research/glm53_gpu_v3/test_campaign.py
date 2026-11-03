"""V3 gating/planning fixtures only; no server, native runtime or GPU executes."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import sys
import unittest

from research.glm53_gpu_v3.test_power_state import valid_evidence

from research.glm53_gpu_v3.campaign import check_before_launch, classify_stage, make_plan, validate_timed_report


class V3CampaignTests(unittest.TestCase):
    def setUp(self):
        power = patch('research.glm53_gpu_v3.campaign.power_preflight', return_value={
            'passed': True, 'reasons': [], 'snapshot': {'available': True, 'lid_closed': False, 'power_source': 'battery'}})
        power.start(); self.addCleanup(power.stop)
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.gguf'; self.model.write_bytes(b'fixture weights')
        self.binary = self.root / 'server'; self.binary.write_bytes(b'fixture executable')
        self.library = self.root / 'libggml-fixture.dylib'; self.library.write_bytes(b'fixture dylib')
        metrics = {'passed': True, 'finite': True, 'bit_mismatch_count': 0, 'compared': 16, 'elements': 32}
        self.reference = {'fidelity_passed': True, 'guard_status': 'completed', 'model_file_stats_unchanged': True,
            'full_logits': metrics, 'coverage': {k: metrics for k in ('routes', 'router', 'kda')},
            'environment': {'AIRLLM_GPU_CACHE_MIB': '0'},
            'command': ['probe', str(self.model), 'capture', '10', '0', '16', 'La capitale de la France est', '-', 'cpu'],
            'native_libraries_sha256': {self.library.name: hashlib.sha256(self.library.read_bytes()).hexdigest()},
            'model_files': [{'file': self.model.name, 'size': self.model.stat().st_size, 'mtime_ns': self.model.stat().st_mtime_ns}]}
        self.candidate = deepcopy(self.reference)
        self.candidate['environment'].update(AIRLLM_MMAP_NOCACHE='1', AIRLLM_DENSE_LOOKAHEAD='1', AIRLLM_DENSE_LOOKAHEAD_MIB='32')
        self.candidate['observed_state_writes'] = {'passed': True, 'compared_events': 20}
        for arm, report in (('A', self.reference), ('B', self.candidate)):
            capture = self.root / (arm + '-capture')
            capture.mkdir()
            report['command'][2] = str(capture)
            (capture / 'tokens.jsonl').write_text(''.join(json.dumps({'step': step, 'token': 20 + step,
                'next_input_token': 20 + step}) + '\n' for step in range(16)))
            (capture / 'complete.json').write_text(json.dumps({'completed': True, 'captured_steps': 16}))
        self.a_path, self.b_path = self.root / 'A.json', self.root / 'B.json'
        self.write_reports()
        self.args = SimpleNamespace(output=self.root / 'campaign', binary=self.binary, model=self.model,
            reference_fidelity=self.a_path, candidate_fidelity=self.b_path, candidate='cpu-stream', tail_layers=None,
            tokens=16, cycles=1, prompt='La capitale de la France est')

    def tearDown(self):
        self.temp.cleanup()

    def write_reports(self):
        self.a_path.write_text(json.dumps(self.reference)); self.b_path.write_text(json.dumps(self.candidate))

    def completed_request(self, tokens, stop_type='limit'):
        return {'tokens': list(tokens), 'body': {'n_predict': self.args.tokens},
                'final': {'stop': True, 'stop_type': stop_type, 'tokens_predicted': len(tokens),
                          'timings': {'predicted_n': len(tokens)}, 'truncated': False}}

    def test_cpu_stream_abba_has_original_A_and_only_candidate_new_stream_flags(self):
        for tokens in (8, 16, 32, 64):
            self.args.tokens = tokens
            plan = make_plan(self.args)
            self.assertEqual(plan['sequence'], 'ABBA')
            self.assertIn('research/glm53_gpu_v3/power_state.py', plan['source_fingerprints'])
            for job in plan['jobs']:
                command = job['command']
                self.assertEqual(command[0], sys.executable)
                self.assertIn('research.glm53_gpu_v3.benchmark_native', command)
                self.assertEqual(command[command.index('--expected-power-source')+1], 'battery')
                self.assertEqual('--mmap-nocache' in command, job['arm'] == 'B')
                self.assertEqual('--dense-lookahead' in command, job['arm'] == 'B')
                self.assertFalse(job['gpu_commands_required'])
                self.assertNotIn('--metal-timing', command)
                for name, value in (('--threads', '10'), ('--batch', '8'), ('--context', '512'),
                                    ('--max-rss-gib', '8.5'), ('--max-swap-mib', '32'), ('--timeout', '900')):
                    self.assertEqual(command[command.index(name) + 1], value)
            self.assertEqual(plan['fidelity']['A']['captured_steps'], 16)
            self.assertEqual(plan['fidelity']['B']['captured_steps'], 16)

    def test_reference_candidate_steps_states_or_stream_mismatch_are_rejected(self):
        for kind in ('reference-short', 'state', 'lookahead', 'reference-newflag'):
            a, b = deepcopy(self.reference), deepcopy(self.candidate)
            if kind == 'reference-short': a['full_logits']['compared'] = 4
            elif kind == 'state': b['observed_state_writes']['passed'] = False
            elif kind == 'lookahead': b['environment']['AIRLLM_DENSE_LOOKAHEAD_MIB'] = '64'
            else: a['environment']['AIRLLM_MMAP_NOCACHE'] = '1'
            self.a_path.write_text(json.dumps(a)); self.b_path.write_text(json.dumps(b))
            with self.assertRaises(ValueError): make_plan(self.args)

    def test_head_and_tail_require_matching_gpu_modes_and_observations(self):
        self.args.candidate = 'head-stream'
        self.candidate['command'][-1] = 'head-only'
        self.candidate['environment']['AIRLLM_METAL_Q8_CPU_ARITH'] = '1'; self.write_reports()
        head = make_plan(self.args)
        self.assertIn('--head-only', head['jobs'][1]['command'])
        self.assertTrue(head['jobs'][1]['gpu_commands_required'])
        self.args.candidate, self.args.tail_layers = 'tail-stream', 12
        self.candidate['command'][-1] = 'quant-tail'
        self.candidate['environment'].update(AIRLLM_METAL_Q8_CPU_ARITH='2', AIRLLM_GLM_Q8_ONLY='1',
            AIRLLM_METAL_CPU_MAPPED_WEIGHTS='1', AIRLLM_Q8_TAIL_PATTERN='^fixture$'); self.write_reports()
        with patch('research.glm53_gpu_v2.placement.plan', return_value={'tensor_pattern': '^fixture$'}):
            tail = make_plan(self.args)
        self.assertIn('--mapped-cpu-weights', tail['jobs'][1]['command'])
        raw = {'status': 'complete', 'exit_code': 0, 'guard_violation': None, 'model_file_stats_unchanged': True,
               'gpu_command_times': {'valid_commands': 0}}
        self.assertEqual(classify_stage(tail['jobs'][1], 0, raw)['status'], 'failed')
        raw['gpu_command_times']['valid_commands'] = 1
        self.assertEqual(classify_stage(tail['jobs'][1], 0, raw)['status'], 'complete')

    def test_cpu_stream_does_not_need_gpu_timers_and_guard_failure_stops(self):
        job = make_plan(self.args)['jobs'][1]
        raw = {'status': 'complete', 'exit_code': 0, 'guard_violation': None, 'model_file_stats_unchanged': True}
        self.assertEqual(classify_stage(job, 0, raw)['status'], 'complete')
        raw['guard_violation'] = {'swap': 100}
        self.assertEqual(classify_stage(job, 0, raw)['status'], 'failed')

    def test_prelaunch_rejects_runtime_model_or_fidelity_drift(self):
        plan = make_plan(self.args)
        self.assertTrue(check_before_launch(self.args, plan)['passed'])
        self.library.write_bytes(b'changed dylib')
        with self.assertRaises(ValueError): check_before_launch(self.args, plan)
        self.library.write_bytes(b'fixture dylib')
        self.model.write_bytes(b'changed weights')
        with self.assertRaises(ValueError): check_before_launch(self.args, plan)


    def test_timed_ids_and_raw_hashes_are_bound_to_the_passed_fidelity(self):
        self.args.tokens = 8
        plan = make_plan(self.args)
        job = plan['jobs'][1]
        raw = {**valid_evidence(), 'exit_code': 0, 'binary_sha256': plan['binary_fingerprint']['sha256'],
            'native_libraries_sha256': {name: row['sha256'] for name, row in plan['native_library_fingerprints'].items()},
            'requests': [self.completed_request(list(range(20, 28)))]}
        self.assertTrue(validate_timed_report(job, raw, plan)['passed'])
        raw['requests'][0]['tokens'][0] = 999
        self.assertFalse(validate_timed_report(job, raw, plan)['fidelity_prefix_matches'])
        raw['requests'][0]['tokens'] = list(range(20, 28))
        raw['binary_sha256'] = 'different'
        self.assertFalse(validate_timed_report(job, raw, plan)['passed'])
        raw['binary_sha256'] = plan['binary_fingerprint']['sha256']
        raw['native_libraries_sha256'] = {}
        self.assertFalse(validate_timed_report(job, raw, plan)['passed'])

    def test_fidelity_token_capture_drift_is_rejected_before_a_later_stage(self):
        plan = make_plan(self.args)
        capture = Path(self.candidate['command'][2])
        (capture / 'tokens.jsonl').write_text('changed')
        with self.assertRaises(ValueError): check_before_launch(self.args, plan)

    def test_gpu_evidence_must_cover_the_request_and_long_prefix_scope_stays16(self):
        self.args.tokens = 32
        plan = make_plan(self.args)
        job = dict(plan['jobs'][1], gpu_commands_required=True)
        raw = {**valid_evidence(), 'exit_code': 0, 'binary_sha256': plan['binary_fingerprint']['sha256'],
            'native_libraries_sha256': {name: row['sha256'] for name, row in plan['native_library_fingerprints'].items()},
            'gpu_command_times': {'valid_commands': 10},
            'requests': [dict(self.completed_request(list(range(20, 36)) + [999] * 16),
                              gpu_command_times={'valid_commands': 0})]}
        self.assertFalse(validate_timed_report(job, raw, plan)['passed'])
        raw['requests'][0]['gpu_command_times']['valid_commands'] = 1
        result = validate_timed_report(job, raw, plan)
        self.assertTrue(result['passed'])
        self.assertEqual(result['checked_prefix_steps'], 16)
        self.assertEqual(result['generated_ids_logged'], 32)

    def test_actual_eos_capture27_extends_prefix_but_requested64_does_not(self):
        self.args.tokens = 64
        for report in (self.reference, self.candidate):
            report['full_logits']['compared'] = 27
            capture = Path(report['command'][2])
            (capture/'tokens.jsonl').write_text(''.join(json.dumps({'step': step, 'token': 20+step,
                'next_input_token': 20+step})+'\n' for step in range(27)))
            (capture/'complete.json').write_text(json.dumps({'completed': True, 'captured_steps': 27}))
        self.reference['command'][5] = '64'
        self.candidate['command'][5] = '27'
        self.write_reports()
        plan = make_plan(self.args)
        raw = {**valid_evidence(), 'exit_code': 0, 'binary_sha256': plan['binary_fingerprint']['sha256'],
            'native_libraries_sha256': {name: row['sha256'] for name, row in plan['native_library_fingerprints'].items()},
            'requests': [self.completed_request(list(range(20, 47)), 'eos')]}
        result = validate_timed_report(plan['jobs'][1], raw, plan)
        self.assertTrue(result['passed'])
        self.assertEqual(result['checked_prefix_steps'], 27)
        self.assertEqual(result['fidelity_prefix_scope_steps'], 27)
        raw['requests'][0]['tokens'][25] = 999
        self.assertFalse(validate_timed_report(plan['jobs'][1], raw, plan)['passed'])

    def test_two_request_reset_capture_is_not_a_continuous_trajectory(self):
        self.candidate['environment']['AIRLLM_DIAGNOSTIC_REQUESTS'] = '2'
        self.write_reports()
        with self.assertRaisesRegex(ValueError, 'two-request'):
            make_plan(self.args)

    def test_unmatched_lock_environment_cannot_enter_either_arm(self):
        self.reference['environment']['AIRLLM_DENSE_LOCK'] = '1'
        self.write_reports()
        with self.assertRaisesRegex(ValueError, 'reference must have no dense lock'):
            make_plan(self.args)
        del self.reference['environment']['AIRLLM_DENSE_LOCK']
        self.candidate['environment']['AIRLLM_DENSE_LOCK'] = '1'
        self.write_reports()
        with self.assertRaisesRegex(ValueError, 'environment differs'):
            make_plan(self.args)

    def test_exact_lock_is_candidate_only_and_timed_retention_is_required(self):
        self.args.tokens = 8
        from research.glm53_gpu_v3.dense_lock_plan import lock_environment, retained_log
        dense = {'path': str(self.root/'selector.json'), 'sha256': 'd'*64, 'budget_mib': 128,
            'page_size': 16384, 'names': ['blk.3.ssm_f_a.weight'],
            'selected_raw_bytes': 32, 'selected_page_union_bytes': 16384}
        native_log = self.root/'native.log'
        native_log.write_text('AIRLLM_DENSE_LOCK status=retained retained=1 names=1 raw_bytes=32 '
            'page_union_bytes=16384 locked_bytes=16384 lock_calls=1 unlock_calls=0 '
            'verified_pages_at_lock=1 error=0 unlock_error=0\n')
        evidence = retained_log(native_log, dense)
        self.args.dense_lock_plan = Path(dense['path'])
        self.candidate['environment'].update(lock_environment(dense))
        self.candidate.update(dense_lock_plan=dense, dense_lock=evidence, dense_lock_accepted=True,
            dense_lock_prelaunch={'passed': True, 'sha256': dense['sha256']},
            dense_lock_postrun={'passed': True, 'sha256': dense['sha256']})
        self.write_reports()
        with patch('research.glm53_gpu_v3.campaign.load_dense_lock_plan', return_value=dense):
            plan = make_plan(self.args)
            self.assertNotIn('--dense-lock-plan', plan['jobs'][0]['command'])
            self.assertIn('--dense-lock-plan', plan['jobs'][1]['command'])
            raw = {**valid_evidence(), 'exit_code': 0, 'binary_sha256': plan['binary_fingerprint']['sha256'],
                'native_libraries_sha256': {n:r['sha256'] for n,r in plan['native_library_fingerprints'].items()},
                'research_environment': lock_environment(dense), 'dense_lock_plan': dense,
                'dense_lock': evidence, 'dense_lock_postrun': {'passed': True, 'sha256': dense['sha256']},
                'requests': [self.completed_request(list(range(20, 28)))]}
            self.assertTrue(validate_timed_report(plan['jobs'][1], raw, plan)['passed'])
            raw['dense_lock'] = dict(evidence, passed=False)
            self.assertFalse(validate_timed_report(plan['jobs'][1], raw, plan)['passed'])
            self.candidate['environment']['AIRLLM_DENSE_LOCK_MIB'] = '256'
            self.write_reports()
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                make_plan(self.args)

    def test_short_limit_and_nonzero_native_exit_cannot_qualify(self):
        plan = make_plan(self.args)
        raw = {**valid_evidence(), 'exit_code': 0, 'binary_sha256': plan['binary_fingerprint']['sha256'],
            'native_libraries_sha256': {n:r['sha256'] for n,r in plan['native_library_fingerprints'].items()},
            'requests': [self.completed_request(list(range(20, 35)))]}
        result = validate_timed_report(plan['jobs'][1], raw, plan)
        self.assertTrue(result['fidelity_prefix_matches'])
        self.assertFalse(result['completion_contract']['passed'])
        self.assertFalse(result['passed'])
        raw['requests'] = [self.completed_request(list(range(20, 36)))]
        self.assertTrue(validate_timed_report(plan['jobs'][1], raw, plan)['passed'])
        raw['exit_code'] = -9
        self.assertFalse(validate_timed_report(plan['jobs'][1], raw, plan)['passed'])


if __name__ == '__main__':
    unittest.main()
