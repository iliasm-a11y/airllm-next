"""Deliberately fabricated evidence; no inference or measured model gains."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

from test_analyze_research51 import fixture, save

SPEC = importlib.util.spec_from_file_location('verify_hyperstream2',
    Path(__file__).resolve().parents[1] / 'benchmarks/verify_hyperstream2.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def synthetic(folder):
    campaign = fixture(folder)
    retained = [job for job in campaign['plan'] if job['id'].startswith('task_a.')]
    # AB then BA, using the actual HyperStream 2 label convention.
    retained = [retained[0], retained[1], retained[3], retained[2]]
    old_runs = {run['id']: run for run in campaign['runs']}
    campaign['plan'], campaign['runs'], tasks_by_job = [], [], {}
    for job in retained:
        old_label = job['id']
        variant = old_label.rsplit('.', 1)[1]
        repeat = int(old_label.split('.r')[1].split('.')[0])
        label = f'{variant}-p{repeat:02d}'
        candidate = variant == 'candidate'
        worker = json.loads((folder / (old_label + '.json')).read_text())
        guard = json.loads((folder / (old_label + '.guard.json')).read_text())
        run = old_runs[old_label]
        old_controls = job['args']
        command = run['command'][:-len(old_controls)] if old_controls else run['command'][:]
        for flag, value in (('--label', label), ('--output', str(folder / (label + '.json')))):
            command[command.index(flag) + 1] = value
        command += ['--model', worker['model']['path']]
        controls = ['--hyper-prefill', '--hyper-prefill-max-tokens', '128'] if candidate else []
        command += controls
        job.update(id=label, args=controls)
        run.update(id=label, command=command, guard_file=str(folder / (label + '.guard.json')))
        worker.update(label=label, command=command[1:] + ['--worker'])
        worker['model']['bytes'] = 24 << 30
        worker['model_config'] = dict(num_hidden_layers=1, num_experts=8, num_experts_per_tok=2)
        worker['options'].update(zero_copy_weights=True, device='gpu', prefetch=False, prefill_chunk=16)
        worker['backend_options'] = dict(kernel='metal', packed_direct_io=True)
        if candidate:
            worker['backend_options'].update(hyper_prefill=True, hyper_prefill_max_tokens=128)
        turn = worker['turns'][0]
        turn['decode_tokens_per_second'] = 2. if candidate else 1.
        routing = {'0': {'0': [1, 2], '1': [3, 4], '2': [1, 4]}}
        turn['routing_by_layer_position'] = routing
        turn['routing_sha256'] = hashlib.sha256(json.dumps(routing, sort_keys=True, separators=(',', ':'),
                                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        for index, step in enumerate(turn['step_checks']):
            values = [1. + (1e-5 if candidate else 0.), 2., 3.]
            raw = struct.pack('<3f', *values)
            artifact = f'{label}.step-{index:03d}.f32'
            (folder / artifact).write_bytes(raw)
            step['logits_sha256'] = hashlib.sha256(raw).hexdigest()
            step['logits_artifact'] = dict(path=artifact, sha256=step['logits_sha256'], dtype='float32', shape=[1, 1, 3])
        guard.update(command=command + ['--worker'], sampled_peak_processes=1)
        save(folder / (label + '.json'), worker)
        save(folder / (label + '.guard.json'), guard)
        campaign['plan'].append(job)
        campaign['runs'].append(run)
        tasks_by_job[label] = campaign['tasks_by_job'][old_label]
    campaign['tasks_by_job'] = tasks_by_job
    save(folder / 'campaign.json', campaign)
    attestation = dict(minimum_pairs=2,
        model_integrity=dict(before_sha256='1' * 64, after_sha256='1' * 64, bytes=24 << 30, stable_file=True),
        comparison=dict(options={}, backend_options=dict(hyper_prefill=[None, True], hyper_prefill_max_tokens=[None, 128]),
                        numerical=dict(mode='tolerance', rtol=1e-4, atol=1e-4)))
    save(folder / 'campaign-integrity.json', attestation)
    return campaign


class IndependentHyperstream2Tests(unittest.TestCase):
    def test_reads_raw_vectors_and_recalculates_pair_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            synthetic(folder)
            result = CHECKER.verify(folder)
            self.assertEqual((result['worker_count'], result['pair_count']), (4, 2))
            self.assertTrue(result['fixture'])
            self.assertFalse(result['all_logits_bitwise_exact'])
            self.assertEqual(result['paired_ttft_speedup']['median'], 2.)
            self.assertEqual(result['aggregate']['baseline']['decode_tok_s'], 1.)
            self.assertEqual(result['aggregate']['candidate']['decode_tok_s'], 2.)
            self.assertEqual(result['pairs'][0]['logits_elements_checked'], 6)
            self.assertGreater(result['pairs'][0]['logits_max_absolute_error'], 0.)

    def test_missing_pair_and_incomplete_worker_are_rejected(self):
        for mode in ('missing', 'incomplete', 'ordering'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                campaign = synthetic(folder)
                if mode == 'missing':
                    campaign['plan'].pop()
                    campaign['runs'].pop()
                elif mode == 'incomplete':
                    campaign['runs'][0]['status'] = 'stopped'
                else:
                    campaign['plan'][2:] = reversed(campaign['plan'][2:])
                    campaign['runs'][2:] = reversed(campaign['runs'][2:])
                save(folder / 'campaign.json', campaign)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_changed_tokens_routes_sources_guard_and_model_are_rejected(self):
        for mode in ('tokens', 'routes', 'source', 'guard', 'checkpoint', 'controls', 'position'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                name = ('candidate-p00.guard.json' if mode == 'guard' else
                        'campaign-integrity.json' if mode == 'checkpoint' else 'candidate-p00.json')
                path = folder / name
                document = json.loads(path.read_text())
                if mode == 'tokens':
                    document['turns'][0]['generated_ids'][0] = 42
                elif mode == 'routes':
                    document['turns'][0]['routing_by_layer_position']['0']['0'] = [5, 6]
                elif mode == 'source':
                    document['source_sha256']['fixture.py'] = '2' * 64
                elif mode == 'guard':
                    document['limits']['swap_growth_bytes'] *= 2
                elif mode == 'checkpoint':
                    document['model_integrity']['after_sha256'] = '2' * 64
                elif mode == 'controls':
                    document['options']['dtype'] = 'float16'
                else:
                    document['turns'][0]['step_checks'][0]['position'] += 1
                save(path, document)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_self_consistent_but_out_of_tolerance_vector_is_rejected(self):
        for value in (2., float('nan'), float('inf')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                path = folder / 'candidate-p00.json'
                worker = json.loads(path.read_text())
                raw = struct.pack('<3f', value, 2., 3.)
                artifact = worker['turns'][0]['step_checks'][0]['logits_artifact']
                (folder / artifact['path']).write_bytes(raw)
                artifact['sha256'] = hashlib.sha256(raw).hexdigest()
                worker['turns'][0]['step_checks'][0]['logits_sha256'] = artifact['sha256']
                save(path, worker)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_self_consistent_routes_still_require_all_layers_positions_and_topk(self):
        for mode in ('layer', 'position', 'topk', 'range', 'duplicate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                path = folder / 'candidate-p00.json'
                worker = json.loads(path.read_text())
                turn = worker['turns'][0]
                routing = turn['routing_by_layer_position']
                if mode == 'layer':
                    routing['1'] = routing['0'].copy()
                elif mode == 'position':
                    del routing['0']['0']
                elif mode == 'topk':
                    routing['0']['0'] = [1]
                elif mode == 'range':
                    routing['0']['0'] = [1, 8]
                else:
                    routing['0']['0'] = [1, 1]
                turn['routing_sha256'] = hashlib.sha256(json.dumps(routing, sort_keys=True,
                    separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
                save(path, worker)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_missing_raw_output_or_process_samples_are_rejected(self):
        for field in ('text', 'raw_text', 'sampled_peak_rss_bytes', 'sampled_peak_processes'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                guard_field = field.startswith('sampled_')
                path = folder / ('candidate-p00.guard.json' if guard_field else 'candidate-p00.json')
                document = json.loads(path.read_text())
                if guard_field:
                    document[field] = 0
                else:
                    document['turns'][0][field] = None
                save(path, document)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_raw_artifact_hash_and_path_traversal_are_rejected(self):
        for mode in ('hash', 'path', 'truncated'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                path = folder / 'candidate-p00.json'
                worker = json.loads(path.read_text())
                artifact = worker['turns'][0]['step_checks'][0]['logits_artifact']
                if mode == 'path':
                    artifact['path'] = '../outside.f32'
                else:
                    (folder / artifact['path']).write_bytes(b'\0' * (4 if mode == 'truncated' else 12))
                save(path, worker)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)

    def test_loosened_tolerance_and_exact_claim_reject_different_logits(self):
        for mode in ('loose', 'exact', 'nan'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                synthetic(folder)
                path = folder / 'campaign-integrity.json'
                integrity = json.loads(path.read_text())
                numerical = integrity['comparison']['numerical']
                if mode == 'loose':
                    numerical['atol'] = .1
                elif mode == 'exact':
                    numerical.update(mode='exact', rtol=0., atol=0.)
                else:
                    numerical['atol'] = float('nan')
                save(path, integrity)
                with self.assertRaises(ValueError):
                    CHECKER.verify(folder)


if __name__ == '__main__':
    unittest.main()
