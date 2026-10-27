"""Fabricated evidence only: exercise the independent recording checker."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location('verify_gguf_speculation',
    Path(__file__).resolve().parents[1]/'benchmarks/verify_gguf_speculation.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def synthetic(folder):
    model = dict(path='/fixture/model.gguf', bytes=24 << 30, inode=123, mtime_ns=456)
    base = dict(status='completed', model=model, git_head='a'*40, git_head_after='a'*40,
        source_sha256={'fixture.py': 'b'*64}, source_unchanged=True,
        protocol_unchanged=True, model_stat_unchanged=True, engine_head_unchanged=True,
        model_config=dict(vocab_size=8, num_hidden_layers=2, num_experts=4, num_experts_per_tok=2))
    def step(label, index, start, supplied, predictions, perturb=0.):
        values = []
        for predicted in predictions:
            values.extend([((10. if token == predicted else -3.)+perturb) for token in range(8)])
        raw = struct.pack('<'+'f'*len(values), *values)
        name = f'{label}.step-{index:03d}.f32'
        (folder/name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        return dict(input_ids=supplied, start_position=start, position=start+len(supplied),
            finite_logits_checked=True, logits_sha256=digest,
            logits_artifact=dict(path=name, sha256=digest, dtype='float32', shape=[1,len(predictions),8]),
            routing_by_layer_position={str(layer): {str(p): [p % 4, (p+1) % 4]
                for p in range(start,start+len(supplied))} for layer in range(2)})
    for label in ('serial', 'candidate'):
        report = copy.deepcopy(base)
        turn = dict(id='fixture', status='completed', full_input_ids=[1,2],
            generated_ids=[3,4,5,6,7], full_output_ids=[1,2,3,4,5,6,7],
            eos_token_ids=[7], stop_reason='eos', text='fabricated', raw_text='fabricated<eos>')
        steps = [step(label,0,0,[1,2],[3]), step(label,1,2,[3],[4])]
        if label == 'serial':
            steps += [step(label,2,3,[4],[5]), step(label,3,4,[5],[6]), step(label,4,5,[6],[7])]
        else:
            # Rejected speculative future [0,0] must not be compared against
            # serial states. The valid [4,5] prefix is restored and replayed.
            steps += [step(label,2,3,[4,5,0,0],[5,6,1,1],1e-5),
                      step(label,3,3,[4,5],[6],1e-5), step(label,4,5,[6],[7],1e-5)]
        turn['step_checks'] = steps
        report['turns'] = [turn]
        save(folder/f'{label}.json', report)
    return folder/'serial.json', folder/'candidate.json'


class IndependentSpeculationTests(unittest.TestCase):
    def test_recalculates_emitted_rows_routes_and_ignores_rejected_future(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = synthetic(Path(temporary))
            result = CHECKER.verify(*paths)
            self.assertEqual(result['status'], 'accepted')
            self.assertFalse(result['checkpoint_hash_integrity_validated'])
            self.assertFalse(result['guard_compliance_validated'])
            self.assertFalse(result['timing_speedup_validated'])
            self.assertFalse(result['full_recurrent_state_parity_validated'])
            row = result['comparisons'][0]
            self.assertEqual(row['candidate_rows_checked'], 8)
            self.assertEqual(row['candidate_rows_matching_emitted_prefix'], 6)
            self.assertEqual(row['full_logits_elements_checked'], 48)
            self.assertEqual(row['recorded_routes_checked'], 12)
            self.assertGreater(row['logits_max_absolute_error'], 0.)
            self.assertFalse(row['logits_bitwise_exact'])

    def test_mutated_transcript_positions_routes_source_and_status_are_rejected(self):
        for mode in ('tokens','input','start','end','routes','route_missing','route_topk','source','status','eos','raw'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                serial, candidate = synthetic(Path(temporary))
                report = json.loads(candidate.read_text())
                turn = report['turns'][0]
                step = turn['step_checks'][2]
                if mode == 'tokens':
                    turn['generated_ids'][2] = 1
                elif mode == 'input':
                    turn['step_checks'][0]['input_ids'][0] = 7
                elif mode == 'start':
                    step['start_position'] = 99
                elif mode == 'end':
                    step['position'] += 1
                elif mode == 'routes':
                    step['routing_by_layer_position']['0']['3'] = [2,3]
                elif mode == 'route_missing':
                    del step['routing_by_layer_position']['0']['3']
                elif mode == 'route_topk':
                    step['routing_by_layer_position']['0']['3'] = [2,2]
                elif mode == 'source':
                    report['source_unchanged'] = False
                elif mode == 'status':
                    report['status'] = 'started'
                elif mode == 'eos':
                    turn['eos_token_ids'] = [5,7]
                else:
                    del turn['raw_text']
                save(candidate, report)
                with self.assertRaises(ValueError):
                    CHECKER.verify(serial, candidate)

    def test_hash_shape_path_missing_rows_and_checkpoint_identity_are_rejected(self):
        for mode in ('hash','shape','path','missing','checkpoint','dtype'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                serial, candidate = synthetic(Path(temporary))
                report = json.loads(candidate.read_text())
                artifact = report['turns'][0]['step_checks'][2]['logits_artifact']
                if mode == 'hash':
                    (candidate.parent/artifact['path']).write_bytes(b'0'*128)
                elif mode == 'shape':
                    artifact['shape'][1] = 5
                elif mode == 'path':
                    artifact['path'] = '../outside.f32'
                elif mode == 'missing':
                    report['turns'][0]['step_checks'].pop()
                elif mode == 'checkpoint':
                    report['model']['mtime_ns'] += 1
                else:
                    artifact['dtype'] = 'float16'
                save(candidate, report)
                with self.assertRaises(ValueError):
                    CHECKER.verify(serial, candidate)

    def test_self_consistent_nonfinite_and_numerical_drift_are_rejected(self):
        for mode in ('nan_discarded','inf_emitted','drift','argmax'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                serial, candidate = synthetic(Path(temporary))
                report = json.loads(candidate.read_text())
                step = report['turns'][0]['step_checks'][2]
                artifact = step['logits_artifact']
                path = candidate.parent/artifact['path']
                values = list(struct.unpack('<32f', path.read_bytes()))
                if mode == 'nan_discarded':
                    values[24] = float('nan')
                elif mode == 'inf_emitted':
                    values[0] = float('inf')
                elif mode == 'drift':
                    values[0] += .5
                else:
                    values[0] = 20.
                raw = struct.pack('<32f', *values)
                path.write_bytes(raw)
                artifact['sha256'] = step['logits_sha256'] = hashlib.sha256(raw).hexdigest()
                save(candidate, report)
                with self.assertRaises(ValueError):
                    CHECKER.verify(serial, candidate)

    def test_reference_must_be_serial_and_tolerances_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory() as temporary:
            serial, candidate = synthetic(Path(temporary))
            with self.assertRaises(ValueError):
                CHECKER.verify(candidate, serial)
            for tolerance in (.1, float('nan'), -1, True):
                with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                    CHECKER.verify(serial, candidate, atol=tolerance)


if __name__ == '__main__':
    unittest.main()
