"""Tiny historical-metadata/retention fixtures; no model or native execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .dense_lock_plan import check_plan, evidence_matches, load_plan, lock_environment, lock_environment_matches, retained_log


class DenseLockPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root/'toy.gguf'
        self.model.write_bytes(b'0123'*16)
        stat = self.model.stat()
        current = {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
                   'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
        digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.history = self.root/'historical-verification.json'
        self.history.write_text(json.dumps({'integrity_verified': True, 'quantization': 'Q2_K_S',
            'revision': '119034bc3d9d76eb41f91508e6f9fb19fde42672', 'files': [
                {'name': self.model.name, 'bytes': stat.st_size, 'integrity_verified': True,
                 'sha256': digest, 'actual_sha256': digest,
                 'identity': [stat.st_dev+1, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns-1]}]}))
        page = os.sysconf('SC_PAGESIZE')
        name = 'blk.3.ssm_f_a.weight'
        self.document = {'schema_version': 1, 'feature': 'bounded_dense_cpu_mlock_v1',
            'budget_mib': 128, 'page_size': page, 'names': [name],
            'model_revision': '119034bc3d9d76eb41f91508e6f9fb19fde42672',
            'model_file_stats': {self.model.name: current}, 'model_sha256': {self.model.name: digest},
            'identity_manifest_path': str(self.history),
            'identity_manifest_sha256': hashlib.sha256(self.history.read_bytes()).hexdigest(),
            'inventory_sha256': 'a'*64, 'current_full_hash_reverified': False,
            'source_alignment': {'source_plan_sha256': 'b'*64, 'operations_sha256': 'c'*64,
                'source_step': 1, 'names_count': 684, 'old_decode_steps': [1,2,3],
                'ordered_matrix_sizes_verified': True},
            'selected_raw_bytes': 32, 'selected_page_union_bytes': page,
            'selected_tensors': [{'name': name, 'file': self.model.name, 'type': 'F32',
                'dimensions': [4,2], 'tensor_offset': 0, 'absolute_start': 16,
                'payload_bytes': 32, 'page_start': 0, 'page_end': page}],
            'page_unions': [{'file': self.model.name, 'start': 0, 'end': page}]}
        self.path = self.root/'selector.json'
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.document))

    def test_current_five_field_identity_and_historical_scope(self):
        plan = load_plan(self.path, self.model)
        self.assertTrue(check_plan(plan, self.model)['passed'])
        scope = plan['historical_identity_scope'][self.model.name]
        self.assertNotEqual(scope['historical_device'], scope['current_device'])
        self.assertFalse(scope['current_full_sha256_recomputed'])
        self.assertIn('ctime_ns', plan['model_file_stats'][self.model.name])
        self.assertTrue(lock_environment_matches(lock_environment(plan), plan))
        self.assertFalse(lock_environment_matches({'AIRLLM_DENSE_LOCK': '0'}, None))
        self.document['model_file_stats'][self.model.name]['ctime_ns'] += 1
        self.write()
        with self.assertRaisesRegex(ValueError, 'identity/stats'):
            load_plan(self.path, self.model)

    def test_union_type_and_historical_hash_binding_are_not_trusted_blindly(self):
        original = deepcopy(self.document)
        for kind in ('union', 'type', 'hash'):
            self.document = deepcopy(original)
            if kind == 'union': self.document['selected_page_union_bytes'] += 1
            elif kind == 'type': self.document['selected_tensors'][0]['type'] = 'Q8_0'
            else: self.document['model_sha256'][self.model.name] = '0'*64
            self.write()
            with self.assertRaises(ValueError):
                load_plan(self.path, self.model)

    def test_retained_record_must_match_every_bounded_byte_count(self):
        plan = load_plan(self.path, self.model)
        log = self.root/'native.log'
        size = plan['selected_page_union_bytes']
        line = (f'AIRLLM_DENSE_LOCK status=retained retained=1 names=1 raw_bytes=32 '
            f'page_union_bytes={size} locked_bytes={size} lock_calls=1 unlock_calls=0 '
            'verified_pages_at_lock=1 error=0 unlock_error=0\n')
        log.write_text(line)
        evidence = retained_log(log, plan)
        self.assertTrue(evidence_matches(evidence, plan, plan))
        bad = deepcopy(evidence)
        bad['rows'][0]['locked_bytes'] = '1'
        self.assertFalse(evidence_matches(bad, plan, plan))
        log.write_text(line+'AIRLLM_DENSE_LOCK status=admission_poisoned_after_unlock_failure retained=0\n')
        self.assertFalse(retained_log(log, plan)['passed'])

    def test_manifest_and_historical_evidence_drift_rejected(self):
        plan = load_plan(self.path, self.model)
        self.history.write_text(self.history.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'historical checksum evidence changed'):
            check_plan(plan, self.model)
