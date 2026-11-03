"""Only tiny synthetic fixtures in a temporary directory; no real proof dedup."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import psutil
import dedup_closed_captures as d


def gone(pid):raise psutil.NoSuchProcess(pid)
class DedupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        patch=mock.patch.object(d,'BENCHMARK_LOCK',self.root/'benchmark.lock');patch.start();self.addCleanup(patch.stop)
        self.run=self.root/'native-fixture';self.capture=self.run/'capture';self.capture.mkdir(parents=True)
        self.guard={'status':'completed','returncode':0,'owned_pid':999999,
            'owned_identity':{'pid':999999,'create_time':123.0,'executable':'/fixture/native-helper'}}
        self.guardpath=self.run/'guard.json';self.guardpath.write_text(json.dumps(self.guard))
        self.rows=[]
        for index,raw in enumerate((b'a'*1024,b'a'*1024,b'b'*1024)):
            name=f'tensor-{index}.bin';(self.capture/name).write_bytes(raw)
            self.rows.append({'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'name':'kda_scan_out-0'})
        self.manifest=self.capture/'tensors.jsonl'
        self.manifest.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        (self.capture/'report.json').write_text(json.dumps({'requests':[]}))
    def tearDown(self):self.temp.cleanup()
    def call(self,apply=False):
        with mock.patch.object(d,'WORK_ROOT',self.root):return d.deduplicate(self.run,apply=apply,process_factory=gone)
    def test_plan_hashes_and_compares_bytes_without_replacing_payloads(self):
        before=[(self.capture/f'tensor-{i}.bin').stat().st_ino for i in range(3)]
        report=self.call()
        self.assertEqual(report['mode'],'plan');self.assertEqual(report['payload_paths_replaced'],0)
        self.assertEqual(report['redundant_logical_bytes_estimate'],1024)
        self.assertEqual(before,[(self.capture/f'tensor-{i}.bin').stat().st_ino for i in range(3)])
    def test_apply_preserves_all_names_sizes_hashes_and_metadata(self):
        meta_before=self.manifest.read_bytes();guard_before=self.guardpath.read_bytes()
        sentinel=self.capture/'unrelated-user-note.txt';sentinel.write_text('fixture: do not touch')
        report=self.call(True)
        a,b,c=[self.capture/f'tensor-{i}.bin' for i in range(3)]
        self.assertEqual(a.stat().st_ino,b.stat().st_ino);self.assertNotEqual(a.stat().st_ino,c.stat().st_ino)
        self.assertEqual(report['payload_paths_replaced'],1);self.assertTrue(report['all_original_names_sizes_sha_verified'])
        self.assertEqual(self.manifest.read_bytes(),meta_before);self.assertEqual(self.guardpath.read_bytes(),guard_before)
        self.assertEqual(sentinel.read_text(),'fixture: do not touch')
        self.assertEqual(d.digest(a),self.rows[0]['sha256']);self.assertEqual(d.digest(b),self.rows[1]['sha256'])
        self.assertFalse(list(self.capture.glob('.*.dedup-*')))
    def test_exact_native_result_output_name_is_eligible(self):
        self.rows[1]['name']='result_output'
        self.manifest.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        report=self.call(False)
        self.assertEqual(report['eligible_payload_paths'],3)
        self.assertEqual(report['redundant_logical_bytes_estimate'],1024)
    def test_active_or_unclosed_capture_is_refused(self):
        self.guard['status']='running';self.guard['returncode']=None;self.guardpath.write_text(json.dumps(self.guard))
        with self.assertRaises(ValueError):self.call(True)
        self.guard.update(status='completed',returncode=0,owned_pid=os.getpid(),owned_identity={'pid':os.getpid(),
            'create_time':psutil.Process().create_time(),'executable':psutil.Process().exe()})
        self.guardpath.write_text(json.dumps(self.guard))
        with mock.patch.object(d,'WORK_ROOT',self.root):
            with self.assertRaises(ValueError):d.deduplicate(self.run,apply=True)
    def test_hash_mismatch_and_symlink_are_refused_before_replacement(self):
        target=self.capture/'tensor-1.bin';original_inode=target.stat().st_ino;target.write_bytes(b'c'*1024)
        with self.assertRaises(ValueError):self.call(True)
        self.assertEqual(target.stat().st_ino,original_inode)
        target.unlink();outside=self.root/'external.bin';outside.write_bytes(b'a'*1024);target.symlink_to(outside)
        with self.assertRaises(ValueError):self.call(True)
        self.assertEqual(outside.read_bytes(),b'a'*1024)
    def test_atomic_replacement_rolls_back_if_post_hash_verification_fails(self):
        a,b=self.capture/'tensor-0.bin',self.capture/'tensor-1.bin';old_inode=b.stat().st_ino;real_digest=d.digest
        def faulty(path):
            if path==b and b.stat().st_ino==a.stat().st_ino:return '0'*64
            return real_digest(path)
        with mock.patch.object(d,'digest',side_effect=faulty):
            with self.assertRaises(ValueError):d.replace_by_hardlink(a,b,self.rows[0]['sha256'])
        self.assertEqual(b.stat().st_ino,old_inode);self.assertEqual(b.read_bytes(),a.read_bytes())
        self.assertFalse(list(self.capture.glob('.*.dedup-*')))
    def test_ongoing_benchmark_refuses_even_closed_capture_hashing(self):
        with d.BENCHMARK_LOCK.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with mock.patch.object(d,'digest') as hashed:
                with self.assertRaises(BlockingIOError):self.call(True)
                hashed.assert_not_called()
    def test_outside_work_run_is_refused_before_payload_access(self):
        with mock.patch.object(d,'WORK_ROOT',self.root/'another-workspace'):
            with self.assertRaises(ValueError):d.check_closed(self.run,process_factory=gone)

if __name__=='__main__':unittest.main()
