"""Low-cost synthetic Python children only; no model, native context or inference."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import psutil
from research.glm53_batched_prefill import uid_supervisor as s

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.helper=psutil.Process(os.getpid()).exe()
        self.wrapper=self.root/'fixture_wrapper.py'
        self.fdreport=self.root/'fd-report.json'
        self.benchmark=self.root/'benchmark.lock'
        self.child_code=('import os,json,time;from pathlib import Path;rows=[]\n'
            'for fd in range(3,64):\n'
            ' try:\n'
            '  v=os.fstat(fd);rows.append({"fd":fd,"device":v.st_dev,"inode":v.st_ino})\n'
            ' except OSError:pass\n'
            f'Path({str(self.fdreport)!r}).write_text(json.dumps(rows))\n'
            'time.sleep(1.2)\n')
        self.wrapper.write_text('import argparse,fcntl,subprocess,sys\nfrom pathlib import Path\n'
            'p=argparse.ArgumentParser();p.add_argument("--binary");p.add_argument("--output");p.add_argument("--mode");a=p.parse_args()\n'
            f'lock=Path({str(self.benchmark)!r}).open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
            f'child=subprocess.Popen([a.binary,"-c",{self.child_code!r},"unused1","unused2",a.mode])\n'
            'raise SystemExit(child.wait())\n')
    def tearDown(self):self.temp.cleanup()
    def args(self,mode='full'):
        return ['--binary',self.helper,'--output',str(self.root/'native-output'),'--mode',mode]
    def test_prelaunch_ownership_inherits_fd_to_exact_child_without_double_benchmark_lock(self):
        original=self.wrapper.read_bytes()
        with mock.patch.object(s.tempfile,'gettempdir',return_value=str(self.root)):
            report=s.supervise(self.wrapper,self.args(),self.root/'receipt.json',poll_seconds=.05)
        self.assertEqual(report['status'],'completed')
        self.assertTrue(report['uid_model_lock']['acquired_before_wrapper_spawn'])
        self.assertEqual(len(report['native_children']),1)
        child=report['native_children'][0]
        self.assertTrue(child['identity_verified'])
        self.assertTrue(child['pass_fds_requested'])
        inherited=json.loads(self.fdreport.read_text())
        self.assertIn({'fd':report['uid_model_lock']['fd'],'device':report['uid_model_lock']['device'],'inode':report['uid_model_lock']['inode']},inherited)
        self.assertEqual(self.wrapper.read_bytes(),original)
        self.assertTrue(report['wrapper_file_unchanged'])
        self.assertTrue(report['cleanup']['parent_uid_fd_closed'])
        self.assertEqual(report['native_signal_actions'],[])
        self.assertTrue(self.benchmark.exists())
    def test_busy_uid_lock_refuses_before_wrapper_or_native_launch(self):
        with mock.patch.object(s.tempfile,'gettempdir',return_value=str(self.root)):
            holder,_=s.acquire_uid_lock()
            try:report=s.supervise(self.wrapper,self.args(),self.root/'busy.json')
            finally:holder.close()
        self.assertEqual(report['status'],'blocked')
        self.assertIn('ENGINE_BUSY',report['error'])
        self.assertNotIn('wrapper_process',report)
        self.assertFalse(self.fdreport.exists())
        self.assertFalse(self.benchmark.exists())
    def test_receipt_cannot_precreate_native_output_directory(self):
        with self.assertRaises(ValueError):s.supervise(self.wrapper,self.args(),self.root/'native-output'/'supervisor.json')
    def test_only_full_restore_are_forwarded(self):
        with self.assertRaises(SystemExit):s.supervise(self.wrapper,self.args('split'),self.root/'bad-mode.json')
    def test_native_descendant_retains_fd_after_bootstrap_owner_is_killed(self):
        # This deliberately kills only the synthetic bootstrap created below.
        # The retained descriptor, not the supervisor's descriptor, must block
        # a second UID-model owner until the synthetic grandchild is stopped.
        receipt=self.root/'events.jsonl';receipt.write_text('')
        with mock.patch.object(s.tempfile,'gettempdir',return_value=str(self.root)):
            holder,lock=s.acquire_uid_lock()
        bootstrap=subprocess.Popen([sys.executable,'-c',s.BOOTSTRAP,str(holder.fileno()),str(receipt),
            str(Path(self.helper).resolve()),str(self.wrapper),*self.args()],pass_fds=(holder.fileno(),),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        native=None;competitor=None
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                rows=s._events(receipt)
                spawned=[row for row in rows if row.get('event')=='native_spawned']
                if spawned and self.fdreport.exists():break
                time.sleep(.02)
            self.assertTrue(spawned)
            native=spawned[0]
            s.own_identity(bootstrap.pid,psutil.Process(os.getpid()).exe())
            bootstrap.kill();bootstrap.wait(timeout=3)
            holder.close()
            competitor=Path(lock['path']).open('a')
            with self.assertRaises(BlockingIOError):fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity=s.own_identity(native['pid'],self.helper)
            self.assertEqual(identity['create_time'],native['create_time'])
        finally:
            if not holder.closed:holder.close()
            if bootstrap.poll() is None:bootstrap.kill();bootstrap.wait(timeout=3)
            if native is not None:
                try:
                    identity=s.own_identity(native['pid'],self.helper)
                    if identity['create_time']==native['create_time']:
                        process=psutil.Process(native['pid']);process.terminate()
                        try:process.wait(timeout=3)
                        except psutil.TimeoutExpired:pass
                except (psutil.Error,RuntimeError):pass
            if competitor is not None:competitor.close()

if __name__=='__main__':unittest.main()
