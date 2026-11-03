"""Fake diagnostic orchestration only; no native executable, model or counters."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import run_native


class NativeLockAcceptanceTests(unittest.TestCase):
    def test_guard_rejection_cannot_accept_retention_and_returned_stderr_is_used(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, probe = root/'toy.gguf', root/'probe'
            model.write_bytes(b'fixture weights'); probe.write_bytes(b'fixture binary')
            dense = {'budget_mib': 128, 'names': ['blk.3.ssm_f_a.weight'], 'sha256': 'a'*64}
            args = SimpleNamespace(requests=1, state_hashes=False, policy='cpu', gpu_layers=0,
                cache_mib=0, q8_shared=False, q8_cpu_arith=False, reference=None,
                tokens=4, model=model, probe=probe, threads=10, prompt='fixture', source_plan=False,
                mmap_nocache=True, dense_lookahead=True, lookahead_mib=32, profile_ops=False,
                profile_pages=False, layer_local_hc=False, disable_hc_post=False,
                cache_lookup_split=False, metal_timing=False, expert_drop=False,
                mapped_cpu_weights=False, runtime_libraries=root, tail_layers=5,
                dense_lock_plan=root/'selector.json')
            for status, expected in (('stopped_or_failed', False), ('completed', True)):
                args.output = root/status
                stderr = args.output/'returned-path.stderr.log'
                result = {'status': status, 'guard_violation': None if expected else 'fixture_guard',
                          'stderr_file': str(stderr)}
                previous = dict(os.environ)
                with patch.object(run_native, 'load_plan', return_value=dense), \
                     patch.object(run_native, 'check_plan', return_value={'passed': True, 'sha256': dense['sha256']}), \
                     patch.object(run_native, 'retained_log', return_value={'passed': True}) as retained, \
                     patch.object(run_native, 'SystemMemoryObserver') as observer, \
                     patch.object(run_native, 'guard', return_value=result), patch('builtins.print'):
                    observer.return_value.snapshot.return_value = {'available': True}
                    code = run_native.run(args)
                self.assertEqual(dict(os.environ), previous)
                retained.assert_called_once_with(str(stderr), dense)
                manifest = json.loads((args.output/'experiment.json').read_text())
                self.assertEqual(manifest['dense_lock_accepted'], expected)
                self.assertEqual(code, 0 if expected else 2)
