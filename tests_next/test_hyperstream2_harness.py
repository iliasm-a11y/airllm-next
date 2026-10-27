"""Stdlib-only harness safety checks; every engine/supervisor is a stub.

Subprocess checks run with -S, so even third-party site packages are unavailable.
The restoration test exercises report persistence through the real worker's
finally block, without executing a forward pass or loading any model weights.
"""
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'benchmarks/benchmark_research51.py'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HarnessSafetyTests(unittest.TestCase):
    def setUp(self):
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.model = self.folder / 'toy.gguf'
        self.model.write_bytes(b'protected test input, never an inference checkpoint')
        self.tasks = self.folder / 'tasks.json'
        self.tasks.write_text(json.dumps([dict(id='one', prompt='Question', max_new_tokens=3)]))
        self.output = self.folder / 'trial.json'
        self.harness = load_source('hs2_harness_safety', SCRIPT)
        self.addCleanup(sys.modules.pop, 'hs2_harness_safety', None)
        self.engine = self.folder / 'engine'
        source = self.engine / 'air_llm/airllm/nextgen/stub.py'
        source.parent.mkdir(parents=True)
        source.write_text('# protected synthetic source\n')

    def argv(self, *flags):
        return [str(SCRIPT), '--model', str(self.model), '--tasks', str(self.tasks),
                '--output', str(self.output), '--engine-root', str(self.engine), *flags]

    def guard_modules(self):
        # Load actual path validation and conservative dataclass defaults with
        # psutil stubbed; neither system sampling nor supervision is exercised.
        with patch.dict(sys.modules, {'psutil': ModuleType('psutil')}):
            guard = load_source('hs2_guard_safety', ROOT / 'air_llm/airllm/nextgen/experiment_guard.py')
        self.addCleanup(sys.modules.pop, 'hs2_guard_safety', None)
        guard.supervise = Mock(name='forbidden real supervision')
        packages = {}
        for name in ['airllm', 'airllm.nextgen']:
            package = ModuleType(name)
            package.__path__ = []
            packages[name] = package
        packages['airllm.nextgen.experiment_guard'] = guard
        return guard, packages

    def test_help_works_without_site_packages_or_engine_imports(self):
        result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ['--request-trace', '--numerical-evidence', '--hyper-prefill']:
            self.assertIn(flag, result.stdout)
        self.assertFalse(self.output.exists())

    def test_trace_and_numerical_flags_reject_multiple_tasks_before_dependencies(self):
        self.tasks.write_text(json.dumps([
            dict(id='one', prompt='Question', max_new_tokens=3),
            dict(id='two', prompt='Other', max_new_tokens=3)]))
        for flag in ['--request-trace', '--numerical-evidence']:
            with self.subTest(flag=flag):
                result = subprocess.run([sys.executable, '-S', *self.argv(flag)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('exactly one task per worker', result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertFalse(self.output.exists())

    def test_all_trace_and_vector_destinations_are_validated_before_launch(self):
        guard, packages = self.guard_modules()
        validate = Mock(wraps=guard.validate_output_paths)
        guard.validate_output_paths = validate
        with patch.dict(sys.modules, packages), patch.object(sys, 'argv', self.argv(
                '--request-trace', '--numerical-evidence')):
            self.harness.main()
        expected = [self.folder / 'trial.trace.json',
                    *[self.folder / ('trial.step-' + str(i).zfill(3) + '.f32') for i in range(3)]]
        self.assertEqual(validate.call_args.kwargs['extra_outputs'], expected)
        self.assertIn(self.model, validate.call_args.kwargs['protected'])
        self.assertIn(self.tasks, validate.call_args.kwargs['protected'])
        guard.supervise.assert_called_once()
        command, output, limits = guard.supervise.call_args.args
        self.assertIn('--worker', command)
        self.assertEqual(output, self.output)
        self.assertEqual(limits.rss_bytes, 2 << 30)
        self.assertEqual(limits.reserve_bytes, 3 << 30)
        self.assertEqual(limits.swap_growth_bytes, 128 << 20)
        self.assertEqual(limits.max_processes, 32)
        self.assertFalse(self.output.exists())

    def test_existing_evidence_in_any_destination_prevents_launch_and_is_preserved(self):
        guard, packages = self.guard_modules()
        destinations = [self.output, self.folder / 'trial.log', self.folder / 'trial.guard.json',
                        self.folder / 'trial.trace.json', self.folder / 'trial.step-000.f32',
                        self.folder / 'trial.step-002.f32']
        for destination in destinations:
            with self.subTest(destination=destination.name):
                destination.write_bytes(b'existing evidence')
                with patch.dict(sys.modules, packages), patch.object(sys, 'argv', self.argv(
                        '--request-trace', '--numerical-evidence')):
                    with self.assertRaisesRegex(FileExistsError, 'prior experiment evidence'):
                        self.harness.main()
                self.assertEqual(destination.read_bytes(), b'existing evidence')
                guard.supervise.assert_not_called()
                destination.unlink()

    def test_vector_hardlink_to_input_and_trace_broken_symlink_are_rejected(self):
        guard, packages = self.guard_modules()
        vector = self.folder / 'trial.step-001.f32'
        vector.hardlink_to(self.model)
        before = self.model.read_bytes()
        with patch.dict(sys.modules, packages), patch.object(sys, 'argv', self.argv('--numerical-evidence')):
            with self.assertRaisesRegex(ValueError, 'protected input'):
                self.harness.main()
        vector.unlink()
        trace = self.folder / 'trial.trace.json'
        trace.symlink_to(self.folder / 'absent target')
        with patch.dict(sys.modules, packages), patch.object(sys, 'argv', self.argv('--request-trace')):
            with self.assertRaisesRegex(FileExistsError, 'prior experiment evidence'):
                self.harness.main()
        self.assertEqual(self.model.read_bytes(), before)
        self.assertTrue(trace.is_symlink())
        guard.supervise.assert_not_called()

    def test_observer_restoration_failure_is_saved_as_failed_evidence(self):
        guard, modules = self.guard_modules()
        guard.system_snapshot = lambda: dict(available_bytes=4 << 30, swap_used_bytes=0, pressure_level=1)

        @dataclass
        class Options:
            synthetic: bool = True

            def __init__(self, **kwargs):
                self.synthetic = True

        class Checkpoint:
            def __init__(self, *args, **kwargs):
                self.metadata = {}
                self.stats = dict(logical_bytes=0, read_calls=0)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class Model:
            forwards = 0

            def __init__(self, *args, **kwargs):
                self.config = {}
                self.checkpoint = Checkpoint()
                self.reader = SimpleNamespace()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def forward(self, *args, **kwargs):
                Model.forwards += 1
                raise AssertionError('No forward pass is authorized in this test')

            def metrics(self):
                return {}

        class Tokenizer:
            def __init__(self, *args):
                pass

        class Chat:
            def __init__(self, *args, **kwargs):
                pass

            def reply(self, *args, **kwargs):
                return dict(text='synthetic reply', ttft_seconds=.01,
                            decode_tokens_per_second=1., stop_reason='eos')

        class BrokenObserver:
            routing = {}

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                raise RuntimeError('synthetic restoration failure')

            def routing_sha256(self):
                return 'synthetic hash'

            def summary(self):
                return {}

        for name, exports in {
            'numpy': {}, 'airllm.nextgen.config': dict(NextConfig=Options),
            'airllm.nextgen.gguf': dict(GGUFReader=Checkpoint),
            'airllm.nextgen.gguf_chat': dict(GGUFChatSession=Chat),
            'airllm.nextgen.gguf_qwen': dict(NativeGGUFQwenModel=Model),
            'airllm.nextgen.gguf_tokenizer': dict(GGUFTokenizer=Tokenizer),
            'airllm.nextgen.physical_io': dict(physical_read_bytes=lambda: 0),
            'hyperstream2_observer': dict(RuntimeObserver=BrokenObserver),
        }.items():
            module = ModuleType(name)
            module.__dict__.update(exports)
            modules[name] = module
        manifest = dict(git_head='synthetic commit', source_sha256={'synthetic.py': 'synthetic hash'})
        original_path = list(sys.path)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, modules))
                stack.enter_context(patch.object(sys, 'argv', self.argv('--worker', '--numerical-evidence')))
                stack.enter_context(patch.object(self.harness, 'engine_manifest', return_value=manifest))
                stack.enter_context(patch.object(self.harness, 'source_manifest', return_value=manifest['source_sha256']))
                stack.enter_context(patch('importlib.metadata.version', return_value='synthetic version'))
                stack.enter_context(patch.object(self.harness.platform, 'platform', return_value='synthetic platform'))
                stack.enter_context(patch.object(self.harness.subprocess, 'check_output', return_value='synthetic cpu\n'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, 'Observer restoration failed; evidence invalid'):
                    self.harness.main()
        finally:
            sys.path[:] = original_path
        report = json.loads(self.output.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIn('synthetic restoration failure', report['observer_restoration_error'])
        self.assertIn('after', report)
        self.assertTrue(report['source_unchanged'])
        self.assertTrue(report['protocol_unchanged'])
        self.assertTrue(report['model_stat_unchanged'])
        self.assertTrue(report['engine_head_unchanged'])
        self.assertEqual(Model.forwards, 0)
        guard.supervise.assert_not_called()


if __name__ == '__main__':
    unittest.main()
