"""CPU/stdlib-only checks for observer transparency, routing and restoration."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[1] / 'benchmarks/hyperstream2_observer.py'
_SPEC = importlib.util.spec_from_file_location('isolated_hs2_observer', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RuntimeObserver = _MODULE.RuntimeObserver


class Array:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (1, len(rows), 3)

    def tolist(self):
        return [list(row) for row in self.rows]


class Reader:
    def __init__(self, checkpoint, mx):
        self.checkpoint = checkpoint
        self.mx = mx

    def linear(self, x, key):
        self.checkpoint._pread(4, 0, tensor=True)
        self.mx.eval(x)
        return x


class Checkpoint:
    fail = False

    def _pread(self, count, offset, **kwargs):
        if self.fail:
            raise OSError('read failed')
        return b'x' * count


class Mixer:
    label = 'original mixer'

    def __call__(self, x, cache):
        return x


class MLP:
    label = 'original MLP'

    def __init__(self, module, reader):
        self.module = module
        self.reader = reader

    def __call__(self, x):
        self.reader.linear(x, 'expert')
        self.module.qwen_route(None, x, None, {})
        return x


class RuntimeObserverTests(unittest.TestCase):
    def fixture(self):
        package = ModuleType('observer_test_package')
        package.__path__ = []
        module = ModuleType('observer_test_package.gguf_qwen')
        module.__package__ = 'observer_test_package'
        module.qwen_route = lambda mx, x, weight, config: (x, 'original scores')
        packed = ModuleType('observer_test_package.gguf_packed_io')
        packed._read_into_once = lambda fd, destination, offset: len(destination)
        mx = SimpleNamespace(eval=lambda *args: None, synchronize=lambda: None)
        checkpoint = Checkpoint()
        reader = Reader(checkpoint, mx)
        model_type = type('Model', (), {'__module__': module.__name__})
        model = model_type()
        model.reader = reader
        model.checkpoint = checkpoint
        model.mx = mx
        model.mixers = [Mixer()]
        model.mlps = [MLP(module, reader)]
        modules = {package.__name__: package, module.__name__: module, packed.__name__: packed}
        return model, module, packed, modules

    def test_routes_use_absolute_positions_across_prefill_and_decode(self):
        model, module, packed, modules = self.fixture()
        original_lists = model.mixers, model.mlps
        original_route = module.qwen_route
        original_eval = model.mx.eval
        original_syscall = packed._read_into_once
        with patch.dict(sys.modules, modules), RuntimeObserver(model) as observer:
            self.assertEqual(model.mixers[0].label, 'original mixer')
            self.assertEqual(model.mlps[0].label, 'original MLP')
            for phase, start, rows in [('prefill', 0, [[7, 2], [1, 4]]),
                                       ('prefill', 2, [[3, 6]]), ('decode', 3, [[2, 5]])]:
                observer.phase = phase
                array = Array(rows)
                cache = SimpleNamespace(position=start)
                self.assertIs(model.mixers[0](array, cache), array)
                self.assertIs(model.mlps[0](array), array)
            self.assertEqual(packed._read_into_once(0, b'abcd', 0), 4)
            self.assertEqual(observer.routing, {'0': {'0': [7, 2], '1': [1, 4],
                                                   '2': [3, 6], '3': [2, 5]}})
            routes = observer.routing
            routes['0']['0'][0] = -10
            self.assertEqual(observer.routing['0']['0'], [7, 2])
            self.assertEqual(len(observer.routing_sha256()), 64)
            summary = observer.summary()
            self.assertEqual(summary['route_positions'], 4)
            self.assertEqual(summary['methods']['_pread']['returned_bytes'], 12)
            self.assertEqual(summary['methods']['eval']['calls'], 3)
            self.assertEqual(summary['methods']['_read_into_once']['returned_bytes'], 4)
            self.assertEqual(summary['phases']['prefill']['qwen_route']['calls'], 2)
            self.assertEqual(summary['phases']['decode']['qwen_route']['calls'], 1)
            self.assertFalse(summary['attribution']['adds_synchronization_commands'])
        self.assertIs(model.mixers, original_lists[0])
        self.assertIs(model.mlps, original_lists[1])
        self.assertIs(module.qwen_route, original_route)
        self.assertIs(model.mx.eval, original_eval)
        self.assertIs(packed._read_into_once, original_syscall)
        self.assertNotIn('linear', model.reader.__dict__)
        self.assertNotIn('_pread', model.checkpoint.__dict__)

    def test_read_error_is_propagated_and_all_wrappers_are_restored(self):
        model, module, packed, modules = self.fixture()
        route = module.qwen_route
        mixers = model.mixers
        with patch.dict(sys.modules, modules):
            observer = RuntimeObserver(model)
            with self.assertRaisesRegex(OSError, 'read failed'):
                with observer:
                    model.checkpoint.fail = True
                    model.reader.linear(Array([[1, 2]]), 'expert')
            self.assertEqual(observer.summary()['methods']['linear']['errors'], 1)
            self.assertEqual(observer.summary()['methods']['_pread']['errors'], 1)
            self.assertIs(module.qwen_route, route)
            self.assertIs(model.mixers, mixers)
            self.assertNotIn('linear', model.reader.__dict__)
            model.checkpoint.fail = False
            with RuntimeObserver(model):
                model.reader.linear(Array([[1, 2]]), 'expert')

    def test_partial_enter_failure_restores_methods_and_releases_process_lock(self):
        model, module, packed, modules = self.fixture()
        route = module.qwen_route
        del module.qwen_route
        original_eval = model.mx.eval
        original_mixers = model.mixers
        with patch.dict(sys.modules, modules):
            with self.assertRaises(AttributeError):
                with RuntimeObserver(model):
                    self.fail('entry should fail')
            self.assertIs(model.mx.eval, original_eval)
            self.assertIs(model.mixers, original_mixers)
            self.assertNotIn('_pread', model.checkpoint.__dict__)
            module.qwen_route = route
            with RuntimeObserver(model):
                pass

    def test_conflicting_routes_at_one_position_fail_instead_of_hiding_change(self):
        model, module, packed, modules = self.fixture()
        with patch.dict(sys.modules, modules), RuntimeObserver(model) as observer:
            cache = SimpleNamespace(position=0)
            array = Array([[1, 2]])
            model.mixers[0](array, cache)
            model.mlps[0](array)
            with self.assertRaisesRegex(ValueError, 'Conflicting route'):
                model.mlps[0](Array([[3, 2]]))
            self.assertEqual(observer.routing, {'0': {'0': [1, 2]}})

    def test_nested_observers_are_rejected_without_disturbing_active_observer(self):
        model, module, packed, modules = self.fixture()
        with patch.dict(sys.modules, modules), RuntimeObserver(model) as observer:
            wrapped = model.mx.eval
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                with RuntimeObserver(model):
                    pass
            self.assertIs(model.mx.eval, wrapped)
            model.mx.eval()
            self.assertEqual(observer.summary()['methods']['eval']['calls'], 1)


if __name__ == '__main__':
    unittest.main()
