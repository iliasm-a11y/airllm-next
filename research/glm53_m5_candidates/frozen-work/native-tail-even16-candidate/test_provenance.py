"""Negative provenance fixtures only; no compiler/native process is executed."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
b=module('provenance_build_fixture','build_helper.py')
v=module('provenance_compare_fixture','variant_compare.py')

def build_fixture(root):
    headers={}
    for name in b.HEADER_NAMES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('// synthetic header fixture\n')
        headers[name]=str(path)
    return {'libraries':str(root),'output':str(root/'not-a-real-binary'), 'command':['never-run-a-compiler'],
        'runtime_libraries_expected_sha256':{},'header_paths':headers,
        'header_sha256':{name:b.sha(path) for name,path in headers.items()},
        'inputs_sha256':{path:b.sha(path) for path in headers.values()},'build_attempted':False,'build_performed':False,'binary_rebuilt':False}

def module_fixture(root):
    pins={}
    for name,relative in v.DELEGATED_MODULE_PATHS.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('FIXTURE_ONLY = True\n')
        pins[name]=v.sha(path)
    return pins

class ProvenanceTests(unittest.TestCase):
    def test_each_explicit_header_drift_before_compile_refuses_without_process(self):
        for name in b.HEADER_NAMES:
            with self.subTest(header=name), tempfile.TemporaryDirectory() as directory:
                record=build_fixture(Path(directory));Path(record['header_paths'][name]).write_text('// changed after plan\n')
                with mock.patch.object(b.subprocess,'run',side_effect=AssertionError('must not spawn')) as spawned:
                    with self.assertRaisesRegex(ValueError,'before compile'):b.compile_plan(record)
                    spawned.assert_not_called()
    def test_each_header_drift_during_synthetic_callback_refuses_before_smoke(self):
        for name in b.HEADER_NAMES:
            with self.subTest(header=name), tempfile.TemporaryDirectory() as directory:
                record=build_fixture(Path(directory))
                def synthetic_callback(*args,**kwargs):
                    Path(record['output']).write_bytes(b'fixture bytes, not a compiled executable')
                    Path(record['header_paths'][name]).write_text('// changed during synthetic callback\n')
                    return SimpleNamespace(returncode=0,stdout='',stderr='')
                with mock.patch.object(b.subprocess,'run',side_effect=synthetic_callback) as spawned:
                    with self.assertRaisesRegex(ValueError,'after compile'):b.compile_plan(record)
                    self.assertEqual(spawned.call_count,1) # Fake callback only; no metadata smoke reached.
                self.assertNotIn('parameter_smoke',record)
    def test_header_declaration_outside_tracked_inputs_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            record=build_fixture(Path(directory));del record['inputs_sha256'][record['header_paths']['vendor/nlohmann/json.hpp']]
            with self.assertRaisesRegex(ValueError,'outside tracked'):b.verify_tracked_inputs(record,'before compile')
    def test_each_delegate_file_drift_is_rejected_without_touching_real_modules(self):
        for name,relative in v.DELEGATED_MODULE_PATHS.items():
            with self.subTest(module=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);pins=module_fixture(root)
                v.verify_module_files(root,pins,loaded_modules={})
                (root/relative).write_text('FIXTURE_ONLY = False\n')
                with self.assertRaisesRegex(ValueError,'source changed'):v.verify_module_files(root,pins,loaded_modules={})
    def test_canonical_loader_checks_fingerprints_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pins=module_fixture(root);first=next(iter(v.DELEGATED_MODULE_PATHS.values()))
            (root/first).write_text('CHANGED_BEFORE_IMPORT = True\n')
            (root/'prepare-manifest.json').write_text(json.dumps({'delegated_module_hashes':pins}))
            with mock.patch.object(v,'ROOT',root),mock.patch.object(v,'REPO',root), \
                 mock.patch.object(v.importlib,'import_module',side_effect=AssertionError('must not import')) as imported:
                with self.assertRaisesRegex(ValueError,'source changed'):v.canonical_key({})
                imported.assert_not_called()
    def test_cached_delegate_from_another_file_and_missing_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pins=module_fixture(root);name=next(iter(v.DELEGATED_MODULE_PATHS))
            wrong={name:SimpleNamespace(__file__=str(root/'wrong.py'))}
            with self.assertRaisesRegex(ValueError,'another source'):v.verify_module_files(root,pins,loaded_modules=wrong)
            del pins[name]
            with self.assertRaisesRegex(ValueError,'complete delegated'):v.verify_module_files(root,pins,loaded_modules={})

if __name__=='__main__':unittest.main()
