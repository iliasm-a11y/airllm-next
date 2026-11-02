"""Packaging helpers only: no build, signing, native import or inference."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_app import absolute_python_path, bundle_monitor, bundle_runtime, refresh_monitor_manifest, refresh_runtime_manifest


def runtime_fixture(root):
    runtime=root/'runtime'; binary=runtime/'bin'; binary.mkdir(parents=True)
    (binary/'llama-server').write_bytes(b'fixture-server')
    (binary/'libfixture.1.dylib').write_bytes(b'fixture-library')
    (binary/'libfixture.dylib').symlink_to('libfixture.1.dylib')
    (runtime/'licenses').mkdir(); (runtime/'licenses/LICENSE').write_text('fixture-license')
    artifacts=[{'path':'bin/'+p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (binary/'llama-server',binary/'libfixture.1.dylib')]
    artifacts.append({'path':'bin/libfixture.dylib','symlink':'libfixture.1.dylib'})
    (runtime/'manifest.json').write_text(json.dumps({'status':'built',
        'source_revision':'3f8af15a76ac8675812e856da45454bafc90f12c','artifacts':artifacts}))
    (root/'verification.json').write_text(json.dumps({'files':[],'integrity_verified':True}))
    return {'runtime_binary':'runtime/bin/llama-server','runtime_manifest_file':'runtime/manifest.json',
        'verification_file':'verification.json','model_path':'~/Models/external-model','state_dir':'/old/work/state',
        'expected_runtime_sha256':artifacts[0]['sha256']}


class BuildHelperTests(unittest.TestCase):
    def test_embedded_runtime_moves_with_resources_and_preserves_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config=runtime_fixture(root); binary=root/'runtime/bin'
            (root / 'never-copy.gguf').write_bytes(b'fixture-weight-outside-runtime')
            resources = root / 'Resources'; resources.mkdir()
            config = bundle_runtime(config, resources, root)
            self.assertEqual(config['runtime_binary'], 'native-runtime/bin/llama-server')
            self.assertEqual(config['runtime_manifest_file'], 'native-runtime/manifest.json')
            self.assertEqual(config['verification_file'], 'model-verification.json')
            self.assertEqual(config['model_path'], '~/Models/external-model')
            self.assertNotIn('state_dir', config)
            self.assertTrue((resources / 'native-runtime/bin/libfixture.dylib').is_symlink())
            self.assertFalse(list(resources.rglob('*.gguf')))
            moved = root / 'MovedResources'; resources.rename(moved)
            copied_binary = moved / config['runtime_binary']; copied_binary.write_bytes(b'fixture-after-signing')
            manifest = refresh_runtime_manifest(moved)
            refreshed = json.loads((moved / 'runtime-config.json').read_text())
            self.assertEqual(refreshed['expected_runtime_sha256'], hashlib.sha256(copied_binary.read_bytes()).hexdigest())
            link = next(row for row in manifest['artifacts'] if row['path'] == 'bin/libfixture.dylib')
            self.assertEqual(link['symlink'], 'libfixture.1.dylib')
            self.assertEqual((binary / 'llama-server').read_bytes(), b'fixture-server')

    def test_tampered_source_binary_and_dylib_are_rejected_before_copy(self):
        for name in ('llama-server','libfixture.1.dylib'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); config=runtime_fixture(root); file=root/'runtime/bin'/name
                original=file.read_bytes(); file.write_bytes(b'X'+original[1:])
                resources=root/'Resources'; resources.mkdir()
                with self.assertRaisesRegex(ValueError,'hash mismatch'): bundle_runtime(config,resources,root)
                self.assertFalse((resources/'native-runtime').exists())

    def test_wrong_expected_hash_and_undeclared_library_are_rejected(self):
        for mutation in ('expected','extra','escaping_link'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); config=runtime_fixture(root); resources=root/'Resources'; resources.mkdir()
                if mutation=='expected': config['expected_runtime_sha256']='0'*64
                if mutation=='extra': (root/'runtime/bin/libextra.dylib').write_bytes(b'not-declared')
                if mutation=='escaping_link':
                    link=root/'runtime/bin/libfixture.dylib'; link.unlink(); link.symlink_to('../../external.dylib')
                with self.assertRaises(ValueError): bundle_runtime(config,resources,root)
                self.assertFalse((resources/'native-runtime').exists())

    def test_modified_copy_is_rejected_before_manifest_refresh(self):
        import build_app
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); config=runtime_fixture(root); resources=root/'Resources'; resources.mkdir()
            original_copytree=build_app.shutil.copytree
            def changed_copy(source,destination,*args,**kwargs):
                result=original_copytree(source,destination,*args,**kwargs)
                if Path(destination).name=='bin':
                    file=Path(destination)/'llama-server'; data=file.read_bytes(); file.write_bytes(b'X'+data[1:])
                return result
            with mock.patch.object(build_app.shutil,'copytree',side_effect=changed_copy):
                with self.assertRaisesRegex(ValueError,'hash mismatch'): bundle_runtime(config,resources,root)
            self.assertFalse((resources/'runtime-config.json').exists())

    def test_refresh_refuses_unverified_pristine_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            resources=Path(temp); root=resources/'native-runtime'; root.mkdir()
            path=root/'manifest.json'; original=json.dumps({'status':'built','artifacts':[]}); path.write_text(original)
            with self.assertRaisesRegex(ValueError,'verified pristine'): refresh_runtime_manifest(resources)
            self.assertEqual(path.read_text(),original)

    def test_python_launcher_keeps_venv_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / 'base-python'; base.write_bytes(b'fixture'); base.chmod(0o755)
            launcher = root / 'venv/bin/python'; launcher.parent.mkdir(parents=True); launcher.symlink_to(base)
            result = absolute_python_path(launcher)
            self.assertTrue(result.is_absolute())
            self.assertEqual(result, launcher)
            self.assertNotEqual(result, result.resolve())

    def test_only_monitor_package_and_license_are_copied_with_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); package = root / 'site-packages/psutil'; package.mkdir(parents=True)
            (package / '__init__.py').write_text('__version__ = "7.2.2"\n')
            (package / '_psutil_osx.abi3.so').write_bytes(b'fixture-not-a-real-extension')
            (package / '__pycache__').mkdir(); (package / '__pycache__/ignored.pyc').write_bytes(b'ignored')
            info = package.parent / 'psutil-7.2.2.dist-info'; info.mkdir()
            (info / 'LICENSE').write_text('fixture license')
            (info / 'WHEEL').write_text('Tag: cp36-abi3-macosx_11_0_arm64\n')
            (info / 'METADATA').write_text('Name: psutil\nVersion: 7.2.2\n')
            (package.parent / 'other_dependency').mkdir()
            resources = root / 'Resources'; resources.mkdir()
            manifest = bundle_monitor(package, resources)
            self.assertEqual(manifest['version'], '7.2.2')
            self.assertTrue(manifest['license_bundled'])
            self.assertFalse(manifest['guard_thresholds_changed'])
            self.assertFalse((resources / 'air_llm/other_dependency').exists())
            self.assertFalse((resources / 'air_llm/psutil/__pycache__').exists())
            for item in manifest['files']:
                self.assertEqual(item['bundled_sha256'], hashlib.sha256((resources / item['path']).read_bytes()).hexdigest())
            copied = resources / 'air_llm/psutil/_psutil_osx.abi3.so'
            original_hash = next(item['source_sha256'] for item in manifest['files'] if item['path'].endswith('.so'))
            copied.write_bytes(b'fixture-signature-transform')
            refreshed = refresh_monitor_manifest(resources)
            extension = next(item for item in refreshed['files'] if item['path'].endswith('.so'))
            self.assertEqual(extension['source_sha256'], original_hash)
            self.assertNotEqual(extension['bundled_sha256'], original_hash)

    def test_no_option_copies_no_monitor_and_records_fallback_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            resources = Path(temp)
            manifest = bundle_monitor(None, resources)
            self.assertFalse(manifest['bundled'])
            self.assertEqual(manifest['files'], [])
            self.assertIn('fallback', manifest['selection'])
            self.assertFalse((resources / 'air_llm').exists())
            self.assertEqual(json.loads((resources / 'monitor-manifest.json').read_text()), manifest)


if __name__ == '__main__': unittest.main(verbosity=2)
