"""CLI/config contracts, including preflight failures before any checkpoint read."""
from argparse import Namespace
import builtins
import json
from types import SimpleNamespace

import pytest

from airllm.nextgen.cli import config_for, main
from airllm.nextgen.config import NextConfig


@pytest.mark.parametrize('field', [
    'cache_bytes', 'io_bytes', 'kv_bytes', 'workspace_bytes', 'reserve_bytes',
    'max_tensor_bytes', 'io_workers', 'max_open_files', 'row_chunk', 'prefill_chunk', 'max_context',
])
@pytest.mark.parametrize('value', [True, 1.5, float('inf'), float('nan'), '1024'])
def test_memory_and_size_options_require_integers(field, value):
    with pytest.raises(ValueError, match=field):
        NextConfig(**{field: value})


@pytest.mark.parametrize('field', [
    'prefetch', 'coalesce_reads', 'selective_fp8_scales', 'lossless_verify_hash',
    'load_tokenizer', 'use_saved_tuning',
])
def test_switch_options_require_booleans(field):
    with pytest.raises(ValueError, match=field):
        NextConfig(**{field: 'false'})


def test_memory_and_lossless_controls_reach_runtime_config(tmp_path,monkeypatch):
    from airllm.nextgen import tuning
    monkeypatch.setattr(tuning.psutil,'virtual_memory',lambda:SimpleNamespace(total=16*1024**3,available=128*1024**2))
    args = Namespace(profile='balanced', device='cpu', token_ids=[1],
        cache_mib=0, io_mib=2, kv_mib=3, workspace_mib=4, reserve_mib=5,
        max_tensor_mib=1, row_chunk=7, prefill_chunk=2, io_workers=1,
        lossless_dir=str(tmp_path/'packed'), lossless_native_library=None,
        max_context=32)
    cfg = config_for(args)
    assert (cfg.cache_bytes,cfg.io_bytes,cfg.kv_bytes,cfg.workspace_bytes,cfg.reserve_bytes) == tuple(x*1024**2 for x in (0,2,3,4,5))
    assert (cfg.max_tensor_bytes,cfg.row_chunk,cfg.prefill_chunk,cfg.io_workers,cfg.max_context) == (1024**2,7,2,1,32)
    assert cfg.lossless_dir == str(tmp_path/'packed')
    assert cfg.lossless_verify_hash and not cfg.load_tokenizer


def test_automatic_memory_reserve_remains_enforced(monkeypatch):
    from airllm.nextgen import tuning
    monkeypatch.setattr(tuning.psutil,'virtual_memory',lambda:SimpleNamespace(total=16*1024**3,available=128*1024**2))
    with pytest.raises(MemoryError,match='available memory'):
        tuning.auto_config()
    with pytest.raises(MemoryError,match='plus reserve'):
        tuning.auto_config(cache_bytes=128*1024**2,io_bytes=1,kv_bytes=1,workspace_bytes=1,reserve_bytes=0)


@pytest.mark.parametrize('command',['prepare','tune','limits','convert','verify','pack'])
def test_metadata_config_does_not_request_inference_memory(command,monkeypatch):
    from airllm.nextgen import cli
    def no_inference(*args,**kwargs):raise AssertionError('Inference budget must not be requested')
    monkeypatch.setattr(cli,'auto_config',no_inference)
    cfg=config_for(Namespace(command=command,profile='balanced',device='cpu'))
    assert cfg.device == 'cpu'


@pytest.mark.parametrize('arguments', [
    ['run', 'absent', '--max-context', '0'],
    ['run', 'absent', '--cache-mib', '-1'],
    ['run', 'absent', '--workspace-mib', '1.5'],
    ['run', 'absent', '--max-new-tokens', '-1'],
    ['run', 'absent', '--token-ids', '1,,2'],
    ['run', 'absent', '--token-ids', '-1'],
    ['run', 'absent', '--token-ids', '2147483648'],
    ['run', 'absent', '--lossless-native-library', '/absent/library'],
    ['run', 'absent', '--lossless-dir', '/a', '--quantized-dir', '/b'],
    ['run', 'absent', '--backend', 'qwen-experimental', '--draft-model', '/draft'],
    ['serve', 'absent', '--token-ids', '1,2'],
    ['serve', 'absent', '--port', '65536'],
    ['pack', 'absent', '--output', '/absent', '--block-bytes', '3'],
    ['inspect', 'absent', '--tokens', '-1'],
])
def test_invalid_cli_options_fail_during_preflight(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_doctor_reports_missing_mlx_as_json_and_failing_status(monkeypatch,capsys):
    original_import = builtins.__import__
    def import_without_mlx(name,*args,**kwargs):
        if name == 'mlx.core':raise ImportError('test-only missing backend')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',import_without_mlx)
    assert main(['doctor']) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report['metal_available'] and not report['checkpoint_generation_tested']
    assert report['mlx'] is None and 'missing backend' in report['mlx_import_error']
    assert report['backends']['qwen-experimental']['experimental']


def test_cli_pack_creates_bounded_overlay_for_real_checkpoint(tiny_glm,tmp_path,capsys):
    from airllm.nextgen.lossless import LosslessOverlay
    from airllm.nextgen.storage import OffsetCheckpoint
    import numpy as np
    root,_ = tiny_glm
    destination = tmp_path/'overlay'
    key = 'model.embed_tokens.weight'
    main(['pack',str(root),'--output',str(destination),'--key',key,
          '--max-total-bytes',str(1024**2),'--block-bytes','128',
          '--cache-dir',str(tmp_path/'metadata')])
    report=json.loads(capsys.readouterr().out)
    assert report['base_checkpoint_required'] and report['base_checkpoint_unchanged']
    assert not report['performance_gain_certified']
    assert set(report['manifest']['entries']) == {key}
    with OffsetCheckpoint(root) as source:
        expected=source.read_array(key)
        packed=LosslessOverlay(source,destination)
        try:
            np.testing.assert_array_equal(packed.read_array(key),expected)
        finally:
            packed.close()


def test_cli_standalone_pack_prepares_without_original_checkpoint(tiny_glm,tmp_path,capsys):
    root,_=tiny_glm
    destination=tmp_path/'standalone'
    main(['pack',str(root),'--output',str(destination),'--standalone',
          '--max-total-bytes',str(10*1024**2),'--block-bytes','128',
          '--cache-dir',str(tmp_path/'metadata')])
    report=json.loads(capsys.readouterr().out)
    assert not report['base_checkpoint_required']
    assert report['format']=='lossless_original_weight_standalone'
    root.rename(tmp_path/'original-no-longer-at-source-path')
    main(['prepare',str(destination),'--cache-dir',str(tmp_path/'metadata')])
    prepared=json.loads(capsys.readouterr().out)
    assert prepared['prepared'] and not prepared['generation_ready']
    assert prepared['tensor_count']==len(report['manifest']['entries'])


def test_cli_llama_generates_reference_tokens(tmp_path,capsys):
    import ast
    import torch
    from conftest import save_tiny
    root=tmp_path/'llama'
    reference=save_tiny(root)
    with torch.inference_mode():
        expected=reference.generate(torch.tensor([[1,8,12]]),max_new_tokens=4,do_sample=False).tolist()[0]
    main(['run',str(root),'--backend','llama','--device','cpu','--float32',
          '--token-ids','1,8,12','--max-new-tokens','4','--cache-mib','1',
          '--io-mib','1','--kv-mib','1','--workspace-mib','1','--reserve-mib','0',
          '--cache-dir',str(tmp_path/'metadata')])
    assert ast.literal_eval(capsys.readouterr().out.strip())==expected
