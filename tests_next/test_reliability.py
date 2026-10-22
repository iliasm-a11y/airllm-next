import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.tuning import auto_config,measure_storage
from test_native import options


def test_long_session_matches_recompute(tmp_path):
    from test_fp8_model import save_glm64
    root=tmp_path/'long';ref=save_glm64(root);prefix=[1,8,9];cache=None
    with AirLLMNextModel(root,next_config=options(tmp_path,max_context=128,cache_bytes=12000)) as model,torch.inference_mode():
        # Predetermined tokens avoid amplifying legal boundary top-k ties into a
        # different greedy trajectory; all logits are still checked each step.
        for token in range(16):
            ids=np.array([prefix if cache is None else [prefix[-1]]],np.int32)
            result=model(ids,past_key_values=cache);cache=result.past_key_values
            expected=ref(torch.tensor([prefix])).logits[:,-1:]
            np.testing.assert_allclose(np.array(result.logits[:,-1:]),expected.numpy(),atol=8e-6,rtol=3e-4)
            prefix.append(10+token)
        assert cache.position==len(prefix)-1
        assert model.reader.cache.bytes<=12000


def test_interrupted_forward_can_retry_committed_prefix(tiny_glm,tmp_path,monkeypatch):
    root,_=tiny_glm
    with AirLLMNextModel(root,next_config=options(tmp_path)) as model:
        cache=model([[1,8]]).past_key_values
        original=model.mlps[1]
        class Broken:
            def __call__(self,x):raise OSError('external SSD disconnected')
        model.mlps[1]=Broken()
        with pytest.raises(OSError):model([[9]],past_key_values=cache)
        assert cache.position==2
        model.mlps[1]=original
        result=model([[9]],past_key_values=cache)
        expected=model([[1,8,9]])
        np.testing.assert_allclose(np.array(result.logits),np.array(expected.logits[:,-1:]),atol=4e-6,rtol=2e-4)


def test_tuning_persists_real_read_measurements(tiny_glm,tmp_path):
    root,_=tiny_glm;cfg=options(tmp_path)
    with OffsetCheckpoint(root,cfg.metadata_dir()) as cp:
        result=measure_storage(cp,cfg,samples=3)
        assert result['selected_io_workers'] in (1,2,4)
        assert all(x['logical_bytes']>0 for x in result['candidates'])
        assert (cfg.metadata_dir()/(cp.fingerprint+'.tuning.json')).is_file()


def test_memory_reserve_and_safe_kv_failure(tiny_glm,tmp_path):
    root,_=tiny_glm
    with AirLLMNextModel(root,next_config=options(tmp_path,kv_bytes=100)) as model:
        with pytest.raises(MemoryError,match='KV budget'):model([[1]])


def test_native_hardware_validation(tiny_glm,tmp_path):
    import mlx.core as mx
    if not mx.metal.is_available():pytest.skip('Apple Metal hardware unavailable; CPU MLX tests are separate')
    root,ref=tiny_glm
    with AirLLMNextModel(root,next_config=options(tmp_path,device='gpu')) as model,torch.inference_mode():
        np.testing.assert_allclose(np.array(model([[1,8,9]]).logits),ref(torch.tensor([[1,8,9]])).logits.numpy(),atol=2e-5,rtol=3e-4)


def test_lightweight_import_does_not_initialize_torch_or_transformers():
    import subprocess,sys
    script='import sys; import airllm; from airllm import AutoModel; import airllm.nextgen.cli; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'air_llm')+os.pathsep+os.environ.get('PYTHONPATH',''))
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr


def test_runtime_prefetch_calibration_is_reused(tiny_glm,tmp_path):
    from airllm.nextgen.tuning import calibrate_prefetch
    root,_=tiny_glm;cfg=options(tmp_path)
    with AirLLMNextModel(root,next_config=cfg) as model:
        report=calibrate_prefetch(model,[[1,8,9]],tokens=3)
        assert len(report['measurements'])==4
        assert all(x['generation_seconds']>0 for x in report['measurements'])
    with AirLLMNextModel(root,next_config=cfg) as model:
        assert model.reader.prefetch_enabled==report['prefetch_enabled']
