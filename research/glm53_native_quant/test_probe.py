from argparse import Namespace
import json
from pathlib import Path
import pytest
from . import probe
from research.glm53_ultra_decode.build import sha


@pytest.mark.parametrize('format,rows,width,error',[
    (4,2048,4096,ValueError),(2,7,4096,ValueError),(8,1537,4096,ValueError),
    (2,2048,4095,ValueError),(8,8192,16384,MemoryError),
    (2,8192,16384,MemoryError),(2,True,4096,ValueError),
])
def test_selection_refuses_unbounded_or_wrong_native_layout(format,rows,width,error):
    with pytest.raises(error):probe.validate_selection(format,rows,width)


def test_real_component_shapes_fit_packed_budget():
    assert probe.validate_selection(2,2048,4096)==2752512
    assert probe.validate_selection(8,1536,4096)==6684672


def fixture_build(tmp_path):
    build=tmp_path/'build';build.mkdir()
    runtime=tmp_path/'runtime';runtime.mkdir()
    source=tmp_path/'source.cpp';source.write_text('source')
    (build/'native.dylib').write_bytes(b'binary')
    (runtime/'fixture.dylib').write_bytes(b'library')
    record={'passed':True,'inputs_sha256':{str(source):sha(source)},
        'binaries_sha256':{'native.dylib':sha(build/'native.dylib')},
        'native_libraries_sha256':{'fixture.dylib':sha(runtime/'fixture.dylib')}}
    (build/'build.json').write_text(json.dumps(record))
    return build,runtime,source


@pytest.mark.parametrize('changed',['source','binary','library','additional_library'])
def test_changed_build_or_runtime_cannot_inherit_evidence(tmp_path,changed):
    build,runtime,source=fixture_build(tmp_path)
    assert probe.verify_build(build,runtime)==sha(build/'build.json')
    target={'source':source,'binary':build/'native.dylib','library':runtime/'fixture.dylib',
        'additional_library':runtime/'extra.dylib'}[changed]
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):probe.verify_build(build,runtime)


def test_busy_slot_is_admission_refusal_before_arrays_or_device(tmp_path,monkeypatch):
    sample={'available_bytes':4<<30,'pressure_level':1,'swap_used_bytes':100,'physical_bytes':16<<30}
    monkeypatch.setattr(probe,'system_snapshot',lambda:sample)
    class Gate:
        def __init__(self,**kwargs):pass
        def sample(self):return {'passed':True}
    monkeypatch.setattr(probe,'PowerGate',Gate)
    monkeypatch.setattr(probe.shutil,'disk_usage',lambda p:Namespace(free=40<<30))
    monkeypatch.setattr(probe,'inspect_native_checkpoint',lambda *a:{'integrity_verified':True,'architecture_matches':True})
    def busy():raise RuntimeError('ENGINE_BUSY: retained lease')
    monkeypatch.setattr(probe,'acquire_uid_lock',busy)
    args=Namespace(output=tmp_path,model=tmp_path/'model',receipt=tmp_path/'receipt',campaign_swap_used_bytes=100)
    result=probe.preflight(args)
    assert result['blockers']==['ENGINE_BUSY: retained lease']


def test_failed_worker_readmission_writes_refusal_without_loading_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'preflight',lambda args:{'blockers':['engine busy']})
    assert probe.worker(Namespace(output=tmp_path))==2
    assert json.loads((tmp_path/'result.json').read_text())['status']=='worker_admission_refused'


def test_slot_acquired_elsewhere_after_observation_still_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'preflight',lambda args:{'blockers':[]})
    def busy():raise RuntimeError('ENGINE_BUSY: raced lease')
    monkeypatch.setattr(probe,'acquire_uid_lock',busy)
    assert probe.worker(Namespace(output=tmp_path))==2
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='worker_admission_refused'
    assert 'ENGINE_BUSY' in result['error']


def test_nonfinite_outputs_fail_the_existing_component_criteria():
    import numpy as np
    from research.glm53_batched_prefill.metal_probe import numerical_comparison
    actual=np.array([float('nan'),1],np.float32)
    expected=np.array([0,1],np.float32)
    result=numerical_comparison(actual,expected)
    assert not result['bit_identical']
    assert not result['metal_tolerance']['passed']
