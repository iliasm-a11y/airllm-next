import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from research.glm53_gpu_v2.profile_summary import summarize


def operation(step=1,op='MUL_MAT_ID',status='ok'):
    page={'status':status,'reason':'' if status=='ok' else 'mincore_failed',
          'logical_selected_weight_bytes':100,'inspection_seconds':.02,'page_size':16384,
          'inspected_pages':10 if status=='ok' else None,'resident_pages':6 if status=='ok' else None,
          'nonresident_pages':4 if status=='ok' else None}
    return {'name':'ffn_moe_gate-40','step':step,'op':op,'seconds':.5,'cpu_seconds':2.0,
            'os_read_bytes':1234,'pageins':3,'weight_pages_before':page}


def test_aggregation_scope_and_errors():
    report=summarize([operation(),operation(step=0,status='error')])
    total=report['totals']
    assert total['operations']==2 and total['op_wall_seconds']==1
    assert total['inspection_seconds_excluded']==.04
    assert total['cpu_seconds_per_op_wall_second']==4
    assert total['resident_fraction_of_successful_snapshots']==.6
    assert total['logical_selected_weight_bytes']==200
    assert total['residency_operations_error']==1
    assert report['by_layer']['40']['operations']==2
    assert report['by_phase']['decode']['operations']==1
    assert report['by_op_layer_step']['MUL_MAT_ID/layer=40/step=0']['operations']==1


def test_unknown_does_not_become_zero():
    row=operation(status='error');del row['cpu_seconds'];del row['os_read_bytes'];del row['pageins']
    total=summarize([row])['totals']
    assert total['cpu_seconds'] is None and total['os_read_bytes'] is None
    assert total['resident_pages_snapshot_sum'] is None
    assert total['resident_fraction_of_successful_snapshots'] is None
    row['weight_pages_before']['resident_pages']=0
    with pytest.raises(ValueError):summarize([row])


def test_invalid_metrics_and_inconsistent_pages():
    row=operation();row['seconds']=-1
    with pytest.raises(ValueError):summarize([row])
    row=operation();row['weight_pages_before']['resident_pages']=20
    with pytest.raises(ValueError):summarize([row])


def test_mincore_without_reading_protected_weight_pages(tmp_path):
    if sys.platform!='darwin' or not shutil.which('clang++'):pytest.skip('Darwin compiler fixture')
    source=Path(__file__).with_name('native_profile.cpp')
    binary=tmp_path/'pages-fixture'
    subprocess.run(['clang++','-std=c++17','-O2','-DAIRLLM_PROFILE_RESIDENCY_TEST',str(source),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    success,bounded,unavailable=[json.loads(line) for line in result.stdout.splitlines()]
    assert success['status']=='ok' and success['inspected_pages']==2
    assert success['resident_pages']+success['nonresident_pages']==2
    assert bounded['status']=='error' and bounded['resident_pages'] is None
    assert unavailable['status']=='error' and unavailable['errno']>0
    assert unavailable['resident_pages'] is None


def test_real_cpu_mapped_tensor_selected_ids_no_weight_reads(tmp_path):
    if sys.platform!='darwin' or not shutil.which('clang++'):pytest.skip('Darwin native tensor fixture')
    source=Path(__file__).with_name('native_profile.cpp')
    work=Path(__file__).resolve().parents[3]
    runtime=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_SOURCE',work/'llama-v2'))
    libraries=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_LIBRARIES',work/'runtime-v2-build'/'bin'))
    if not (runtime/'include'/'llama.h').exists() or not (libraries/'libllama.dylib').exists():
        pytest.skip('pinned native runtime source/libraries required')
    binary=tmp_path/'tensor-pages-fixture'
    subprocess.run(['clang++','-std=c++17','-O2','-DAIRLLM_PROFILE_TENSOR_TEST',
        '-I'+str(runtime/'include'),'-I'+str(runtime/'ggml'/'include'),str(source),
        '-L'+str(libraries),'-lllama','-lggml','-lggml-base','-lggml-cpu',
        '-Wl,-rpath,'+str(libraries),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    selected,dense,invalid=[json.loads(line) for line in result.stdout.splitlines()]
    assert selected['status']=='ok' and selected['selected_expert_count']==2
    assert selected['logical_selected_weight_bytes']==43008
    assert dense['logical_selected_weight_bytes']==86016 and dense['status']=='ok'
    assert selected['inspected_pages']<dense['inspected_pages']
    assert selected['weight_buffer_type']=='CPU_Mapped'
    assert selected['weight_device']=='unregistered' and selected['weight_buffer_usage']==1
    assert invalid['status']=='error' and invalid['resident_pages'] is None
