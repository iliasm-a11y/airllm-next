import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from .components import validate_shape
from .model_profile import resource_blockers,REQUIRED_DISK,REQUIRED_MEMORY,run
from .profile_analysis import analyze_capture,conditional_bound
from .scheduler import Group,select

@pytest.mark.parametrize('shape',[(8,0,4096,1),(2,1,255,1),(8,True,4096,1),
    (3,16,256,1),(8,16,256,3),(2,8192,16384,2)])
def test_component_rejects_unbounded_or_malformed_payload(shape):
    with pytest.raises((ValueError,MemoryError)):validate_shape(*shape)

def test_admission_keeps_all_independent_failures():
    assert set(resource_blockers({'pressure_level':2,'available_bytes':0},0,{'passed':False}))=={
        'memory_pressure_not_normal','available_memory_below_5.5GiB','disk_reserve_below_32GiB','power_lid_or_clock_admission'}
    assert resource_blockers({'pressure_level':1,'available_bytes':REQUIRED_MEMORY},REQUIRED_DISK,{'passed':True})==[]

def test_refusal_never_launches_worker(tmp_path,monkeypatch):
    import research.glm53_ultra_decode.model_profile as m
    model=tmp_path/'model';model.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    probe=tmp_path/'probe';probe.write_bytes(b'not executed')
    monkeypatch.setattr(m,'system_snapshot',lambda:{'pressure_level':1,'available_bytes':REQUIRED_MEMORY-1})
    monkeypatch.setattr(m,'inspect_native_checkpoint',lambda *a:{'integrity_verified':True,'architecture_matches':True})
    monkeypatch.setattr(m,'PowerGate',lambda **kw:SimpleNamespace(sample=lambda:{'passed':True}))
    monkeypatch.setattr(m,'supervise',lambda *a,**kw:pytest.fail('model launched after failed admission'))
    args=SimpleNamespace(worker=False,output=tmp_path/'out',model=model,runtime=runtime,probe=probe,
                         receipt=tmp_path/'receipt',tokens=32,diagnostic=True,prompt='test')
    assert run(args)==2
    report=json.loads((args.output/'report.json').read_text())
    assert not report['native_process_launched'] and not report['performance_measured']
    assert report['decode_tokens_per_second'] is None

def capture(tmp_path,diagnostic=False):
    (tmp_path/'complete.json').write_text(json.dumps({'completed':True,'captured_steps':3}))
    tokens=[{'step':0,'token':10,'eog':False},{'step':1,'token':11,'eog':False},{'step':2,'token':12,'eog':True}]
    (tmp_path/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tokens))
    rows=[{'phase':'prefill','step':0,'wall_seconds':10},
          {'phase':'decode','step':1,'decode_call_seconds':2,'os_read_bytes':100},
          {'phase':'decode','step':2,'decode_call_seconds':3,'os_read_bytes':200}]
    (tmp_path/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows

def test_count_decode_intervals_excludes_prefill_and_first_output(tmp_path):
    capture(tmp_path);r=analyze_capture(tmp_path,diagnostic=False)
    assert r['decode_intervals']==2 and r['output_ids']==3
    assert r['decode_call_seconds']==5 and r['decode_tokens_per_second']==.4
    assert r['os_read_bytes']==300 and r['physical_nand_bytes'] is None and r['natural_eog']

def test_serialized_diagnostic_never_reports_model_throughput(tmp_path):
    capture(tmp_path);r=analyze_capture(tmp_path,diagnostic=True)
    assert r['decode_tokens_per_second'] is None
    assert r['diagnostic_decode_intervals_per_second']==.4

@pytest.mark.parametrize('mutation',['missing','duplicate','nan','negative'])
def test_incomplete_or_corrupt_timings_rejected(tmp_path,mutation):
    rows=capture(tmp_path)
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[-1]['step']=1
    elif mutation=='nan':rows[-1]['decode_call_seconds']=float('nan')
    else:rows[-1]['decode_call_seconds']=-1
    (tmp_path/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):analyze_capture(tmp_path,diagnostic=False)

def test_physical_bound_requires_explicit_coverage_and_overlap():
    r=conditional_bound(12e9,4e9,2e9,1,True)
    assert r['minimum_token_seconds']==4 and r['conditional_upper_tokens_per_second']==.25
    assert conditional_bound(12e9,4e9,2e9,1,False)['minimum_token_seconds']==5
    with pytest.raises(ValueError):conditional_bound(1,2,1,0,True)
    with pytest.raises(ValueError):conditional_bound(1,0,0,0,True)

def evidence():
    group=Group('Q8_0',1536,4096)
    e={'scope':'whole_model_decode','runtime_sha256':'a'*64,'group':group.__dict__,
       'full_logits_passed':True,'ordered_routes_passed':True,'recurrent_states_passed':True,
       'original_thresholds_passed':True,'guard_completed':True,'weights_unchanged':True,
       'captured_steps':32,'all_boundary_costs_included':True,
       'abba_cycles':[{'order':['A','B','B','A'],'total_seconds':[12,10,10,12]}]*2}
    args={'enabled':True,'runtime_sha256':'a'*64,'resident':True,'available_bytes':4<<30,'pressure_level':1}
    return group,e,args

def test_scheduler_disabled_even_for_good_evidence():
    group,e,args=evidence();args['enabled']=False
    assert select(group,e,**args)['backend']=='cpu'

@pytest.mark.parametrize('key',['full_logits_passed','ordered_routes_passed','recurrent_states_passed',
                              'original_thresholds_passed','guard_completed','weights_unchanged'])
def test_scheduler_requires_all_fidelity_and_guard_gates(key):
    group,e,args=evidence();e[key]=False
    assert select(group,e,**args)['backend']=='cpu'

@pytest.mark.parametrize('mutation',['component','runtime','few_steps','one_cycle','slow','costs','wrong_group','swap','pressure','residency'])
def test_scheduler_refuses_unqualified_or_costly_gpu(mutation):
    group,e,args=evidence();e=copy.deepcopy(e)
    if mutation=='component':e['scope']='component'
    elif mutation=='runtime':args['runtime_sha256']='b'*64
    elif mutation=='few_steps':e['captured_steps']=4
    elif mutation=='one_cycle':e['abba_cycles']=e['abba_cycles'][:1]
    elif mutation=='slow':e['abba_cycles'][0]['total_seconds']=[12,13,13,12]
    elif mutation=='costs':e['all_boundary_costs_included']=False
    elif mutation=='wrong_group':e['group']['width']=8192
    elif mutation=='swap':args['swap_growth_bytes']=(32<<20)+1
    elif mutation=='pressure':args['pressure_level']=2
    else:args['resident']=False
    assert select(group,e,**args)['backend']=='cpu'

def test_experimental_policy_selects_only_repeated_faithful_net_gain():
    group,e,args=evidence();r=select(group,e,**args)
    assert r['backend']=='gpu' and r['median_ratio']==1.2
