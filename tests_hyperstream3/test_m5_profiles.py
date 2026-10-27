"""Frozen profile provenance and strict comparison controls, using CPU fixtures."""
from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import m5_validation as campaign
import plan_hyperstream3 as planner
import verify_m5_validation as checker


def prepared(tmp_path,monkeypatch,profile):
    h1,h2=tmp_path/'h1',tmp_path/'h2'
    h1.mkdir();h2.mkdir()
    model=tmp_path/'cpu-fixture.model';model.write_bytes(b'CPU plan fixture only')
    revisions={str(h1):'c8f81e57463ed3d7fd3d7dba226a65500439a9ce',
               str(h2):'5e8234181e736c0319a64df25e6813e92dd3e6da',str(ROOT):'1'*40}
    monkeypatch.setattr(campaign,'engine_manifest',lambda root:dict(git_head=revisions[str(root)],source_sha256={}))
    monkeypatch.setattr(campaign.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    output=tmp_path/'campaign'
    args=Namespace(pairs=2,expected_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   h1=h1,h2=h2,candidate=ROOT,output=output,model=model,kernel_report=None)
    if profile is not None:
        args.profile=profile
    result=campaign.prepare(args)
    config=json.loads((output/'config.json').read_text())
    return result,config


@pytest.mark.parametrize('profile',[None,'retained'])
def test_retained_default_preserves_original_24_experiments_96_jobs(tmp_path,monkeypatch,profile):
    result,config=prepared(tmp_path,monkeypatch,profile)
    assert config['profile']=='retained'
    assert result['experiment_count']==24 and result['jobs']==96
    assert config['experiments'][0]['name']=='h1_vs_h2'
    assert config['experiments'][1]['name']=='prefill16_vs16'
    assert not any('--router-reference-chunk' in job['args']
                   for trial in config['experiments'] for job in trial['plan'])
    first=config['experiments'][0]['plan']
    assert first[0]['engine_root']==config['engine_roots']['h1']
    assert first[1]['engine_root']==config['engine_roots']['h2']
    checker.validate_config(config)


def test_reference16_freezes_25_experiments_100_jobs_and_two_historical_h1_controls(tmp_path,monkeypatch):
    result,config=prepared(tmp_path,monkeypatch,'router_reference16')
    assert result['experiment_count']==25 and result['jobs']==100
    assert config['profile']=='router_reference16'
    assert [trial['name'] for trial in config['experiments'][:2]]==[
        'h1_vs_h2_reference16','general_long_h1_vs_h2_reference16']
    current=config['engine_roots']['candidate']
    historical=config['engine_roots']['h1']
    for ordinal,trial in enumerate(config['experiments']):
        assert [job['id'] for job in trial['plan']]==[
            'baseline-p00','candidate-p00','candidate-p01','baseline-p01']
        for job in trial['plan']:
            h1_control=ordinal<2 and job['id'].startswith('baseline')
            assert job['engine_root']==(historical if h1_control else current)
            if h1_control:
                assert '--router-reference-chunk' not in job['args']
            else:
                assert job['args'].count('--router-reference-chunk')==1
                assert job['args'][job['args'].index('--router-reference-chunk')+1]=='16'
                if job['id'].startswith('baseline'):
                    # The frozen worker's default is off. An explicit off flag
                    # would incorrectly request a nonexistent accelerator grid.
                    assert '--metal-batch-backend' not in job['args']
    long=config['experiments'][1]
    assert long['tasks_kind']=='general_long_noncopy'
    assert all(Path(job['tasks']).name=='general-long.json' for job in long['plan'])
    assert all(config['tasks_by_path'][job['tasks']][0]['max_new_tokens']==384 for job in long['plan'])
    assert '--hyper-prefill' not in long['plan'][0]['args']
    assert '--hyper-prefill' in long['plan'][1]['args']
    assert config['engine_manifests'][config['engine_roots']['h2']]['git_head']=='5e8234181e736c0319a64df25e6813e92dd3e6da'
    checker.validate_config(config)


def test_corrected_combination_requires_corrected_controls_only():
    trials=planner.build_experiments('/h1','/h2','/candidate',pairs=2,profile='router_reference16')
    combo=trials[-1]
    assert set(combo['requires_prior_pass'])=={
        'h1_vs_h2_reference16','tensorops_vs_h2_reference16',
        'grouped_transport_vs_h2_reference16','speculation_depth8_vs_h2_reference16'}
    assert set(combo['requires_prior_pass'])<={trial['name'] for trial in trials[:-1]}
    assert not any(trial['name'].endswith('_vs_h2') for trial in trials)


def test_corrected_controls_do_not_require_or_get_excluded_by_accelerator_grid():
    trials=planner.build_experiments('/h1','/h2','/candidate',pairs=2,profile='router_reference16')
    controls=trials[:2]
    assert [trial['name'] for trial in controls]==[
        'h1_vs_h2_reference16','general_long_h1_vs_h2_reference16']
    # Exercise the real campaign gate with no report at all. These controls
    # must launch without claiming a nonexistent mode=off grid qualification.
    for trial in trials:
        if not any('--metal-batch-backend' in job['args'] for job in trial['plan']):
            assert campaign.require_kernels({},trial) is None
    for trial in controls:
        assert not any('--metal-batch-backend' in job['args'] for job in trial['plan'])
        assert campaign.require_kernels({},trial) is None


@pytest.mark.parametrize('value',['0','16'])
def test_schema_admits_declared_router_values_without_precision_changes(tmp_path,monkeypatch,value):
    _,config=prepared(tmp_path,monkeypatch,'router_reference16')
    flags=config['experiments'][0]['plan'][1]['args']
    flags[flags.index('--router-reference-chunk')+1]=value
    checker.validate_config(config)
    assert config['numerical_tolerances']==dict(atol=1e-4,rtol=1e-4)


def test_schema_refuses_unsupported_router_value(tmp_path,monkeypatch):
    _,config=prepared(tmp_path,monkeypatch,'router_reference16')
    flags=config['experiments'][0]['plan'][1]['args']
    flags[flags.index('--router-reference-chunk')+1]='17'
    with pytest.raises(checker.RejectedEvidence,match='router reference chunk'):
        checker.validate_config(config)


def test_reference16_metadata_does_not_accept_an_ordered_expert_permutation(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('m5_profile_fabricated_recordings',
        ROOT/'tests_hyperstream3/test_m5_fidelity_outcomes.py')
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    config,trial,jobs=fixtures.experiment_fixture(tmp_path,monkeypatch)
    config['profile']='router_reference16'
    for job in trial['plan']:
        job['args']=['--numerical-evidence','--router-reference-chunk','16']
    path=tmp_path/'p0/candidate.json'
    report=json.loads(path.read_text())
    routes=report['turns'][0]['step_checks'][0]['routing_by_layer_position']['0']['0']
    before=list(routes)
    routes[:]=reversed(routes)
    assert routes!=before and set(routes)==set(before)
    campaign.save(path,report)
    with pytest.raises(checker.VerifiedFidelityRejection) as caught:
        checker.summarize_experiment(tmp_path,config,trial,jobs)
    assert caught.value.result['status']=='rejected_fidelity'
    assert caught.value.result['rejected_pairs'][0]['category']=='verified_ordered_expert_routes_mismatch'
    assert not caught.value.result['performance_comparison_accepted']


def test_invalid_profile_is_rejected_before_preparation():
    with pytest.raises(ValueError,match='profile'):
        planner.build_experiments('/h1','/h2','/candidate',pairs=2,profile='unfrozen')
