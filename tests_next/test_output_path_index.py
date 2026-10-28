"""Read-only preflight aliases and bounded operation counts; no workers/GPU."""
import errno
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.experiment_guard import validate_output_paths


def legacy_reference(output,protected=(),extra_outputs=()):
    """Retained pairwise rule, exercised only on tiny filesystem fixtures."""
    report=Path(output)
    paths=[report,report.with_suffix('.log'),report.with_suffix('.guard.json'),*map(Path,extra_outputs)]
    def aliases(a,b):
        if a.resolve()==b.resolve():
            return True
        return a.exists() and b.exists() and a.samefile(b)
    for index,path in enumerate(paths):
        if any(aliases(path,previous) for previous in paths[:index]):
            raise ValueError('Experiment report/log/guard outputs must be distinct files')
        if any(aliases(path,Path(source)) for source in protected):
            raise ValueError('Experiment output aliases a protected input file')
    return paths


@pytest.mark.parametrize('alias',['none','direct','symlink','hardlink','dangling','duplicate-extra','parent-symlink','suffix'])
def test_small_stable_filesystem_decisions_match_retained_rule(tmp_path,alias):
    model=tmp_path/'model.gguf';model.write_bytes(b'protected CPU fixture')
    output=tmp_path/'report.json'
    protected=[model]
    extras=[]
    if alias=='direct':
        output=model
    elif alias=='symlink':
        output.symlink_to(model)
    elif alias=='hardlink':
        output.hardlink_to(model)
    elif alias=='dangling':
        missing=tmp_path/'future-input'
        output.symlink_to(missing);protected=[missing]
    elif alias=='duplicate-extra':
        extras=[tmp_path/'future.f32']*2
    elif alias=='parent-symlink':
        real=tmp_path/'real';real.mkdir()
        link=tmp_path/'linked';link.symlink_to(real,target_is_directory=True)
        extras=[real/'future.f32',link/'future.f32']
    elif alias=='suffix':
        output=tmp_path/'report.log'
    try:
        expected=legacy_reference(output,protected,extras)
    except ValueError as error:
        with pytest.raises(ValueError,match=str(error)):
            validate_output_paths(output,protected,extras)
    else:
        assert validate_output_paths(output,protected,extras)==expected
    assert model.read_bytes()==b'protected CPU fixture'


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_existing_extra_outputs_cannot_alias_one_another(tmp_path,kind):
    first=tmp_path/'first.f32';first.write_bytes(b'raw previous artifact')
    second=tmp_path/'second.f32'
    if kind=='symlink':
        second.symlink_to(first)
    else:
        second.hardlink_to(first)
    with pytest.raises(ValueError,match='distinct files'):
        validate_output_paths(tmp_path/'report.json',extra_outputs=[first,second])
    assert first.read_bytes()==b'raw previous artifact'


def test_two_dangling_symlinks_to_the_same_future_target_collide(tmp_path):
    target=tmp_path/'absent'
    first=tmp_path/'a.f32';second=tmp_path/'b.f32'
    first.symlink_to(target);second.symlink_to(target)
    with pytest.raises(ValueError,match='distinct files'):
        validate_output_paths(tmp_path/'report.json',extra_outputs=[first,second])
    assert not target.exists() and first.is_symlink() and second.is_symlink()


def test_protected_iterator_is_fully_indexed_once_for_every_destination(tmp_path):
    output=tmp_path/'report.json'
    guard=output.with_suffix('.guard.json');guard.write_bytes(b'protected input')
    protected=(value for value in [tmp_path/'unrelated',guard])
    with pytest.raises(ValueError,match='protected input'):
        validate_output_paths(output,protected=protected)
    assert guard.read_bytes()==b'protected input'


@pytest.mark.parametrize('method',['resolve','stat','lstat'])
def test_permission_failures_are_not_treated_as_missing(tmp_path,monkeypatch,method):
    output=tmp_path/'report.json'
    original=getattr(Path,method)
    def denied(path,*args,**kwargs):
        if path==output:
            raise PermissionError(errno.EACCES,'injected preflight denial')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,method,denied)
    with pytest.raises(PermissionError,match='preflight denial'):
        validate_output_paths(output)
    assert list(tmp_path.iterdir())==[]


def test_permission_failure_during_second_observation_is_fatal(tmp_path,monkeypatch):
    output=tmp_path/'report.json';original=Path.resolve;seen=[]
    def denied(path,*args,**kwargs):
        if path==output:
            seen.append(path)
            if len(seen)==2:
                raise PermissionError(errno.EACCES,'injected revalidation denial')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',denied)
    with pytest.raises(PermissionError,match='revalidation denial'):
        validate_output_paths(output)
    assert len(seen)==2


@pytest.mark.parametrize('error',[OSError(errno.EIO,'injected filesystem error'),RuntimeError('injected programming failure')])
def test_unexpected_errors_propagate_without_a_fallback(tmp_path,monkeypatch,error):
    output=tmp_path/'report.json';original=Path.lstat
    def fail(path,*args,**kwargs):
        if path==output:
            raise error
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',fail)
    with pytest.raises(type(error),match='injected'):
        validate_output_paths(output)


@pytest.mark.parametrize('change',['replace-inode','create-absent','delete-existing','change-metadata','replace-symlink-same-target','retarget-symlink'])
def test_observed_path_or_link_drift_is_refused(tmp_path,monkeypatch,change):
    output=tmp_path/'report.json'
    target=tmp_path/'target';target.write_bytes(b'input stays unchanged')
    replacement=tmp_path/'replacement'
    if change in ('replace-symlink-same-target','retarget-symlink'):
        output.symlink_to(target)
        destination=target
        if change=='retarget-symlink':
            destination=tmp_path/'same-inode-other-name';destination.hardlink_to(target)
        replacement.symlink_to(destination)
    elif change!='create-absent':
        output.write_bytes(b'old output')
        replacement.write_bytes(b'new inode')
    original=Path.resolve;seen=[]
    def mutate(path,*args,**kwargs):
        if path==output:
            seen.append(path)
            if len(seen)==2:
                if change in ('replace-inode','replace-symlink-same-target','retarget-symlink'):
                    replacement.replace(output)
                elif change=='create-absent':
                    output.write_bytes(b'appeared')
                elif change=='delete-existing':
                    output.unlink()
                else:
                    output.write_bytes(b'changed contents on same inode')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',mutate)
    with pytest.raises(RuntimeError,match='path changed during output validation'):
        validate_output_paths(output)
    assert target.read_bytes()==b'input stays unchanged'


def test_symlink_loop_is_refused(tmp_path):
    first=tmp_path/'a';second=tmp_path/'b'
    first.symlink_to(second);second.symlink_to(first)
    with pytest.raises((OSError,RuntimeError)):
        validate_output_paths(first)


@pytest.mark.parametrize('extra_count',[0,17,1537])
def test_large_synthetic_sets_use_two_linear_observations_without_samefile(tmp_path,monkeypatch,extra_count):
    """Count operations, not elapsed time; no legacy1537 run or disk accesses."""
    protected=[tmp_path/f'input-{i}.py' for i in range(31)]
    extras=[tmp_path/f'output-{i:04d}.f32' for i in range(extra_count)]
    count=extra_count+3+len(protected)
    calls=dict(resolve=0,stat=0,lstat=0)
    def budget(name):
        calls[name]+=1
        assert calls[name]<=2*count,'Preflight exceeded its linear observation budget'
    def resolve(path,*args,**kwargs):
        budget('resolve');return path
    def stat(path,*args,**kwargs):
        budget('stat');raise FileNotFoundError(errno.ENOENT,'synthetic absent target')
    def lstat(path,*args,**kwargs):
        budget('lstat');raise FileNotFoundError(errno.ENOENT,'synthetic absent link')
    monkeypatch.setattr(Path,'resolve',resolve)
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(Path,'lstat',lstat)
    monkeypatch.setattr(Path,'samefile',lambda *args:pytest.fail('Pairwise samefile lookup returned'))
    result=validate_output_paths(tmp_path/'report.json',protected=protected,extra_outputs=extras)
    assert len(result)==extra_count+3
    assert calls==dict(resolve=2*count,stat=2*count,lstat=2*count)
