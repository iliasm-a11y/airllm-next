"""Portable controls for targeted captures; these tests never import MLX."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import diagnose_m5_router as diagnostic


def test_profile_reads_actual_intermediates_and_preserves_return_objects():
    original = np.array([[3., 2., 1.]], dtype=np.float32)

    def router(x):
        logits = x * 2
        selected = np.array([[0, 1]])
        scores = logits[:, :2] / logits[:, :2].sum()
        return selected, scores

    result, locals_ = diagnostic._call_with_locals(router, (original,), {})
    assert result[0] is locals_['selected']
    assert result[1] is locals_['scores']
    assert locals_['x'] is original
    assert np.array_equal(locals_['logits'], original * 2)
    assert sys.getprofile() is None


def test_profile_is_restored_when_original_router_fails():
    def router():
        raise FloatingPointError('original arithmetic failure')
    with pytest.raises(FloatingPointError, match='original arithmetic failure'):
        diagnostic._call_with_locals(router, (), {})
    assert sys.getprofile() is None


def test_existing_profiler_is_preserved_instead_of_silently_replaced():
    def installed(frame,event,argument):
        pass
    sys.setprofile(installed)
    try:
        with pytest.raises(RuntimeError, match='unused Python profile hook'):
            diagnostic._call_with_locals(lambda: None, (), {})
        assert sys.getprofile() is installed
    finally:
        sys.setprofile(None)


def synthetic_capture(folder, ids=(0,1), *, input_=None, weight=None):
    folder.mkdir()
    input_ = np.array([[1.,2.]], dtype=np.float32) if input_ is None else input_
    weight = np.array([[2.,1.],[1.,2.],[0.,0.]],dtype=np.float32) if weight is None else weight
    logits = input_ @ weight.T
    p = np.exp(logits-logits.max(axis=-1,keepdims=True))
    p /= p.sum(axis=-1,keepdims=True)
    arrays = dict(routing_input=input_,weight=weight,logits=logits,probabilities=p)
    record = dict(status='target_captured_stopped_before_experts',
        integrity_checked_before_after=True, checkpoint_sha256_before='a'*64,
        checkpoint_sha256_after='a'*64, full_input_ids=[3],
        target_request=dict(layer=30,absolute_position=0),
        target=dict(row=0,selected_ids=list(ids),top_k_boundary_tie=False),
        arrays={k:diagnostic._artifact(folder,k,v) for k,v in arrays.items()})
    path = folder/'capture.json'
    path.write_text(json.dumps(record))
    return path


def test_analysis_keeps_ordered_rejection_with_equal_expert_sets(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    b = synthetic_capture(tmp_path/'b', ids=(1,0))
    result = diagnostic.pair_analysis(a,b)
    assert result['strict_ordered_route_pass'] is False
    assert result['selected_set_equal'] is True
    assert result['activation_bitwise_equal'] is True
    assert result['maximum_activation_absolute_difference'] == 0.
    assert [row['expert'] for row in result['affected_experts']] == [0,1]
    assert result['performance_gain_qualified'] is False
    assert result['complete_generation'] is False


def test_analysis_reports_upstream_activation_changes_without_inventing_cause(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    x = np.array([[1.,np.nextafter(np.float32(2.),np.float32(3.))]],dtype=np.float32)
    b = synthetic_capture(tmp_path/'b',input_=x)
    result = diagnostic.pair_analysis(a,b)
    assert not result['activation_bitwise_equal']
    assert result['maximum_activation_absolute_difference'] > 0
    assert 'does not establish its cause' in result['cause']


def test_changed_raw_artifact_is_rejected(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    b = synthetic_capture(tmp_path/'b')
    (b.parent/'logits.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        diagnostic.pair_analysis(a,b)


def test_different_router_weights_are_not_called_rounding(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    b = synthetic_capture(tmp_path/'b',weight=np.ones((3,2),dtype=np.float32))
    with pytest.raises(ValueError,match='weights differ'):
        diagnostic.pair_analysis(a,b)


def test_signed_zero_weight_change_is_not_bitwise_equal(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    weight = np.array([[2.,1.],[1.,2.],[-0.,0.]],dtype=np.float32)
    b = synthetic_capture(tmp_path/'b',weight=weight)
    with pytest.raises(ValueError,match='weights differ'):
        diagnostic.pair_analysis(a,b)


def test_mismatched_checkpoint_after_hash_is_rejected(tmp_path):
    a = synthetic_capture(tmp_path/'a')
    b = synthetic_capture(tmp_path/'b')
    record = json.loads(b.read_text())
    record['checkpoint_sha256_after'] = 'b'*64
    b.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='integrity-checked'):
        diagnostic.pair_analysis(a,b)


@pytest.mark.parametrize('name',['../outside.npy','/tmp/outside.npy'])
def test_artifact_path_cannot_escape_capture_folder(tmp_path,name):
    with pytest.raises(ValueError,match='direct relative file'):
        diagnostic._load_artifact(tmp_path,dict(file=name))


def test_nonfinite_capture_is_not_saved_as_good_evidence(tmp_path):
    with pytest.raises(FloatingPointError,match='Non-finite'):
        diagnostic._artifact(tmp_path,'bad',np.array([np.nan],dtype=np.float32))
