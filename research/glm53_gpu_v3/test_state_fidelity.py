import json
import struct
from pathlib import Path
import pytest
from research.glm53_gpu_v3.state_fidelity import compare_states, compare_reset


def event(step=0, digest='a'*64):
    return dict(step=step,target='cache_s_l0',op='CPY',type=0,bytes=4,shape=[1,1,1,1],status='hashed',finite=True,sha256=digest)


def write(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def reset_capture(root, per_request=2):
    (root/'complete.json').write_text(json.dumps(dict(completed=True,captured_steps=2*per_request)))
    tokens=[dict(step=i,request=i//per_request,token=1,next_input_token=1,vocab=3)
            for i in range(2*per_request)]
    write(root/'tokens.jsonl',tokens)
    write(root/'state_hashes.jsonl',[event(i) for i in range(2*per_request)])
    for i in range(2*per_request):
        (root/f'logits-{i}.f32').write_bytes(struct.pack('<3f',0.,2.+i%per_request,1.))
    return tokens


def test_state_order_finite_hash_and_missing_coverage_fail_closed(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';write(a,[event(),event(1)]);write(b,[event(),event(1)])
    assert compare_states(a,b)['passed']
    for rows in [[event(1),event()], [event()], [{**event(),'finite':False}], [event(digest='b'*64),event(1)], [{**event(),'status':'skipped'}]]:
        write(b,rows);assert not compare_states(a,b)['passed']
    b.unlink();assert not compare_states(a,b)['passed']


def test_reset_requires_completed_capture_full_logits_and_states(tmp_path):
    reset_capture(tmp_path)
    report=compare_reset(tmp_path,2)
    assert report['passed'] and report['finite_logit_vectors_validated']==4
    assert report['request_token_counts']==[2,2] and report['expected_logit_bytes_per_vector']==12
    (tmp_path/'logits-2.f32').write_bytes(struct.pack('<3f',0.,2.,1.5))
    assert not compare_reset(tmp_path,2)['passed']
    (tmp_path/'complete.json').unlink();assert not compare_reset(tmp_path,2)['passed']


@pytest.mark.parametrize('raw',[b'',b'\0'*8,struct.pack('<3f',float('nan'),2.,1.),
                               struct.pack('<3f',0.,float('inf'),1.),struct.pack('<4f',0.,2.,1.,0.)])
def test_equal_empty_truncated_nonfinite_or_oversized_logits_never_pass(tmp_path,raw):
    reset_capture(tmp_path,1)
    for i in (0,1):(tmp_path/f'logits-{i}.f32').write_bytes(raw)
    assert not compare_reset(tmp_path,1)['passed']


@pytest.mark.parametrize('failure',['empty','short','extra','wrong_request','missing_request','wrong_step',
                                    'boolean_step','zero_vocab','changed_vocab','invalid_token','wrong_argmax','different_forced_input'])
def test_reset_validates_both_complete_token_sequences(tmp_path,failure):
    rows=reset_capture(tmp_path)
    if failure=='empty':rows=[]
    elif failure=='short':rows.pop()
    elif failure=='extra':rows.append(dict(rows[-1],step=4))
    elif failure=='wrong_request':rows[2]['request']=0
    elif failure=='missing_request':del rows[2]['request']
    elif failure=='wrong_step':rows[1]['step']=2
    elif failure=='boolean_step':rows[1]['step']=True
    elif failure=='zero_vocab':rows[0]['vocab']=0
    elif failure=='changed_vocab':rows[2]['vocab']=4
    elif failure=='invalid_token':rows[0]['token']=3
    elif failure=='wrong_argmax':rows[0]['token']=0
    elif failure=='different_forced_input':rows[2]['next_input_token']=2
    write(tmp_path/'tokens.jsonl',rows)
    assert not compare_reset(tmp_path,2)['passed']


@pytest.mark.parametrize('failure',['missing_both_same_step','extra_step','out_of_order','empty','wrong_logical_length'])
def test_reset_rejects_missing_extra_or_invalid_observed_state_coverage(tmp_path,failure):
    reset_capture(tmp_path)
    rows=[event(i) for i in range(4)]
    if failure=='missing_both_same_step':rows=[event(0),event(2)]
    elif failure=='extra_step':rows.append(event(4))
    elif failure=='out_of_order':rows=[event(1),event(0),event(3),event(2)]
    elif failure=='empty':rows=[]
    elif failure=='wrong_logical_length':rows[0]['logical_bytes']=0
    write(tmp_path/'state_hashes.jsonl',rows)
    assert not compare_reset(tmp_path,2)['passed']


@pytest.mark.parametrize('failure',['missing_logits','extra_logits','missing_tokens','escaping_logits'])
def test_reset_requires_exact_capture_files(tmp_path,failure):
    reset_capture(tmp_path,1)
    if failure=='missing_logits':(tmp_path/'logits-1.f32').unlink()
    elif failure=='extra_logits':(tmp_path/'logits-2.f32').write_bytes(b'')
    elif failure=='missing_tokens':(tmp_path/'tokens.jsonl').unlink()
    elif failure=='escaping_logits':
        other=tmp_path.parent/(tmp_path.name+'-outside.f32');other.write_bytes(struct.pack('<3f',0.,2.,1.))
        (tmp_path/'logits-1.f32').unlink();(tmp_path/'logits-1.f32').symlink_to(other)
    assert not compare_reset(tmp_path,1)['passed']


@pytest.mark.parametrize('count',[0,-1,True,1.5,65])
def test_reset_request_length_is_bounded_integer(tmp_path,count):
    reset_capture(tmp_path,1)
    assert not compare_reset(tmp_path,count)['passed']
