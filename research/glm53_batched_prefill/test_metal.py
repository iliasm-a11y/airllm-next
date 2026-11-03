"""Admission tests do not require Metal or a 110GB checkpoint."""
import numpy as np
import pytest
from research.glm53_batched_prefill.metal_probe import validate


def inputs(tokens=8, rows=10, width=96):
    weights = np.zeros((rows,width//32*34),np.uint8)
    return weights, np.ones((tokens,width),np.float32)


def test_admits_bounded_multiple_token_matrix():
    w,x=inputs()
    assert validate(w,x) > w.nbytes + x.nbytes


@pytest.mark.parametrize('tokens',[0,1,3,256])
def test_rejects_unsupported_token_shape(tokens):
    with pytest.raises(ValueError): validate(*inputs(tokens=tokens))


def test_rejects_odd_weight_rows():
    with pytest.raises(ValueError): validate(*inputs(rows=9))


def test_rejects_payload_over_budget():
    with pytest.raises(MemoryError): validate(*inputs(),budget_bytes=10)


@pytest.mark.parametrize('value',[np.nan,np.inf,np.float32(1e-40),np.float32(1e20)])
def test_rejects_unqualified_quantization(value):
    w,x=inputs()
    x[0,0]=value
    if value == np.float32(1e-40): x[0,:32]=value
    with pytest.raises(ValueError): validate(w,x)


def test_rejects_non_contiguous_activation():
    w,x=inputs()
    with pytest.raises(ValueError): validate(w,x[:,::-1])
