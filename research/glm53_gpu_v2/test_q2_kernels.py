import os
import numpy as np
import pytest
from research.glm53_gpu_v2.q2_kernels import Q2KVariants, prepare_activations, payload_requirement, VARIANTS
from research.glm53_gpu.quantized_probe import NativeOracle


def packed_fixture(experts=2,rows=7,width=512):
    rng=np.random.default_rng(14)
    packed=rng.integers(0,256,(experts,rows,width//256*84),dtype='u1')
    b=packed.reshape(experts,rows,width//256,84)
    b[...,80:84]=np.frombuffer(np.array([.01,.02],'<f2').tobytes(),'u1')
    return packed


def test_payload_failclosed():
    packed=packed_fixture()
    assert payload_requirement(packed,512,variant='twopass')>payload_requirement(packed,512)
    with pytest.raises(MemoryError): Q2KVariants(packed,512,budget_bytes=1)
    with pytest.raises(ValueError): payload_requirement(packed,512,activation_count=3)
    with pytest.raises(ValueError): payload_requirement(packed[...,::2],256)
    with pytest.raises(ValueError): payload_requirement(packed,513)


@pytest.mark.parametrize('width',[256,768,4096])
@pytest.mark.parametrize('activation_count',[1,2])
def test_variants_native_bitexact(width,activation_count):
    mx=pytest.importorskip('mlx.core')
    libraries=os.environ.get('AIRLLM_GGML_LIBRARIES')
    if not libraries or not mx.metal.is_available(): pytest.skip('Metal and native oracle required')
    oracle=NativeOracle(libraries)
    p=packed_fixture(width=width)
    rng=np.random.default_rng(19)
    x=rng.normal(size=(activation_count,width)).astype('f4')
    expected=np.stack([oracle.matvec(p[e],x[0 if activation_count==1 else e]) for e in range(2)])
    activations=prepare_activations(x)
    with Q2KVariants(p,width) as matrix:
        for variant in VARIANTS:
            for nsg in (1,4):
                actual=np.asarray(matrix.execute(activations,variant,simd_groups=nsg))
                np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'),err_msg=f'{variant}/{nsg}')
    activations.close()


def test_prepared_input_rejects_invalid():
    pytest.importorskip('mlx.core')
    for x in [np.zeros(257,'f4'),np.full(256,np.inf,'f4'),np.zeros(256,'f8')]:
        with pytest.raises(ValueError): prepare_activations(x)
