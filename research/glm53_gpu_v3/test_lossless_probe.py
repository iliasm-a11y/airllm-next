import pytest
from research.glm53_gpu_v3.lossless_probe import codec_sample

def test_exact_codec_preserves_arbitrary_bytes_and_expansion():
    x=bytes(range(256))*64;r=codec_sample(x)
    assert r['exact'] and r['original_bytes']==len(x) and len(r['decode_seconds'])==7
    for x in [b'',b'0'*(8*1024**2+1)]:
        with pytest.raises(ValueError):codec_sample(x)
