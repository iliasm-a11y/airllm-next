import pytest
from airllm.nextgen.adapters import validated_default_rope
from airllm.nextgen.runtime import normalize_config
from airllm.nextgen.qwen_runtime import normalize_qwen_config


@pytest.mark.parametrize('normalizer,model_type',[(normalize_config,'glm_moe_dsa'),(normalize_qwen_config,'qwen3_5_moe_text')])
@pytest.mark.parametrize('parameters',[
    {'rope_parameters': {'type':'yarn'}},
    {'rope_parameters': {'rope_type':'yarn'}},
    {'rope_parameters': {'rope_type':'default','type':'yarn'}},
    {'rope_scaling': {'type':'linear','factor':2.}},
])
def test_nondefault_rope_never_silently_runs_default(normalizer,model_type,parameters):
    with pytest.raises(NotImplementedError,match='RoPE'):
        normalizer({'model_type':model_type,**parameters})


def test_matching_default_aliases_remain_valid():
    parameters={'type':'default','rope_type':'default','rope_theta':10000.}
    assert validated_default_rope({'rope_parameters':parameters}) == parameters
