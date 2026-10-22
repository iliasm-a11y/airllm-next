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


@pytest.mark.parametrize('theta',[0,-1,float('nan'),float('inf'),True,'10000'])
def test_invalid_rope_theta_fails_before_numerical_execution(theta):
    with pytest.raises(ValueError,match='rope_theta'):validated_default_rope({'rope_parameters':{'rope_theta':theta}})


@pytest.mark.parametrize('parameters',[[],False,'default'])
def test_nonobject_rope_parameters_are_not_silently_ignored(parameters):
    with pytest.raises(ValueError,match='object'):validated_default_rope({'rope_parameters':parameters})


@pytest.mark.parametrize('field,value',[
    ('hidden_size',True),('q_lora_rank',None),('index_topk',0),('n_group',0),
    ('topk_group',2),('rms_norm_eps',float('nan')),('routed_scaling_factor',float('inf')),
])
def test_invalid_glm_dimensions_fail_during_normalization(tiny_glm,field,value):
    import json
    root,_=tiny_glm;config=json.loads((root/'config.json').read_text());config[field]=value
    with pytest.raises(ValueError):normalize_config(config)


def test_glm_default_rope_theta_is_filled_when_parameters_omit_it(tiny_glm):
    import json
    root,_=tiny_glm;config=json.loads((root/'config.json').read_text())
    config['rope_parameters']={'rope_type':'default'};config.pop('rope_theta',None)
    assert normalize_config(config)['rope_parameters']['rope_theta']==10000.
