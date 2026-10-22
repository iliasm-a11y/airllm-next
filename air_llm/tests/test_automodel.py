"""Offline regression for current generic/override AirLLM dispatch.

The original historical test fetched remote config files and expected classes
that upstream has since consolidated into AirLLMBaseModel. Mock only config
retrieval, exercising the actual dispatch logic without checkpoint downloads.
"""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from airllm.auto_model import AutoModel


@pytest.mark.parametrize('architecture,expected',[
 ('LlamaForCausalLM','AirLLMBaseModel'),('QWenLMHeadModel','AirLLMQWen'),
 ('InternLMForCausalLM','AirLLMInternLM'),('ChatGLMModel','AirLLMChatGLM'),
 ('BaichuanForCausalLM','AirLLMBaichuan'),('MistralForCausalLM','AirLLMBaseModel'),
 ('MixtralForCausalLM','AirLLMBaseModel'),('GlmMoeDsaForCausalLM','AirLLMBaseModel')])
def test_current_architecture_dispatch_without_network(architecture,expected):
    with patch('airllm.auto_model.AutoConfig.from_pretrained',return_value=SimpleNamespace(architectures=[architecture])):
        assert AutoModel.get_module_class('local-test-fixture')==('airllm',expected)
