"""AirLLM public API, imported lazily to keep metadata/native startup lightweight.

Legacy class names resolve to the same modules as before. Importing AirLLM no
longer initializes Torch, Transformers and every optional model family eagerly.
"""
import importlib
import sys

_EXPORTS={
 'AutoModel':'.auto_model','AirLLMBaseModel':'.airllm_base',
 'AirLLMLlamaMlx':'.airllm_llama_mlx','AirLLMLlama2':'.airllm',
 'AirLLMChatGLM':'.airllm_chatglm','AirLLMQWen':'.airllm_qwen',
 'AirLLMQWen2':'.airllm_qwen2','AirLLMBaichuan':'.airllm_baichuan',
 'AirLLMInternLM':'.airllm_internlm','AirLLMMistral':'.airllm_mistral',
 'AirLLMMixtral':'.airllm_mixtral','AirLLMKimiK3':'.airllm_kimi_k3',
 'AirLLMQwen3_5':'.airllm_qwen3_5','AirLLMQwen4Exp':'.airllm_qwen4_exp',
 'AirLLMLoRA':'.airllm_lora','AirLLMLoRAQwen4Exp':'.airllm_lora',
 'split_and_save_layers':'.utils','NotEnoughSpaceException':'.utils',
 'compress_layer_state_dict':'.utils','uncompress_layer_state_dict':'.utils',
}
__all__=[k for k in _EXPORTS if k!='AirLLMLlamaMlx' or sys.platform=='darwin']


def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module=importlib.import_module(_EXPORTS[name],__name__)
    value=getattr(module,name);globals()[name]=value;return value


def __dir__():return sorted(set(globals())|set(_EXPORTS))
