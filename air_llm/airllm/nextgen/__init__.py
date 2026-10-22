"""AirLLM native GLM DSA execution, selective streaming and packed weights."""
from .config import NextConfig


def __getattr__(name):
    if name == 'AirLLMNextModel':
        from .runtime import AirLLMNextModel
        return AirLLMNextModel
    if name == 'AirLLMNextLlamaModel':
        from .llama_runtime import AirLLMNextLlamaModel
        return AirLLMNextLlamaModel
    raise AttributeError(name)
