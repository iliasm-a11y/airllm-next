"""AirLLM native GLM DSA execution, selective streaming and packed weights."""
from .config import NextConfig


def __getattr__(name):
    if name == 'AirLLMNextModel':
        from .runtime import AirLLMNextModel
        return AirLLMNextModel
    raise AttributeError(name)
