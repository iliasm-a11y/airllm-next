from .config import UltraConfig

__all__ = ['UltraConfig', 'AirLLMUltraModel']


def __getattr__(name):
    if name == 'AirLLMUltraModel':
        from .runtime import AirLLMUltraModel
        return AirLLMUltraModel
    raise AttributeError(name)
