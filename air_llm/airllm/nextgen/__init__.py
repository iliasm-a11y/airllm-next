"""AirLLM native GLM DSA execution, selective streaming and packed weights."""
from .config import NextConfig

__version__ = '5.0.0a1'


def __getattr__(name):
    if name in ('LocalOrchestrator', 'OrchestratorConfig', 'InferenceRequest', 'InferenceResult'):
        from . import orchestrator
        return getattr(orchestrator, name)
    if name == 'NativeModelBackend':
        from .orchestrator_native import NativeModelBackend
        return NativeModelBackend
    if name in ('ToolBroker', 'ToolLimits'):
        from . import agent_tools
        return getattr(agent_tools, name)
    if name == 'AirLLMNextModel':
        from .runtime import AirLLMNextModel
        return AirLLMNextModel
    if name == 'AirLLMNextLlamaModel':
        from .llama_runtime import AirLLMNextLlamaModel
        return AirLLMNextLlamaModel
    if name == 'AirLLMNextGraniteMoeModel':
        from .granite_runtime import AirLLMNextGraniteMoeModel
        return AirLLMNextGraniteMoeModel
    if name == 'AgentEngine':
        from .agent_engine import AgentEngine
        return AgentEngine
    raise AttributeError(name)
