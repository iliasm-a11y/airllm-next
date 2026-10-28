"""Experimental local conversation bridge; no external actions are executed."""
from .gguf_chat import GGUFChatSession
from .gguf_split import SplitGGUFReader
from .glm5_next_tokenizer import GLM5NextGGUFTokenizer


def load_conversation(model_path, *, next_config=None, system_prompt=None):
    """Load only local files, preserving the embedded GLM template/token IDs.

    Call ``session.reply(prompt, on_token=callback)`` for real token events.
    The caller owns the returned model and must close it. This adapter exposes
    no shell, filesystem write, network tool, or automatic tool execution.
    Numerical/real-checkpoint qualification remains a separate evidence step.
    """
    from .gguf_glm5 import NativeGGUFGLM5Model
    with SplitGGUFReader(model_path, memory_budget_bytes=256 << 20,
                         max_metadata_bytes=192 << 20, bulk_metadata=True,
                         string_buffer_bytes=65536) as checkpoint:
        tokenizer = GLM5NextGGUFTokenizer(checkpoint.metadata)
    model = NativeGGUFGLM5Model(model_path, next_config=next_config)
    try:
        model.tokenizer = tokenizer
        session = GGUFChatSession(model, tokenizer, system_prompt=system_prompt)
    except BaseException:
        model.close()
        raise
    return model, session


def diagnostics():
    return {'backend': 'next-gguf-glm5', 'experimental': True,
            'local_text_conversation_adapter': True,
            'real_checkpoint_generation_validated': False,
            'm5_execution_validated': False, 'ronin_ready': False,
            'vision_execution': False, 'mtp_execution': False,
            'external_tool_execution': False}
