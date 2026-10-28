"""Conversation/dispatch/diagnostic contracts with explicitly mocked models.

These tests do not establish GLM inference or M5 execution fidelity.
"""
from contextlib import nullcontext
import json
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from airllm.auto_model import AutoConfig, AutoModel
from airllm.nextgen.gguf_chat import GGUFChatSession
from airllm.nextgen import glm53_cli, ronin_glm53


class MockTokenizer:
    vocab_size = 256
    eos_token_id = 255

    def __init__(self):
        self.received = []

    def encode(self, text):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False):
        return ''.join('' if value == 255 and skip_special_tokens else chr(value)
                       for value in ids)

    def apply_chat_template(self, messages, **options):
        self.received.append(options)
        return '|'.join(message['content'] for message in messages) + '|'


class MockModel:
    """Return fixed test logits; never label these as checkpoint outputs."""

    def __init__(self):
        self.config = {'vocab_size': 256, 'max_position_embeddings': 1024,
                       'eos_token_id': [255]}
        self.options = SimpleNamespace(max_context=1024)
        self.lock = RLock()
        self.closed = False
        self.mx = SimpleNamespace(stream=lambda _: nullcontext(), argmax=np.argmax,
                                  all=np.all, isfinite=np.isfinite)
        self.stream = object()
        self.stats = {}
        self.calls = 0

    def new_session(self):
        return SimpleNamespace(position=0, peak_bytes=0)

    def forward(self, ids, *, past_key_values, logits_to_keep):
        token = [ord('O'), ord('K'), 255][self.calls]
        self.calls += 1
        past_key_values.position += ids.shape[1]
        logits = np.zeros((1, 1, 256), dtype=np.float32)
        logits[0, 0, token] = 1
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('specialized', [False, True])
def test_template_options_are_architecture_specific_and_reported(specialized):
    tokenizer, model = MockTokenizer(), MockModel()
    if specialized:
        tokenizer.chat_template_options = {'reasoning_effort': 'max', 'clear_thinking': False}
    expected = ({'reasoning_effort': 'max', 'clear_thinking': False} if specialized
                else {'enable_thinking': False, 'preserve_thinking': True})
    events = []
    result = GGUFChatSession(model, tokenizer).reply('Q', max_new_tokens=5,
                                                   on_token=events.append)
    assert tokenizer.received == [dict(expected, tokenize=False, add_generation_prompt=True)]
    assert result['generation_parameters'] == dict(expected, max_new_tokens=5,
                                                   temperature=0., do_sample=False, reuse_kv=True)
    assert result['text'] == 'OK'
    assert result['stop_reason'] == 'eos'
    assert [event['token_id'] for event in events] == [ord('O'), ord('K'), 255]


def test_diagnostics_does_not_claim_unmeasured_readiness():
    result = ronin_glm53.diagnostics()
    assert result['local_text_conversation_adapter'] is True
    assert result['experimental'] is True
    for key in ('real_checkpoint_generation_validated', 'm5_execution_validated',
                'ronin_ready', 'vision_execution', 'mtp_execution', 'external_tool_execution'):
        assert result[key] is False


def install_mock_runtime(monkeypatch, factory):
    module = ModuleType('airllm.nextgen.gguf_glm5')
    module.NativeGGUFGLM5Model = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_automodel_native_dispatch_never_resolves_hub(monkeypatch, tmp_path):
    calls = []
    sentinel = object()
    install_mock_runtime(monkeypatch,
                         lambda path, **options: calls.append((path, options)) or sentinel)
    monkeypatch.setattr(AutoConfig, 'from_pretrained',
                        lambda *args, **kwargs: pytest.fail('Unexpected Transformers/Hub dispatch'))
    path = tmp_path / 'local.gguf'
    assert AutoModel.from_pretrained(path, backend='next-gguf-glm5', kernel='reference') is sentinel
    assert calls == [(path, {'kernel': 'reference'})]
    with pytest.raises(TypeError, match='keyword options'):
        AutoModel.from_pretrained(path, 'positional', backend='next-gguf-glm5')
    assert len(calls) == 1


def test_conversation_closes_index_and_model_when_session_rejected(monkeypatch, tmp_path):
    closed = []

    class MockReader:
        def __init__(self, *args, **kwargs):
            self.metadata = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append('index')

    model = MockModel()
    model.config['vocab_size'] += 1  # actual session validation must reject it
    install_mock_runtime(monkeypatch, lambda *args, **kwargs: model)
    monkeypatch.setattr(ronin_glm53, 'SplitGGUFReader', MockReader)
    monkeypatch.setattr(ronin_glm53, 'GLM5NextGGUFTokenizer', lambda _: MockTokenizer())
    with pytest.raises(ValueError, match='vocabulary'):
        ronin_glm53.load_conversation(tmp_path / 'local.gguf')
    assert closed == ['index']
    assert model.closed is True


def test_conversation_bridge_returns_owned_model_and_local_session(monkeypatch, tmp_path):
    class MockReader:
        def __init__(self, *args, **kwargs):
            self.metadata = {'source': 'unit-test'}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

    model, tokenizer = MockModel(), MockTokenizer()
    calls = []
    install_mock_runtime(monkeypatch,
                         lambda path, **kwargs: calls.append((path, kwargs)) or model)
    monkeypatch.setattr(ronin_glm53, 'SplitGGUFReader', MockReader)
    monkeypatch.setattr(ronin_glm53, 'GLM5NextGGUFTokenizer', lambda _: tokenizer)
    options = object()
    path = tmp_path / 'local.gguf'
    loaded, session = ronin_glm53.load_conversation(path, next_config=options,
                                                  system_prompt='System')
    assert loaded is model and loaded.tokenizer is tokenizer
    assert session.messages == [{'role': 'system', 'content': 'System'}]
    assert model.closed is False
    assert calls == [(path, {'next_config': options})]
    model.close()


def test_cli_inspection_never_reports_inference(monkeypatch, tmp_path):
    module = install_mock_runtime(monkeypatch, type('MockRuntime', (), {
        '_validate_layout': lambda self: None}))
    module.gguf_glm5_config = lambda checkpoint: {'model_type': 'glm5_next'}

    class MockReader:
        def __init__(self, *args, **kwargs):
            self.paths = [tmp_path / 'first.gguf', tmp_path / 'second.gguf']
            self.file_size = 123
            self.tensors = {'weight': SimpleNamespace(dtype='Q2_K')}
            self.stats = {'tensor_bytes': 0}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    from airllm.nextgen import gguf_split
    monkeypatch.setattr(gguf_split, 'SplitGGUFReader', MockReader)
    result = glm53_cli.inspect_model(tmp_path / 'first.gguf')
    assert result['status'] == 'layout_checked'
    assert result['inference_performed'] is False
    assert result['weight_payload_loaded'] is False
    assert result['tensor_types'] == {'Q2_K': 1}
    assert result['io']['tensor_bytes'] == 0


def test_failed_worker_keeps_error_and_unqualified_status(monkeypatch, tmp_path):
    def rejected_load(*args, **kwargs):
        raise RuntimeError('Unit-test checkpoint is deliberately unavailable')

    monkeypatch.setattr(ronin_glm53, 'load_conversation', rejected_load)
    args = SimpleNamespace(model=tmp_path / 'absent.gguf', max_context=128,
                           max_new_tokens=1, prompt='Q', device='cpu',
                           report=tmp_path / 'failure.json')
    with pytest.raises(RuntimeError, match='deliberately unavailable'):
        glm53_cli._worker(args)
    result = json.loads(args.report.read_text())
    assert result['status'] == 'failed'
    assert result['inference_performed'] is False
    assert result['error_type'] == 'RuntimeError'
    assert result['ronin_ready'] is False
    assert result['real_checkpoint_generation_validated'] is False


def test_cli_rejects_missing_report_and_invalid_limits_before_loading():
    with pytest.raises(SystemExit) as missing:
        glm53_cli.main(['run', '/nonexistent/model.gguf'])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as invalid:
        glm53_cli.main(['inspect', '/nonexistent/model.gguf', '--max-context', '0'])
    assert invalid.value.code == 2
