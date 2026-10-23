"""Chat orchestration tests without loading checkpoint weights or GPU work."""
from contextlib import nullcontext
import json
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.gguf_chat import GGUFChatSession


class TextTokenizer:
    vocab_size = 256
    eos_token_id = 255

    def encode(self, text):
        return list(map(ord, text.replace('<|im_end|>', '\xff')))

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(('' if skip_special_tokens else '<|im_end|>')
                       if token == 255 else chr(token) for token in ids)

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=True, enable_thinking=False,
                            preserve_thinking=False):
        assert enable_thinking is False and preserve_thinking is True
        text = ''
        for message in messages:
            text += '<|im_start|>' + message['role'] + '\n'
            if message['role'] == 'assistant':
                text += '<think>\n\n</think>\n\n'
            text += message['content'].strip() + '<|im_end|>\n'
        if add_generation_prompt:
            text += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return self.encode(text) if tokenize else text


class FakeModel:
    def __init__(self, answers=('OK', 'YES'), context=4096):
        self.config = {'vocab_size': 256, 'max_position_embeddings': context,
                       'eos_token_id': [255]}
        self.options = SimpleNamespace(max_context=context)
        self.lock = RLock()
        self.closed = False
        self.mx = SimpleNamespace(stream=lambda stream: nullcontext(), argmax=np.argmax,
                                  all=np.all, isfinite=np.isfinite)
        self.stream = object()
        self.stats = {}
        self.calls = []
        self.answers = answers
        self.failure_call = None

    def new_session(self):
        return SimpleNamespace(position=0, ids=[], peak_bytes=0)

    def forward(self, ids, *, past_key_values, logits_to_keep):
        assert logits_to_keep == 1
        cache = past_key_values
        current = ids.tolist()[0]
        self.calls.append({'ids': current, 'position': cache.position})
        if self.failure_call == len(self.calls):
            raise MemoryError('simulated native guard')
        cache.ids.extend(current)
        cache.position += len(current)
        cache.peak_bytes = max(cache.peak_bytes, cache.position * 4)
        text = TextTokenizer().decode(cache.ids)
        turn = text.count('<|im_start|>user\n') - 1
        generated = text.rsplit('<think>\n\n</think>\n\n', 1)[1]
        answer = self.answers[turn]
        token = ord(answer[len(generated)]) if len(generated) < len(answer) else 255
        logits = np.zeros((1, 1, 256), dtype=np.float32)
        logits[0, 0, token] = 1
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_actual_history_eos_and_json_serializable_measurements():
    model = FakeModel()
    events = []
    session = GGUFChatSession(model, TextTokenizer(), ' System ')
    result = session.reply(' User ', max_new_tokens=5, on_token=events.append)
    assert result['generated_ids'] == [ord('O'), ord('K'), 255]
    assert result['text'] == 'OK'
    assert result['raw_text'] == 'OK<|im_end|>'
    assert result['stop_reason'] == 'eos'
    assert result['messages'] == [
        {'role': 'system', 'content': ' System '},
        {'role': 'user', 'content': ' User '},
        {'role': 'assistant', 'content': 'OK'}]
    assert result['eos_token_ids'] == [255]
    assert result['finite_logits_checked'] is True
    assert result['reused_prefix_tokens'] == 0
    assert result['full_output_ids'] == result['full_input_ids'] + result['generated_ids']
    assert result['native_cache_position'] == len(result['full_input_ids']) + 2
    assert [event['token_id'] for event in events] == result['generated_ids']
    assert [event['token_index'] for event in events] == [0, 1, 2]
    assert len(result['token_seconds']) == 3
    assert all(seconds >= 0 for seconds in result['token_seconds'])
    assert 0 < result['ttft_seconds'] <= result['total_seconds']
    assert result['decode_tokens_per_second'] > 0
    assert result['generation_parameters'] == {
        'max_new_tokens': 5, 'temperature': 0., 'do_sample': False,
        'enable_thinking': False, 'preserve_thinking': True, 'reuse_kv': True}
    assert model.stats['tokens_generated'] == 3
    json.dumps(result, allow_nan=False)
    session.messages[0]['content'] = 'changed'
    assert session.messages[0]['content'] == ' System '


def test_incremental_cache_reuses_consumed_prefix_and_appends_last_eos():
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    first = session.reply('one', max_new_tokens=5)
    second_start = len(model.calls)
    second = session.reply('two', max_new_tokens=5)
    consumed = first['full_input_ids'] + first['generated_ids'][:-1]
    assert second['reused_prefix_tokens'] == len(consumed)
    assert second['cached_prefix_matched'] is True
    assert second['full_input_ids'][:len(consumed)] == consumed
    assert second['appended_input_ids'][0] == 255
    assert model.calls[second_start] == {
        'ids': second['appended_input_ids'], 'position': len(consumed)}
    assert second['text'] == 'YES'
    assert second['native_cache_position'] == len(second['full_input_ids']) + 3
    ablation = GGUFChatSession(FakeModel(), TextTokenizer(), reuse_kv=False)
    same_first = ablation.reply('one', max_new_tokens=5)
    same_second = ablation.reply('two', max_new_tokens=5)
    assert same_first['generated_ids'] == first['generated_ids']
    assert same_second['generated_ids'] == second['generated_ids']
    assert same_second['full_input_ids'] == second['full_input_ids']
    assert same_second['messages'] == second['messages']
    assert same_second['reused_prefix_tokens'] == 0
    assert same_second['appended_input_ids'] == second['full_input_ids']


def test_partial_assistant_is_retained_and_final_unconsumed_token_is_appended():
    model = FakeModel(answers=('ABC', 'YES'))
    session = GGUFChatSession(model, TextTokenizer())
    first = session.reply('one', max_new_tokens=2)
    assert first['text'] == 'AB'
    assert first['stop_reason'] == 'max_new_tokens'
    assert first['messages'][-1]['content'] == 'AB'
    second = session.reply('two', max_new_tokens=4)
    assert second['reused_prefix_tokens'] == len(first['full_input_ids']) + 1
    assert second['appended_input_ids'][0] == ord('B')
    assert second['messages'][1]['content'] == 'AB'


def test_template_change_rebuilds_instead_of_reusing_wrong_state():
    model = FakeModel(answers=(' OK', 'YES'))
    session = GGUFChatSession(model, TextTokenizer())
    first = session.reply('one', max_new_tokens=5)
    assert first['messages'][-1]['content'] == ' OK'
    call_start = len(model.calls)
    second = session.reply('two', max_new_tokens=5)
    assert second['reused_prefix_tokens'] == 0
    assert second['cached_prefix_matched'] is False
    assert model.calls[call_start]['position'] == 0
    assert second['appended_input_ids'] == second['full_input_ids']


def test_context_rejection_preserves_history_and_cache_before_append():
    model = FakeModel(context=4096)
    session = GGUFChatSession(model, TextTokenizer())
    first = session.reply('one', max_new_tokens=5)
    history = session.messages
    calls = len(model.calls)
    model.options.max_context = len(first['full_input_ids'])
    with pytest.raises(MemoryError, match='context'):
        session.reply('two', max_new_tokens=5)
    assert len(model.calls) == calls
    assert session.messages == history
    model.options.max_context = 4096
    assert session.reply('two', max_new_tokens=5)['reused_prefix_tokens'] > 0


def test_failed_generation_does_not_commit_partial_history_and_retry_rebuilds():
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    session.reply('one', max_new_tokens=5)
    history = session.messages
    model.failure_call = len(model.calls) + 2
    with pytest.raises(MemoryError, match='native guard'):
        session.reply('two', max_new_tokens=5)
    assert session.messages == history
    model.failure_call = None
    result = session.reply('two', max_new_tokens=5)
    assert result['reused_prefix_tokens'] == 0
    assert result['text'] == 'YES'


@pytest.mark.parametrize('kwargs,kind', [
    ({'temperature': 1.}, ValueError),
    ({'temperature': float('nan')}, ValueError),
    ({'temperature': True}, ValueError),
    ({'max_new_tokens': 0}, ValueError),
    ({'max_new_tokens': -1}, ValueError),
    ({'max_new_tokens': 1.5}, ValueError),
    ({'do_sample': False}, TypeError),
    ({'top_p': 1.}, TypeError),
    ({'seed': 1}, TypeError),
    ({'on_token': False}, TypeError),
])
def test_unsupported_options_rejected_before_model_call(kwargs, kind):
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    with pytest.raises(kind):
        session.reply('one', **kwargs)
    assert model.calls == [] and session.messages == []


def test_model_stop_ids_take_precedence_and_tokenizer_fallback():
    model = FakeModel()
    model.config['eos_token_id'] = None
    session = GGUFChatSession(model, TextTokenizer())
    assert session.reply('one', max_new_tokens=5)['stop_reason'] == 'eos'


def test_callback_failure_discards_advanced_cache_without_committing_turn():
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    def fail(event):
        raise RuntimeError('callback failure')
    with pytest.raises(RuntimeError, match='callback'):
        session.reply('one', max_new_tokens=5, on_token=fail)
    assert session.messages == []
    assert session.reply('one', max_new_tokens=5)['reused_prefix_tokens'] == 0


def test_closed_model_and_bad_vocabulary_rejected_without_forward():
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    model.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        session.reply('one')
    model.config['vocab_size'] = 300
    with pytest.raises(ValueError, match='vocabulary'):
        GGUFChatSession(model, TextTokenizer())
    assert model.calls == []


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_logits_fail_before_emitting_and_discard_cache(invalid):
    model = FakeModel()
    session = GGUFChatSession(model, TextTokenizer())
    actual = model.forward
    def invalid_forward(*args, **kwargs):
        output = actual(*args, **kwargs)
        output.logits[0, 0, 0] = invalid
        return output
    model.forward = invalid_forward
    events = []
    with pytest.raises(FloatingPointError, match='Non-finite'):
        session.reply('one', max_new_tokens=5, on_token=events.append)
    assert events == [] and session.messages == []
    model.forward = actual
    result = session.reply('one', max_new_tokens=5)
    assert result['reused_prefix_tokens'] == 0
    assert result['text'] == 'OK'
