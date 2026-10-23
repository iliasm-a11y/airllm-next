import pytest

from airllm.nextgen.gguf_tokenizer import GGUFTokenizer


def fixture_metadata():
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['<|im_start|>', '<|im_end|>', '<think>', '</think>']
    return {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.merges': [],
            'tokenizer.ggml.token_type': [1] * 256 + [3, 3, 4, 4],
            'tokenizer.ggml.eos_token_id': 257}


def test_unicode_nfc_and_added_token_boundaries():
    tokenizer = GGUFTokenizer(fixture_metadata())
    assert tokenizer.encode(' e\u0301') == tokenizer.encode(' é')
    ids = tokenizer.encode('<think>été</think>')
    assert ids[0] == 258 and ids[-1] == 259
    assert tokenizer.decode(ids) == '<think>été</think>'
    assert tokenizer.decode(tokenizer.encode('<|im_end|>'), skip_special_tokens=True) == ''


def test_plain_text_template_and_rejections():
    tokenizer = GGUFTokenizer(fixture_metadata())
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Bonjour'}], tokenize=False)
    assert text == '<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert tokenizer.apply_chat_template([{'role': 'user', 'content': '  Bonjour \n'}], tokenize=False) == text
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template([{'role': 'user', 'content': []}])
    broken = fixture_metadata(); broken['tokenizer.ggml.tokens'] += [broken['tokenizer.ggml.tokens'][0]]
    with pytest.raises(ValueError):
        GGUFTokenizer(broken)


def test_system_multi_turn_and_thinking_generation_prefix():
    tokenizer = GGUFTokenizer(fixture_metadata())
    messages = [{'role': 'system', 'content': '  Réponds en français.\n'},
                {'role': 'user', 'content': '\nPremier\n'},
                {'role': 'assistant', 'content': '  Réponse  \n'},
                {'role': 'user', 'content': '  Suivant\n'}]
    expected = ('<|im_start|>system\nRéponds en français.<|im_end|>\n'
                '<|im_start|>user\nPremier<|im_end|>\n'
                '<|im_start|>assistant\nRéponse<|im_end|>\n'
                '<|im_start|>user\nSuivant<|im_end|>\n')
    assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) == expected
    text = expected + '<|im_start|>assistant\n<think>\n'
    assert tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=True) == text
    assert tokenizer.apply_chat_template(messages, enable_thinking=True) == tokenizer.encode(text)


@pytest.mark.parametrize('assistant', [
    {'role': 'assistant', 'content': '<think>\n  raison  \n</think>\n\nréponse'},
    {'role': 'assistant', 'content': 'réponse', 'reasoning_content': '  raison  '},
])
def test_official_historical_thinking_removal_and_preservation(assistant):
    tokenizer = GGUFTokenizer(fixture_metadata())
    messages = [{'role': 'user', 'content': 'A'}, assistant, {'role': 'user', 'content': 'B'}]
    common = '<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n'
    ending = 'réponse<|im_end|>\n<|im_start|>user\nB<|im_end|>\n'
    assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) == common + ending
    assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         preserve_thinking=True) == common + '<think>\nraison\n</think>\n\n' + ending
    # The assistant after the final user keeps reasoning even without preserve_thinking.
    assert tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=False) == (
        common + '<think>\nraison\n</think>\n\nréponse<|im_end|>\n')


def test_completed_assistant_keeps_empty_thinking_wrapper_and_prefix():
    tokenizer = GGUFTokenizer(fixture_metadata())
    messages = [{'role': 'user', 'content': 'Question'}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False)
    completed = messages + [{'role': 'assistant', 'content': 'Réponse'}]
    assert tokenizer.apply_chat_template(completed, tokenize=False, add_generation_prompt=False) == (
        prefix + 'Réponse<|im_end|>\n')
    # preserve_thinking gives the same text when the completed turn becomes historical.
    followup = completed + [{'role': 'user', 'content': 'Ensuite'}]
    assert tokenizer.apply_chat_template(followup, tokenize=False, preserve_thinking=True).startswith(
        prefix + 'Réponse<|im_end|>\n')


@pytest.mark.parametrize('messages', [
    [{'role': 'system', 'content': 'sans utilisateur'}],
    [{'role': 'assistant', 'content': 'sans utilisateur'}],
    [{'role': 'user', 'content': '<tool_response>réponse outil</tool_response>'}],
    [{'role': 'user', 'content': 'A'}, {'role': 'system', 'content': 'B'}],
])
def test_official_invalid_conversation_rejections(messages):
    tokenizer = GGUFTokenizer(fixture_metadata())
    with pytest.raises(ValueError):
        tokenizer.apply_chat_template(messages)


@pytest.mark.parametrize('message', [
    {'role': 'tool', 'content': 'tool'},
    {'role': 'user', 'content': 'A', 'reasoning_content': 'B'},
    {'role': 'assistant', 'content': 'A', 'reasoning_content': None},
    {'role': 'assistant', 'content': 'A', 'tool_calls': []},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'A'}]},
    'not a message',
])
def test_text_only_contract_rejects_unsupported_shapes(message):
    tokenizer = GGUFTokenizer(fixture_metadata())
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template([{'role': 'user', 'content': 'Q'}, message])


def test_template_flags_cannot_silently_coerce_strings():
    tokenizer = GGUFTokenizer(fixture_metadata())
    with pytest.raises(TypeError):
        tokenizer.apply_chat_template([{'role': 'user', 'content': 'Q'}], enable_thinking='false')
