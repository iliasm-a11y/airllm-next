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
