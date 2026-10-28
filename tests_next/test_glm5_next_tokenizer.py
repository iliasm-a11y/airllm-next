"""Text tokenizer unit tests; real-vocabulary parity is opt-in and offline."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from airllm.nextgen.glm5_next_tokenizer import GLM5NextGGUFTokenizer


TEMPLATE = Path(__file__).parent / 'fixtures' / 'glm53' / 'chat_template.jinja'
TEMPLATE_SHA256 = '2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4'


def fixture_metadata():
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['<|endoftext|>', '[gMASK]', '<sop>',
                                            '<|system|>', '<|user|>', '<|assistant|>',
                                            '<|observation|>', '<think>', '</think>',
                                            'abc']
    return {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'glm4',
            'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.merges': [],
            'tokenizer.ggml.token_type': [1] * 256 + [3] * 7 + [4, 4, 1],
            'tokenizer.ggml.eos_token_id': 256, 'tokenizer.ggml.bos_token_id': 257,
            'tokenizer.ggml.eot_token_id': 260, 'tokenizer.ggml.eom_token_id': 262,
            'tokenizer.ggml.add_bos_token': False, 'tokenizer.ggml.add_eos_token': False,
            'tokenizer.chat_template': TEMPLATE.read_text()}


def test_official_template_provenance():
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == TEMPLATE_SHA256


def test_glm_bpe_preserves_unicode_and_direct_vocab_tokens():
    tokenizer = GLM5NextGGUFTokenizer(fixture_metadata())
    assert tokenizer.encode('é') != tokenizer.encode('e\u0301')
    for text in ('é', 'e\u0301', '你好 🧑🏽\u200d💻', '\n  \t\r\n'):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    # GLM4 ignore_merges allows the entire regex piece if it is in vocab,
    # even when no merge path creates that vocabulary entry.
    assert tokenizer.encode('abc') == [265]
    assert [piece for piece, _ in tokenizer.engine.pre_tokenizer.pre_tokenize_str('1234567')] == [
        '123', '456', '7']
    assert tokenizer.encode('<think>abc</think>') == [263, 265, 264]
    assert tokenizer.decode([256], skip_special_tokens=True) == ''
    assert tokenizer.decode([263], skip_special_tokens=True) == '<think>'
    assert tokenizer.stop_token_ids == {256, 260, 262}


def test_actual_glm_template_golden_and_reasoning_history():
    tokenizer = GLM5NextGGUFTokenizer(fixture_metadata())
    messages = [{'role': 'user', 'content': '  Bonjour\n'}]
    expected = ('[gMASK]<sop><|system|>Reasoning Effort: Max'
                '<|user|>  Bonjour\n<|assistant|><think>')
    assert tokenizer.apply_chat_template(messages, tokenize=False) == expected
    assert tokenizer.apply_chat_template(messages) == tokenizer.encode(expected)
    assert tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort='low') == (
        expected.replace('Effort: Max', 'Effort: Low'))
    followup = [{'role': 'user', 'content': 'A'},
                {'role': 'assistant', 'content': '<think>raison</think> réponse '},
                {'role': 'user', 'content': 'B'}]
    assert '<|assistant|><think>raison</think>réponse<|user|>B' in tokenizer.apply_chat_template(
        followup, tokenize=False, preserve_thinking=True)
    assert '<|assistant|><think></think>réponse<|user|>B' in tokenizer.apply_chat_template(
        followup, tokenize=False, clear_thinking=True)


@pytest.mark.parametrize('patch,error', [
    ({'tokenizer.ggml.pre': 'qwen35'}, NotImplementedError),
    ({'tokenizer.ggml.add_bos_token': True}, NotImplementedError),
    ({'tokenizer.ggml.eos_token_id': -1}, ValueError),
    ({'tokenizer.ggml.merges': ['absent token']}, ValueError),
    ({'tokenizer.chat_template': ''}, ValueError),
])
def test_metadata_rejected_before_use(patch, error):
    metadata = fixture_metadata()
    metadata.update(patch)
    with pytest.raises(error):
        GLM5NextGGUFTokenizer(metadata)


def test_unsupported_options_and_non_text_fail_closed():
    tokenizer = GLM5NextGGUFTokenizer(fixture_metadata())
    messages = [{'role': 'user', 'content': 'Q'}]
    with pytest.raises(NotImplementedError, match='cannot disable thinking'):
        tokenizer.apply_chat_template(messages, enable_thinking=False)
    with pytest.raises(ValueError):
        tokenizer.apply_chat_template(messages, reasoning_effort='unlimited')
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}]}])
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template(messages + [{'role': 'tool', 'content': 'x'}])
    for ids in ([True], [-1], [1.2], [tokenizer.vocab_size]):
        with pytest.raises(ValueError):
            tokenizer.decode(ids)


def test_real_gguf_vocabulary_against_independent_source_tokenizer():
    """Set both paths to verified small assets, never full model weights.

    GLM53_TOKENIZER_METADATA is the metadata JSON of the first Q2_K_S shard;
    GLM53_HF_TOKENIZER is Z.ai's tokenizer.json from pinned source revision.
    Neither fixture is silently downloaded by this test.
    """
    metadata_path = os.environ.get('GLM53_TOKENIZER_METADATA')
    reference_path = os.environ.get('GLM53_HF_TOKENIZER')
    if not metadata_path or not reference_path:
        pytest.skip('Verified real tokenizer assets were not provided')
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import render_jinja_template
    metadata = json.loads(Path(metadata_path).read_text())
    assert hashlib.sha256(Path(reference_path).read_bytes()).hexdigest() == (
        '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d')
    tokenizer = GLM5NextGGUFTokenizer(metadata)
    reference = Tokenizer.from_file(reference_path)
    samples = ['Bonjour, café et cafe\u0301. 123456!', '你好 world\nHi',
               '[gMASK]<sop><|user|>Bonjour', '000000000123456789', '<think>été</think>',
               '\r\n  \tfoo_bar:=True; 🧑🏽\u200d💻', '[PAD154856]', 'Hello\x00World']
    for text in samples:
        expected = reference.encode(text, add_special_tokens=False).ids
        assert tokenizer.encode(text) == expected
        assert tokenizer.decode(expected) == reference.decode(expected, skip_special_tokens=False)
    conversations = [[{'role': 'system', 'content': 'Réponds brièvement.'},
                      {'role': 'user', 'content': 'Bonjour'}],
                     [{'role': 'user', 'content': 'A'},
                      {'role': 'assistant', 'content': '<think>raison</think>réponse'},
                      {'role': 'user', 'content': 'B'}]]
    for conversation in conversations:
        for effort in ('low', 'high', 'max'):
            for clear in (False, True):
                expected, _ = render_jinja_template(
                    [conversation], chat_template=TEMPLATE.read_text(),
                    add_generation_prompt=True, reasoning_effort=effort, clear_thinking=clear)
                actual = tokenizer.apply_chat_template(conversation, tokenize=False,
                                                       reasoning_effort=effort,
                                                       clear_thinking=clear)
                assert actual == expected[0]
                assert tokenizer.encode(actual) == reference.encode(actual, add_special_tokens=False).ids
