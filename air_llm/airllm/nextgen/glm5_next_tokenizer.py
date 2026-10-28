"""GLM4 BPE and the actual GGUF chat template, for GLM-5.3 text input.

The vocabulary, merges, token attributes and Jinja template come from the opened
GGUF, never a Hub download. The GLM4 pre-tokenizer and ``ignore_merges`` match
llama.cpp 74a4fb0037243f1b6196d0f990ff0aff031356cf/src/llama-vocab.cpp.
Unlike the Qwen35 adapter, GLM does not normalize Unicode to NFC and groups
digits by threes. Vision, audio, video and tool execution are not implemented.
"""
from collections.abc import Mapping
import json
import operator


GLM4_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


class GLM5NextGGUFTokenizer:
    """Bounded text-only adapter preserving the checkpoint's BPE and template.

    ``vocab_size`` includes GGUF's unused padding entries. ``encode`` adds no
    BOS/EOS; the official chat template itself supplies ``[gMASK]<sop>``.
    Decode's ``skip_special_tokens`` follows GGUF control-token attributes;
    these are coarser than the source tokenizer's ``special`` flags.
    """

    chat_template_options = {'reasoning_effort': 'max', 'clear_thinking': False}

    def __init__(self, metadata):
        from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

        if (metadata.get('tokenizer.ggml.model') != 'gpt2'
                or metadata.get('tokenizer.ggml.pre') != 'glm4'):
            raise NotImplementedError('GLM5 Next requires the distributed gpt2/glm4 tokenizer')
        tokens = metadata.get('tokenizer.ggml.tokens')
        merges = metadata.get('tokenizer.ggml.merges')
        kinds = metadata.get('tokenizer.ggml.token_type')
        if (not isinstance(tokens, list) or not tokens or len(tokens) > 300_000
                or any(not isinstance(token, str) for token in tokens)):
            raise ValueError('Missing or excessive GGUF vocabulary')
        if (not isinstance(merges, list) or len(merges) > 1_000_000
                or any(not isinstance(merge, str) for merge in merges)):
            raise ValueError('Missing or excessive GGUF BPE merges')
        if (not isinstance(kinds, list) or len(kinds) != len(tokens)
                or any(type(kind) is not int or kind not in (1, 3, 4, 5) for kind in kinds)):
            raise ValueError('Invalid GGUF token attributes')
        vocab = {token: index for index, token in enumerate(tokens)}
        if len(vocab) != len(tokens):
            raise ValueError('Duplicate GGUF vocabulary entry')
        pairs = []
        for merge in merges:
            pair = merge.split(' ')
            if len(pair) != 2 or any(token not in vocab for token in pair):
                raise ValueError('Invalid GGUF BPE merge')
            pairs.append(tuple(pair))
        self.engine = Tokenizer(models.BPE(vocab=vocab, merges=pairs, ignore_merges=True))
        # No normalizer: canonically equivalent Unicode strings need not have
        # identical model token IDs in GLM's checkpoint.
        self.engine.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(GLM4_PATTERN), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        self.engine.decoder = decoders.ByteLevel()
        self.engine.add_tokens([
            AddedToken(token, normalized=False, special=(kind == 3),
                       lstrip=False, rstrip=False, single_word=False)
            for token, kind in zip(tokens, kinds) if kind in (3, 4)])
        self.vocab_size = len(tokens)
        self.eos_token_id = self._id(metadata, 'eos_token_id')
        self.bos_token_id = self._id(metadata, 'bos_token_id', required=False)
        self.stop_token_ids = frozenset(
            value for value in [self.eos_token_id,
                               self._id(metadata, 'eot_token_id', required=False),
                               self._id(metadata, 'eom_token_id', required=False)] if value is not None)
        for name in ('add_bos_token', 'add_eos_token'):
            if metadata.get('tokenizer.ggml.' + name, False) is not False:
                raise NotImplementedError('Unexpected automatic GLM BOS/EOS insertion')
        self.chat_template = metadata.get('tokenizer.chat_template')
        if (not isinstance(self.chat_template, str) or not self.chat_template
                or len(self.chat_template.encode('utf-8')) > 1 << 20):
            raise ValueError('Missing or excessive embedded GLM chat template')
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
        self._template = env.from_string(self.chat_template)

    def _id(self, metadata, name, *, required=True):
        value = metadata.get('tokenizer.ggml.' + name)
        if value is None and not required:
            return None
        if type(value) is not int or not 0 <= value < self.vocab_size:
            raise ValueError(f'Invalid GLM {name}')
        return value

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Text must be a string')
        return self.engine.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        if type(skip_special_tokens) is not bool:
            raise TypeError('skip_special_tokens must be boolean')
        values = []
        for value in ids:
            if isinstance(value, bool):
                raise ValueError('Token IDs must be integers in the vocabulary')
            try:
                value = operator.index(value)
            except TypeError as exc:
                raise ValueError('Token IDs must be integers in the vocabulary') from exc
            if not 0 <= value < self.vocab_size:
                raise ValueError('Token ID is outside the GLM vocabulary')
            values.append(value)
        return self.engine.decode(values, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True,
                            reasoning_effort='max', clear_thinking=False,
                            preserve_thinking=None, enable_thinking=None):
        """Render the embedded official template with its native GLM options.

        GLM's template always opens ``<think>`` and offers reasoning_effort,
        not Qwen's enable_thinking=False option. ``preserve_thinking`` is an
        explicit inverse alias of GLM's clear_thinking for generic clients.
        """
        if any(type(flag) is not bool for flag in (tokenize, add_generation_prompt, clear_thinking)):
            raise TypeError('Chat template flags must be booleans')
        if enable_thinking is not None and enable_thinking is not True:
            raise NotImplementedError('The GLM checkpoint template cannot disable thinking')
        if preserve_thinking is not None:
            if type(preserve_thinking) is not bool:
                raise TypeError('preserve_thinking must be boolean or None')
            if clear_thinking and preserve_thinking:
                raise ValueError('Conflicting thinking-history options')
            clear_thinking = not preserve_thinking
        if reasoning_effort not in ('low', 'high', 'max'):
            raise ValueError('reasoning_effort must be low, high or max')
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError('At least one text message is required')
        clean = []
        for index, message in enumerate(messages):
            if (not isinstance(message, Mapping) or not {'role', 'content'} <= set(message)
                    or set(message) - {'role', 'content', 'reasoning_content'}
                    or message['role'] not in ('system', 'user', 'assistant')
                    or not isinstance(message['content'], str)):
                raise NotImplementedError('Only plain text system/user/assistant messages are supported')
            if message['role'] == 'system' and index != 0:
                raise ValueError('System message must be first')
            if 'reasoning_content' in message and (message['role'] != 'assistant'
                    or not isinstance(message['reasoning_content'], str)):
                raise NotImplementedError('reasoning_content must be assistant text')
            clean.append(dict(message))
        if not any(message['role'] == 'user' for message in clean):
            raise ValueError('A user query is required')
        text = self._template.render(messages=clean, tools=None,
                                     add_generation_prompt=add_generation_prompt,
                                     reasoning_effort=reasoning_effort,
                                     clear_thinking=clear_thinking)
        return self.encode(text) if tokenize else text
