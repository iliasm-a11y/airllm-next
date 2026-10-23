"""Text-only Qwen35 BPE from the vocabulary actually embedded in GGUF.

Regex and AddedToken conventions follow Qwen's official tokenizer_config.json.
No weights, tokenizer assets, or executable remote code are downloaded here.
Multimodal/tool chat rendering is deliberately rejected by this small adapter.
"""

QWEN35_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


class GGUFTokenizer:
    def __init__(self, metadata):
        from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
        if metadata.get('tokenizer.ggml.model') != 'gpt2' or metadata.get('tokenizer.ggml.pre') != 'qwen35':
            raise NotImplementedError('Only the Qwen35 GGUF tokenizer is supported')
        tokens = metadata.get('tokenizer.ggml.tokens')
        merges = metadata.get('tokenizer.ggml.merges')
        if not isinstance(tokens, list) or not tokens or len(tokens) > 300_000:
            raise ValueError('Missing or excessive GGUF vocabulary')
        if not isinstance(merges, list) or len(merges) > 1_000_000:
            raise ValueError('Missing or excessive GGUF merges')
        if any(not isinstance(t, str) for t in tokens + merges):
            raise ValueError('GGUF tokens and merges must be strings')
        vocab = {token: i for i, token in enumerate(tokens)}
        if len(vocab) != len(tokens):
            raise ValueError('Duplicate vocabulary entry')
        parsed = []
        for merge in merges:
            pair = merge.split(' ')
            if len(pair) != 2 or any(t not in vocab for t in pair):
                raise ValueError('Invalid GGUF BPE merge')
            parsed.append(tuple(pair))
        self.engine = Tokenizer(models.BPE(vocab=vocab, merges=parsed))
        self.engine.normalizer = normalizers.NFC()
        self.engine.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        self.engine.decoder = decoders.ByteLevel()
        kinds = metadata.get('tokenizer.ggml.token_type')
        if not isinstance(kinds, list) or len(kinds) != len(tokens):
            raise ValueError('Missing GGUF token types for AddedToken boundaries')
        additions = [AddedToken(token, normalized=False, special=(kind == 3),
                                lstrip=False, rstrip=False, single_word=False)
                     for token, kind in zip(tokens, kinds) if kind in (3, 4)]
        self.engine.add_tokens(additions)
        self.vocab_size = len(tokens)
        self.eos_token_id = metadata.get('tokenizer.ggml.eos_token_id')
        if type(self.eos_token_id) is not int or not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError('Invalid EOS ID')

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Text must be a string')
        return self.engine.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.engine.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False, preserve_thinking=False):
        """Render the official Qwen3.6 text-only conversation contract.

        ``enable_thinking=False`` is an explicit adapter default. The official
        template opens a thinking block when it is enabled. Historical thinking
        is omitted by that template unless ``preserve_thinking=True``; the latter
        also makes completed assistant turns retain the generation prefix.
        """
        from collections.abc import Mapping

        if not messages:
            raise ValueError('At least one text message is required')
        if any(type(flag) is not bool for flag in
               (tokenize, add_generation_prompt, enable_thinking, preserve_thinking)):
            raise TypeError('Chat template flags must be booleans')
        for index, message in enumerate(messages):
            if (not isinstance(message, Mapping) or not {'role', 'content'} <= set(message)
                    or set(message) - {'role', 'content', 'reasoning_content'}
                    or message['role'] not in ('system', 'user', 'assistant')
                    or not isinstance(message['content'], str)):
                raise NotImplementedError('This adapter supports plain text system/user/assistant messages only')
            if message['role'] == 'system' and index != 0:
                raise ValueError('System message must be at the beginning')
            if 'reasoning_content' in message and (message['role'] != 'assistant'
                    or not isinstance(message['reasoning_content'], str)):
                raise NotImplementedError('reasoning_content must be text on an assistant message')

        # Match the official last-query rule even when tool-response delimiters
        # happen to occur in text. Actual tool messages remain unsupported.
        user_indices = [i for i, message in enumerate(messages)
                        if message['role'] == 'user'
                        and not (message['content'].strip().startswith('<tool_response>')
                                 and message['content'].strip().endswith('</tool_response>'))]
        if not user_indices:
            raise ValueError('No user query found in messages')
        last_query_index = user_indices[-1]
        pieces = []
        for index, message in enumerate(messages):
            content = message['content'].strip()
            pieces.append(f"<|im_start|>{message['role']}\n")
            if message['role'] == 'assistant':
                reasoning = message.get('reasoning_content', '')
                if 'reasoning_content' not in message and '</think>' in content:
                    reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                    content = content.split('</think>')[-1].lstrip('\n')
                if preserve_thinking or index > last_query_index:
                    pieces.append(f'<think>\n{reasoning.strip()}\n</think>\n\n')
            pieces.append(content + '<|im_end|>\n')
        if add_generation_prompt:
            pieces.append('<|im_start|>assistant\n')
            pieces.append('<think>\n' if enable_thinking else '<think>\n\n</think>\n\n')
        text = ''.join(pieces)
        return self.encode(text) if tokenize else text
