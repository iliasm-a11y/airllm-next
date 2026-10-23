"""Greedy, batch-one text conversations using the native GGUF Qwen backend.

This adapter does not load a model, launch a process, or relax the native
memory guards. Real checkpoints still require an independently supervised
process. The tokenizer must come from the checkpoint being evaluated.
"""
from threading import RLock
from time import perf_counter

import numpy as np

from .generation import eos_ids, generation_options, token_batch


class GGUFChatSession:
    """Keep actual generated history and reuse only an identical token prefix.

    ``preserve_thinking=True`` is the official template option that preserves
    the empty thought wrapper used for no-thinking generation. It makes normal
    multi-turn prompts appendable. If template stripping, Unicode normalization
    or BPE boundaries change an earlier token, the adapter rebuilds the native
    session instead of using an inconsistent recurrent/KV state.

    The last sampled token has not yet been forwarded through the model. It is
    included in the next turn's appended input, including an EOS token. This
    avoids an unnecessary model pass after every completed response.
    """

    def __init__(self, model, tokenizer, system_prompt=None, *, reuse_kv=True):
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError('system_prompt must be a string or None')
        if type(reuse_kv) is not bool:
            raise TypeError('reuse_kv must be a boolean')
        if tokenizer.vocab_size != model.config['vocab_size']:
            raise ValueError('Tokenizer vocabulary does not match the model')
        self.model = model
        self.tokenizer = tokenizer
        self.reuse_kv = reuse_kv
        self._messages = ([] if system_prompt is None else
                          [{'role': 'system', 'content': system_prompt}])
        self._cache = None
        self._cached_ids = []
        self._lock = RLock()

    @property
    def messages(self):
        """A defensive copy of the actual system/user/assistant history."""
        with self._lock:
            return [dict(message) for message in self._messages]

    def reply(self, prompt, *, max_new_tokens=128, temperature=0., on_token=None,
              **kwargs):
        """Generate one response; all returned fields are JSON serializable.

        Only greedy decoding (temperature zero) is implemented. Other options
        are rejected, rather than silently ignored. ``on_token(event)`` receives
        token_id, a zero-based token_index, elapsed_seconds and step_seconds.
        The first step includes prefill; subsequent steps measure decode.

        The whole requested output must fit the context limit before any model
        state is touched. On failure the unfinished turn is not committed and
        its native cache is discarded, so a retry rebuilds valid state.
        """
        if not isinstance(prompt, str):
            raise TypeError('prompt must be a string')
        if kwargs:
            raise TypeError(f'Unsupported GGUF chat generation options: {sorted(kwargs)}')
        count, temperature = generation_options(max_new_tokens, temperature)
        if count < 1:
            raise ValueError('max_new_tokens must be positive for a chat reply')
        if temperature != 0.:
            raise ValueError('GGUF chat supports greedy temperature=0 only')
        if on_token is not None and not callable(on_token):
            raise TypeError('on_token must be callable or None')

        with self._lock, self.model.lock:
            if self.model.closed:
                raise RuntimeError('Model is closed')
            proposed = self.messages + [{'role': 'user', 'content': prompt}]
            render_options = dict(add_generation_prompt=True,
                                  enable_thinking=False, preserve_thinking=True)
            prompt_text = self.tokenizer.apply_chat_template(
                proposed, tokenize=False, **render_options)
            full_ids = token_batch([self.tokenizer.encode(prompt_text)],
                                   self.model.config['vocab_size']).tolist()[0]
            limit = min(self.model.options.max_context,
                        self.model.config['max_position_embeddings'])
            if len(full_ids) + count > limit:
                raise MemoryError('GGUF chat prompt plus requested output exceeds context limit')
            stop_ids = eos_ids(self.model.config.get('eos_token_id'),
                               self.model.config['vocab_size'])
            if not stop_ids:
                stop_ids = eos_ids(self.tokenizer.eos_token_id,
                                   self.model.config['vocab_size'])

            prefix_matched = (self._cache is not None and
                              self._cache.position == len(self._cached_ids) and
                              full_ids[:len(self._cached_ids)] == self._cached_ids)
            reuse = self.reuse_kv and prefix_matched
            reused = len(self._cached_ids) if reuse else 0
            consumed = list(self._cached_ids) if reuse else []
            if not reuse:
                # Do not keep an obsolete session's arrays alive while a new
                # prefill allocates another complete set of recurrent/KV banks.
                self._cache = None
                self._cached_ids = []
            cache = self._cache if reuse else self.model.new_session()
            appended = full_ids[reused:]
            if not appended:
                raise ValueError('A chat turn must append at least one token')
            current = appended
            generated = []
            token_seconds = []
            first = None
            begin = perf_counter()
            try:
                for index in range(count):
                    step_begin = perf_counter()
                    output = self.model.forward(
                        np.array([current], dtype=np.int32),
                        past_key_values=cache, logits_to_keep=1)
                    cache = output.past_key_values
                    consumed.extend(current)
                    # argmax.item() synchronizes MLX before recording a token.
                    with self.model.mx.stream(self.model.stream):
                        if not bool(self.model.mx.all(self.model.mx.isfinite(output.logits)).item()):
                            raise FloatingPointError('Non-finite GGUF chat logits')
                        token = int(self.model.mx.argmax(output.logits[0, -1]).item())
                    now = perf_counter()
                    if first is None:
                        first = now
                    generated.append(token)
                    token_seconds.append(now - step_begin)
                    if on_token is not None:
                        on_token({'token_id': token, 'token_index': index,
                                  'elapsed_seconds': now - begin,
                                  'step_seconds': token_seconds[-1]})
                    if token in stop_ids:
                        break
                    current = [token]
                end = perf_counter()
                content_ids = generated[:-1] if generated[-1] in stop_ids else generated
                content = self.tokenizer.decode(content_ids, skip_special_tokens=False)
                text = self.tokenizer.decode(content_ids, skip_special_tokens=True)
                raw_text = self.tokenizer.decode(generated, skip_special_tokens=False)
            except BaseException:
                self._cache = None
                self._cached_ids = []
                raise

            self._messages = proposed + [{'role': 'assistant', 'content': content}]
            self._cache = cache if self.reuse_kv else None
            self._cached_ids = consumed if self.reuse_kv else []
            n = len(generated)
            ttft = first - begin
            decode_rate = (n - 1) / (end - first) if n > 1 and end > first else 0.
            if hasattr(self.model, 'stats'):
                self.model.stats.update(tokens_generated=n, generation_seconds=end - begin,
                                        ttft_seconds=ttft,
                                        decode_tokens_per_second=decode_rate,
                                        context_payload_peak_bytes=cache.peak_bytes)
            return {
                'messages': self.messages,
                'prompt_text': prompt_text,
                'full_input_ids': full_ids,
                'appended_input_ids': appended,
                'generated_ids': generated,
                'full_output_ids': full_ids + generated,
                'text': text,
                'raw_text': raw_text,
                'stop_reason': 'eos' if generated[-1] in stop_ids else 'max_new_tokens',
                'eos_token_ids': sorted(stop_ids),
                'token_seconds': token_seconds,
                'ttft_seconds': ttft,
                'decode_tokens_per_second': decode_rate,
                'total_seconds': end - begin,
                'reused_prefix_tokens': reused,
                'cached_prefix_matched': bool(prefix_matched),
                'native_cache_position': cache.position,
                'context_limit': limit,
                'finite_logits_checked': True,
                'generation_parameters': {
                    'max_new_tokens': count, 'temperature': temperature,
                    'do_sample': False, 'enable_thinking': False,
                    'preserve_thinking': True, 'reuse_kv': self.reuse_kv,
                },
            }
