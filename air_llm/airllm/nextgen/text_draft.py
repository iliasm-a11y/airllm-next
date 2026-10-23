"""Conservative local greedy proposals between different tokenizers.

This is text re-encoding, not a stochastic heterogeneous-vocabulary sampler.
The verifier checks every proposed target token; an unstable prefix boundary
returns no proposal. Draft prefill is recomputed and its full cost is included.
"""
from collections import Counter
import numpy as np


class ReencodedModelDraft:
    def __init__(self, model, target_tokenizer, *, max_prefix_tokens=2048):
        if type(max_prefix_tokens) is not int or max_prefix_tokens < 1:
            raise ValueError('max_prefix_tokens must be a positive integer')
        if getattr(model, 'tokenizer', None) is None:
            raise ValueError('Draft requires a local tokenizer')
        self.model, self.target_tokenizer = model, target_tokenizer
        self.checkpoint = model.checkpoint
        self.max_prefix_tokens = max_prefix_tokens
        self.stats = Counter()

    def __call__(self, prefix, count):
        if type(count) is not int or count < 0:
            raise ValueError('Proposal length must be a nonnegative integer')
        if not count:return []
        text = self.target_tokenizer.decode(prefix, skip_special_tokens=False)
        # Exact input token identity is checked before consulting the draft.
        if self.target_tokenizer.encode(text, add_special_tokens=False) != prefix:
            self.stats['unstable_prefix'] += 1
            return []
        ids = self.model.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > self.max_prefix_tokens:
            self.stats['prefix_limit'] += 1
            return []
        remaining = min(self.model.options.max_context,
                        self.model.config['max_position_embeddings'])-len(ids)
        if remaining < 1:
            self.stats['context_limit'] += 1
            return []
        output = self.model.generate(np.asarray([ids], dtype=np.int32),
            max_new_tokens=min(count,remaining), temperature=0.).tolist()[0]
        if output[:len(ids)] != ids:
            raise ValueError('Draft did not preserve its prefix')
        continuation = self.model.tokenizer.decode(output[len(ids):], skip_special_tokens=True)
        retokenized = self.target_tokenizer.encode(text+continuation, add_special_tokens=False)
        self.stats['calls'] += 1
        self.stats['draft_tokens'] += len(output)-len(ids)
        if retokenized[:len(prefix)] != prefix:
            self.stats['unstable_seam'] += 1
            return []
        result = retokenized[len(prefix):len(prefix)+count]
        self.stats['target_candidates'] += len(result)
        return result
