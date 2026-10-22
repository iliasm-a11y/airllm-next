"""Bounded prompt lookup drafts. Every emitted token still needs target verification.

This uses an established retrieval-drafting idea, not a substitute language
model or a claim of a new general acceleration algorithm.
"""
class PromptSpanDraft:
    def __init__(self, *, max_lookback=8192, max_match=64, min_match=3):
        for n in (max_lookback,max_match,min_match):
            if type(n) is not int or n < 1:
                raise ValueError('Prompt lookup limits must be positive integers')
        if min_match > max_match:
            raise ValueError('min_match exceeds max_match')
        self.max_lookback=max_lookback
        self.max_match=max_match
        self.min_match=min_match

    def __call__(self, prefix, count):
        if type(count) is not int or count < 0:
            raise ValueError('Draft count must be nonnegative')
        if not count:
            return []
        # Only tokens provided by the caller are used. No output oracle, model
        # download, provider, tokenizer conversion, or target state is consulted.
        tokens=prefix[-self.max_lookback:]
        for n in range(min(self.max_match,len(tokens)-1),self.min_match-1,-1):
            suffix=tokens[-n:]
            for i in range(len(tokens)-n-1,-1,-1):
                if tokens[i:i+n]==suffix:
                    return list(tokens[i+n:min(i+n+count,len(tokens))])
        return []
