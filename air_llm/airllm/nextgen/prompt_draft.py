"""Bounded prompt lookup drafts. Every emitted token still needs target verification.

This uses an established retrieval-drafting idea, not a substitute language
model or a claim of a new general acceleration algorithm.
"""
class PromptSpanDraft:
    # This implementation only searches the caller's in-memory transcript.
    # Explicit counters let the verifier distinguish known zero transport from
    # a user callable whose storage traffic cannot be observed.
    @staticmethod
    def logical_bytes_read():return 0

    @staticmethod
    def encoded_bytes_read():return 0

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
        if not isinstance(tokens,list):tokens=list(tokens)
        match_limit=min(self.max_match,len(tokens)-1)
        if match_limit < self.min_match:return []
        # Preserve the old fast case for a full match ending just before the
        # suffix (repeated prose/patterns). Eight bounded slice checks avoid
        # constructing the linear index for these common successful lookups.
        suffix=tokens[-match_limit:]
        for distance in range(1,min(8,len(tokens)-match_limit)+1):
            end=len(tokens)-distance
            if tokens[end-match_limit:end]==suffix:
                return tokens[end:min(end+count,len(tokens))]
        # Z values on the reversed transcript give the exact common suffix
        # length at every earlier ending in O(lookback) token comparisons.
        # This avoids repeated slices for every candidate and match length.
        # No hashes, collision assumptions or generated-token oracle are used.
        reverse=tokens[::-1];z=[0]*len(reverse);left=right=0
        best=0;end=None
        for i in range(1,len(reverse)):
            if i<=right:z[i]=min(right-i+1,z[i-left])
            while i+z[i]<len(reverse) and reverse[z[i]]==reverse[i+z[i]]:
                z[i]+=1
            if i+z[i]-1>right:left,right=i,i+z[i]-1
            n=min(match_limit,z[i])
            # Iterating earlier endings newest first preserves the previous
            # longest-match, newest-occurrence tie rule, including overlaps.
            if n>best:best=n;end=len(tokens)-i
            if best==match_limit:break
        if best<self.min_match:return []
        return tokens[end:min(end+count,len(tokens))]
