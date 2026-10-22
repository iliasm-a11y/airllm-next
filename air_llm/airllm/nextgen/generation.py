"""Shared public-input checks, before narrowing ids or touching model state."""
import math
import numbers
import numpy as np


def nonnegative_integer(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def token_batch(input_ids, vocab_size=None):
    if hasattr(input_ids, 'detach'):
        input_ids = input_ids.detach().cpu().numpy()
    ids = np.asarray(input_ids)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Expected nonempty unpadded integer input_ids with batch size one')
    # Check before casting: uint64/int64 narrowing must never wrap into a
    # different valid token, and NumPy negative indexing is not a tokenizer.
    if ids.min() < 0 or ids.max() > np.iinfo(np.int32).max:
        raise ValueError('Token ids must be nonnegative int32 values')
    if vocab_size is not None and ids.max() >= vocab_size:
        raise ValueError('Input token id is outside the target vocabulary')
    return ids.astype(np.int32, copy=False)


def eos_ids(value, vocab_size=None):
    if value is None:
        return frozenset()
    if isinstance(value, numbers.Integral):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        values = value.tolist()
    else:
        raise ValueError('eos_token_id must be an integer or a collection of integers')
    result = set()
    for token in values:
        token = nonnegative_integer(token, 'eos_token_id')
        if token > np.iinfo(np.int32).max or (vocab_size is not None and token >= vocab_size):
            raise ValueError('eos_token_id is outside the target vocabulary')
        result.add(token)
    return frozenset(result)


def generation_options(max_new_tokens, temperature):
    count = nonnegative_integer(max_new_tokens, 'max_new_tokens')
    if (not isinstance(temperature, numbers.Real) or isinstance(temperature, (bool, np.bool_))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError('temperature must be finite and nonnegative')
    return count, float(temperature)


def logits_suffix_start(input_length, logits_to_keep):
    keep = nonnegative_integer(logits_to_keep, 'logits_to_keep')
    return max(0, input_length - keep) if keep else 0


def check_logits_budget(positions, vocab_size, workspace_bytes):
    # A floating head can produce FP32 even with a reduced compute dtype. The
    # concatenated result temporarily coexists with its per-chunk/tile pieces.
    # This caps the requested output alone; attention has its own scratch check.
    if 2 * positions * vocab_size * 4 > workspace_bytes:
        raise MemoryError('Requested logits exceed workspace_bytes; reduce logits_to_keep or input length')
