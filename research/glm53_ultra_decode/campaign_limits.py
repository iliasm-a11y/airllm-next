"""An optional common swap baseline can only tighten each experiment's cap."""
MAX_SWAP_GROWTH = 32 << 20


def remaining_swap_budget(sample, baseline=None):
    if baseline is None:
        return MAX_SWAP_GROWTH
    if type(baseline) is not int or baseline < 0:
        raise ValueError('campaign swap baseline must be a nonnegative byte count')
    used = sample.get('swap_used_bytes')
    if type(used) is not int or used < 0:
        raise MemoryError('campaign_swap_measurement_unavailable')
    growth = max(0, used - baseline)
    if growth > MAX_SWAP_GROWTH:
        raise MemoryError('campaign_swap_growth_above_32MiB')
    # If swap falls, never grant more than the original per-experiment cap.
    return min(MAX_SWAP_GROWTH, MAX_SWAP_GROWTH - growth)
