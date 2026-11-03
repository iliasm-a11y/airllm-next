"""Source model of V4 CPU fallback eligibility; no arithmetic/fidelity claim.

Only ordinary 2D MUL_MAT with a two-row ARM vec-dot kernel is represented.
K-dependent arithmetic, MUL_MAT_ID, alternative kernels and runtime dispatch
are outside this structural model. The measured shape inventory is external.
"""
from __future__ import annotations
from functools import lru_cache


def ceildiv(a, b):
    return (a + b - 1) // b


@lru_cache(maxsize=4096)
def layout(m, n, nth=10, numa=False):
    if any(type(v) is not int or v <= 0 for v in (m, n, nth)) or type(numa) is not bool:
        raise ValueError('positive exact 2D shape/thread integers required')
    chunk = 64 if m == 1 or n == 1 else 16
    nc0, nc1 = ceildiv(m, chunk), ceildiv(n, chunk)
    thread_rechunk = nc0 * nc1 < nth * 4 or numa
    if thread_rechunk:
        nc0, nc1 = (nth, 1) if m > n else (1, nth)
    dr0, dr1 = ceildiv(m, nc0), ceildiv(n, nc1)
    rows = [(start, min(start + dr0, m)) for start in range(0, m, dr0)]
    columns = [(start, min(start + dr1, n)) for start in range(0, n, dr1)]
    # Each entry is the exact nrc choice for one row interval. The row
    # intervals are fixed for this call; no set union can hide a changed row.
    by_column = []
    for first, last in columns:
        row_choices = tuple((a, b, 2 if m % 2 == 0 and n % 2 == 0 and
                             (b - a) % 2 == 0 and (last - first) % 2 == 0 else 1)
                            for a, b in rows)
        # Merge equal adjacent row choices, yielding a canonical nrc map.
        merged = []
        for a, b, nrc in row_choices:
            if merged and merged[-1][1] == a and merged[-1][2] == nrc:
                merged[-1] = (merged[-1][0], b, nrc)
            else:
                merged.append((a, b, nrc))
        by_column.extend([tuple(merged)] * (last - first))
    return {'chunk_size': chunk, 'nchunk0': nc0, 'nchunk1': nc1,
            'dr0': dr0, 'dr1': dr1, 'thread_rechunk': thread_rechunk,
            'nrc_by_column': tuple(by_column)}


def schedule_map(m, batches, nth=10, numa=False):
    return tuple(value for batch in batches
                 for value in layout(m, batch['tokens'], nth, numa)['nrc_by_column'])
