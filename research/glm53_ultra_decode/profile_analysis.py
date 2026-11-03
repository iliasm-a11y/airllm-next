"""Decode interval accounting and conditional physical bounds, never fake IO attribution."""
from __future__ import annotations
import json
import math
from pathlib import Path
from research.glm53_gpu_v2.profile_summary import summarize

def finite_nonnegative(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative metric required')
    return value

def conditional_bound(logical_bytes,covered_bytes,bandwidth_bytes_per_second,compute_seconds,overlap):
    for v in (logical_bytes,covered_bytes,bandwidth_bytes_per_second,compute_seconds):finite_nonnegative(v)
    if covered_bytes>logical_bytes or bandwidth_bytes_per_second<=0 or type(overlap) is not bool:
        raise ValueError('invalid conditional bound assumptions')
    io=max(0,logical_bytes-covered_bytes)/bandwidth_bytes_per_second
    lower=max(io,compute_seconds) if overlap else io+compute_seconds
    return {'assumed_service_bytes_per_token':logical_bytes-covered_bytes,
        'minimum_io_seconds':io,'minimum_token_seconds':lower,
        'conditional_upper_tokens_per_second':1/lower if lower else None,
        'assumptions':{'coverage_bytes':covered_bytes,'interface_bytes_per_second':bandwidth_bytes_per_second,
            'compute_seconds':compute_seconds,'perfect_overlap':overlap},
        'scope':'conditional model; coverage/service bandwidth are supplied assumptions, not measured NAND'}

def analyze_capture(root,*,diagnostic):
    root=Path(root)
    complete=json.loads((root/'complete.json').read_text())
    if complete.get('completed') is not True:raise ValueError('incomplete capture')
    tokens=[json.loads(line) for line in (root/'tokens.jsonl').read_text().splitlines() if line]
    times=[json.loads(line) for line in (root/'timing.jsonl').read_text().splitlines() if line]
    if len(tokens)!=complete.get('captured_steps') or not tokens:raise ValueError('missing emitted IDs')
    if any(row.get('step')!=i for i,row in enumerate(tokens)):raise ValueError('nonconsecutive token IDs')
    decode=[r for r in times if r['phase']=='decode']
    # First output comes from prefill. N emitted IDs require N-1 decode calls.
    if len(decode)!=len(tokens)-1:raise ValueError('decode interval count differs from emitted IDs minus first output')
    if any(row.get('step')!=i+1 for i,row in enumerate(decode)):raise ValueError('decode timing steps incomplete')
    seconds=sum(finite_nonnegative(r['decode_call_seconds']) for r in decode)
    if decode and seconds<=0:raise ValueError('zero decode time')
    out={'output_ids':len(tokens),'decode_intervals':len(decode),'decode_call_seconds':seconds,
        'decode_tokens_per_second':len(decode)/seconds if seconds and not diagnostic else None,
        'diagnostic_decode_intervals_per_second':len(decode)/seconds if seconds and diagnostic else None,
        'natural_eog':tokens[-1].get('eog') is True,
        'denominator':'N-1 completed autoregressive decode calls for N emitted IDs; prefill/first output excluded',
        'physical_nand_bytes':None,'allocation_bytes':None,'pure_io_wait_seconds':None}
    for field in ('os_read_bytes','os_write_bytes','pageins','minor_faults','major_faults','cpu_core_seconds'):
        out[field]=sum(finite_nonnegative(row[field]) for row in decode) if decode and all(field in row for row in decode) else None
    ops=root/'operations.jsonl'
    if diagnostic and ops.is_file():
        out['serialized_operations']=summarize(json.loads(line) for line in ops.read_text().splitlines() if line)
    return out
