"""Default-off policy prototype; no hook into production graph placement.

Only whole-model fidelity and guarded repeated total costs can authorize a GPU
group. Warm component evidence can never satisfy this gate.
"""
from dataclasses import dataclass
import math
import re
import statistics

@dataclass(frozen=True)
class Group:
    format:str
    rows:int
    width:int
    banks:int=1

def positive(values):
    return bool(values) and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values)

def select(group,evidence,*,enabled=False,runtime_sha256='',resident=False,
           available_bytes=0,pressure_level=None,swap_growth_bytes=0):
    result={'backend':'cpu','reason':'disabled','group':group.__dict__}
    if not enabled:return result
    def refuse(reason):return {**result,'reason':reason}
    if group.format not in ('Q2_K','Q8_0') or not 1<=group.rows<=8192 or not 256<=group.width<=16384 or group.width%256 or group.banks not in (1,2):
        return refuse('unsupported_shape')
    payload=group.rows*(group.width//(256 if group.format=='Q2_K' else 32))*(84 if group.format=='Q2_K' else 34)*group.banks
    if payload>24<<20:return refuse('payload_budget')
    if not resident:return refuse('residency_unproven')
    if pressure_level!=1 or available_bytes<3*(1<<30) or swap_growth_bytes>32*(1<<20):
        return refuse('resource_gate')
    if not re.fullmatch('[0-9a-f]{64}',runtime_sha256) or evidence.get('runtime_sha256')!=runtime_sha256:
        return refuse('runtime_identity')
    if evidence.get('scope')!='whole_model_decode':return refuse('component_is_not_model_fidelity')
    if not all(evidence.get(k) is True for k in ('full_logits_passed','ordered_routes_passed',
            'recurrent_states_passed','original_thresholds_passed','guard_completed','weights_unchanged')):
        return refuse('fidelity_or_guard_gate')
    if type(evidence.get('captured_steps')) is not int or evidence['captured_steps']<32:
        return refuse('insufficient_captured_trajectory')
    if evidence.get('group')!=group.__dict__:return refuse('group_identity')
    cycles=evidence.get('abba_cycles',[])
    if len(cycles)<2:return refuse('insufficient_repetitions')
    ratios=[]
    for cycle in cycles:
        if cycle.get('order')!=['A','B','B','A'] or not positive(cycle.get('total_seconds',[])) or len(cycle['total_seconds'])!=4:
            return refuse('invalid_abba_total_costs')
        a,b0,b1,a1=cycle['total_seconds'];ratios.append(((a+a1)/2)/((b0+b1)/2))
    # Require a useful margin in every cycle. Total cost must already include
    # quantization, import, launch and synchronization for this atomic group.
    if min(ratios)<1.10 or evidence.get('all_boundary_costs_included') is not True:
        return refuse('net_gain_or_boundary_costs')
    return {**result,'backend':'gpu','reason':'experimental_whole_model_evidence',
            'median_ratio':statistics.median(ratios),
            'scope':'explicit experimental group only; never a production activation'}
