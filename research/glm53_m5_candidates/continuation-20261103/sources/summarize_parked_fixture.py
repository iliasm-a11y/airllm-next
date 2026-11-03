"""Describe closed GPU component evidence without inferring model qualification."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import psutil
import re
import statistics

WORK=Path(__file__).resolve().parent
ROOT=WORK/'metal-tensorops-q8-parked'
path=ROOT/'fixtures-warm-v2/run2/receipt.json'
d=json.loads(path.read_text())
assert d['status']=='passed_component_contract_only' and d['full_model_qualified'] is False
assert d['power_gate']['passed'] is True and d['source_identity_unchanged'] is True
rows=[];cleanup=[];trace_totals={'gpu_seconds':0.,'copy_weight_us':0.,'command_and_readback_us':0.}
commands=publications=0
for pair in d['results']:
    assert pair['expectation_met'] is True
    cpu,gpu=pair['processes']
    for job in pair['processes']:
        assert job['cleanup']['owned_server_stopped'] is True
        owner=job['owned_identity']
        try:
            p=psutil.Process(owner['pid'])
            same=abs(p.create_time()-owner['create_time'])<.01 and p.exe()==owner['executable']
        except psutil.NoSuchProcess:same=False
        assert not same,'Owned component lifetime still exists'
        cleanup.append({'pid':owner['pid'],'create_time':owner['create_time'],'same_owned_lifetime_absent':True})
    if pair['case']['expectation']=='gpu':
        a,b=cpu['reuse'],gpu['reuse']
        raw=pair['numerical_comparison']
        assert raw['cpu_bitexact'] and raw['all_finite']
        assert gpu['gpu_trace']['published_operations']==5
        row={'case':pair['case'],'cpu_first_s':a['first_seconds'],'gpu_first_s':b['first_seconds'],
             'cpu_warm_s':a['warm_seconds'],'gpu_warm_s':b['warm_seconds'],
             'cpu_warm_median_s':a['warm_median_seconds'],'gpu_warm_median_s':b['warm_median_seconds'],
             'gpu_over_cpu_warm_ratio':b['warm_median_seconds']/a['warm_median_seconds'],
             'raw_comparison':raw,'native_trace':gpu['gpu_trace']}
        rows.append(row)
        log=path.parent/pair['case']['id']/'bridge-on/stderr.log'
        for line in log.read_text().splitlines():
            if 'AIRLLM_CPU_METAL_Q8 ' not in line:continue
            trace=dict(re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)',line))
            if trace.get('event')=='tile_done':
                commands+=1
                for key in trace_totals:trace_totals[key]+=float(trace[key])
            publications+=trace.get('event')=='published'
max_growth=0
for line in (path.parent/'samples.jsonl').read_text().splitlines():
    sample=json.loads(line);max_growth=max(max_growth,sample['swap_growth_bytes'])
result={'schema_version':1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
    'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'status':d['status'],'profile_sha256':d['profile_sha256'],'binary_sha256':d['binary_sha256'],
    'case_count':len(d['results']),'normal_gpu_pairs':len(rows),'per_graph_completed_iterations':5,
    'matched_final_float32_outputs':sum(p['numerical_comparison']['outputs'] for p in d['results'] if p.get('numerical_comparison')),
    'completed_gpu_commands':d['gpu_commands'],'published_gpu_operations':publications,
    'completed_normal_gpu_commands':commands,'normal_gpu_trace_total':trace_totals,
    'expected_late_fatal':d['results'][-1]['processes'][1]['fatal'],
    'rss_cap_bytes':d['child_rss_ceiling_bytes'],'peak_sampled_child_rss_bytes':d['observed_peak_child_rss_bytes'],
    'peak_sampled_controller_rss_bytes':d['observed_peak_controller_rss_bytes'],
    'metal_device_current_allocated_peak_bytes':d['device_allocated_observed_peak_bytes'],
    'global_swap_growth_sampled_peak_bytes':max_growth,'power_passed':d['power_gate']['passed'],
    'source_identity_unchanged':d['source_identity_unchanged'],'all_owned_lifetimes_absent':True,
    'closed_owned_processes':cleanup,'elapsed_seconds':d['measured_elapsed_seconds'],
    'gpu_over_cpu_warm_ratio_range':[min(r['gpu_over_cpu_warm_ratio'] for r in rows),max(r['gpu_over_cpu_warm_ratio'] for r in rows)],
    'excluded_resource_shapes':d['excluded_resource_shapes'],
    'cpu_bitexact_contract_met':d['cpu_bitexact_contract_met'],
    'full_model_qualified':False,'functional_agents_qualified':False,'performance_qualified':False,
    'decision':'do not launch expensive full GLM with this slower provider; retained experimental OFF by default',
    'scope':'Metal4 TensorOps int8xint8-int32; same resident synthetic weights, four warm graph repetitions after a separately reported first operation; no SSD/GLM/cache/EOS claim',
    'normal_gpu_results':rows}
output=WORK/'gpu-parked-fixture-summary.json'
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('normal_gpu_results','closed_owned_processes','expected_late_fatal')}))
