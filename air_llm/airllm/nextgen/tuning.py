"""Measured read-only storage tuning; no implicit huge disk benchmark files."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
import os
import platform
from time import perf_counter
from pathlib import Path
import psutil
import resource
import sys
from ..ultra.checkpoint import atomic_json
from .config import NextConfig,GiB,MiB
from .storage import physical_read_bytes


def process_rss():
    # In nested PID namespaces, os.getpid() may not address the /proc mount's
    # namespace. /proc/self is resolved by the kernel and remains authoritative.
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):return int(line.split()[1])*1024
        except (OSError,ValueError):pass
        return None
    try:return psutil.Process().memory_info().rss
    except (psutil.Error,OSError):return None


def telemetry():
    vm=psutil.virtual_memory();swap=psutil.swap_memory()
    return {'available_bytes':vm.available,'total_bytes':vm.total,'rss_bytes':process_rss(),
            'swap_used_bytes':swap.used,'cpu_percent':psutil.cpu_percent(interval=None),
            'gpu_utilization':None,'temperature_c':None,'thermal_throttled':None,
            'memory_pressure_source':'available physical memory (not macOS pressure level)',
            'physical_read_bytes':physical_read_bytes(),
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'rss_peak_scope':'process lifetime, not isolated per model'}


def auto_config(profile='balanced',**kwargs):
    if profile not in ('balanced','performance'):raise ValueError('Unknown profile')
    vm=psutil.virtual_memory();physical=vm.total
    # Available memory, not just installed RAM: apps already open are respected.
    reserve=kwargs.get('reserve_bytes',min(4*GiB,int(physical*.3)))
    if type(reserve) is not int or reserve<0:raise ValueError('reserve_bytes must be a nonnegative integer')
    usable=max(0,min(int(physical*.6),vm.available-reserve))
    explicit_budgets=all(key in kwargs for key in ('cache_bytes','io_bytes','kv_bytes','workspace_bytes'))
    if usable<384*MiB and not explicit_budgets:
        raise MemoryError('Not enough currently available memory for inference; explicitly configure all cache/IO/KV/workspace budgets and reserve for a smaller workload')
    cache=int(usable*(.48 if profile=='balanced' else .55))
    options=dict(profile=profile,reserve_bytes=reserve,cache_bytes=cache,
       kv_bytes=min(768*MiB,int(usable*.15)),io_bytes=min(384*MiB,int(usable*.10)),
       workspace_bytes=min(GiB,int(usable*.20)),max_tensor_bytes=max(1,min(256*MiB,int(usable*.10))),
       io_workers=2 if profile=='balanced' else 3)
    options.update(kwargs)
    config=NextConfig(**options)
    requested=sum(getattr(config,key) for key in ('cache_bytes','io_bytes','kv_bytes','workspace_bytes'))
    if requested+config.reserve_bytes>vm.available:
        raise MemoryError('Requested inference budgets plus reserve exceed currently available memory')
    return config


def measure_storage(cp,config,samples=12):
    """Read up to ~12 MiB per candidate, process/cache state always disclosed."""
    matrices=[(k,d) for k,d in cp.tensors.items() if len(d['shape'])==2 and d['shape'][0]>0]
    if not matrices:raise ValueError('No matrices to benchmark')
    requests=[]
    for i in range(samples):
        k,d=matrices[(i*7919)%len(matrices)]
        row=(i*131)%d['shape'][0];count=min(d['shape'][0]-row,max(1,MiB//max(1,d['shape'][1]*4)))
        requests.append((k,(row,row+count)))
    candidates=[]
    for mode in ('pread','mmap'):
        for workers in (1,2,4):
            begin=perf_counter();before=cp.stats['logical_bytes'];physical=physical_read_bytes()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda x:cp.read_array(x[0],rows=x[1],mode=mode),requests))
            elapsed=perf_counter()-begin;n=cp.stats['logical_bytes']-before;end=physical_read_bytes()
            candidates.append({'mode':mode,'workers':workers,'seconds':elapsed,'logical_bytes':n,
              'logical_bytes_per_second':n/elapsed,'physical_read_bytes':end-physical if physical is not None and end is not None else None})
    # Only pread feeds the runtime. mmap numbers remain a diagnostic comparison.
    eligible=[v for v in candidates if v['mode']=='pread'];best=max(eligible,key=lambda v:v['logical_bytes_per_second'])
    result={'device':platform.platform(),'source_fingerprint':cp.fingerprint,'cache_state':'OS page cache uncontrolled; repeated reads can be RAM hits',
            'candidates':candidates,'selected_io_workers':best['workers'],'selected_mode':'pread',
            'physical_ssd_throughput_certified':False,'config':replace(config,io_workers=best['workers']).as_dict()}
    path=config.metadata_dir()/(cp.fingerprint+'.tuning.json');atomic_json(path,result)
    return result


def runtime_tuning_key(model):
    """Tie measured prefetch choice to model, sidecar, dtype and local machine."""
    manifest=None
    if model.options.quantized_dir:
        p=Path(model.options.quantized_dir)/'manifest.json'
        manifest=sha256(p.read_bytes()).hexdigest()
    identity={'engine_schema_version':5,'checkpoint':model.checkpoint.fingerprint,'quantized_manifest':manifest,
      'platform':platform.platform(),'machine':platform.machine(),'device':model.device,
      'dtype':model.options.dtype,'profile':model.options.profile,'prefill_chunk':model.options.prefill_chunk,
      'cache_bucket_256MiB':model.options.cache_bytes//(256*MiB),
      'io_bucket_64MiB':model.options.io_bytes//(64*MiB),
      'cache_policy':model.options.policy,'coalesce_reads':model.options.coalesce_reads,'selective_fp8_scales':model.options.selective_fp8_scales,
      'compact_cache_bytes':model.options.compact_cache_bytes,'output_tile_cache':model.options.output_tile_cache,
      'output_prefetch':model.options.output_prefetch,
      'io_mode':model.options.io_mode,'spec_prefetch':model.options.spec_prefetch,
      'spec_prefetch_budget':model.options.spec_prefetch_budget,'spec_prefetch_experts':model.options.spec_prefetch_experts}
    return sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def calibrate_prefetch(model,input_ids,tokens=8):
    """ABBA measurement, warmup excluded. Caller explicitly requests generation.

    A benefit smaller than 5% does not justify concurrent lookahead. No gain is
    asserted from IO alone: the entire actual generation wall time is measured.
    """
    import statistics
    reader=model.reader;old=reader.prefetch_enabled;rows=[]
    try:
        reader.prefetch_enabled=False;model.generate(input_ids,max_new_tokens=tokens,eos_token_id=[])
        for enabled in (False,True,True,False):
            reader.drain();reader.prefetch_enabled=enabled
            begin=perf_counter();model.generate(input_ids,max_new_tokens=tokens,eos_token_id=[])
            rows.append({'prefetch':enabled,'generation_seconds':perf_counter()-begin,
              'decode_tokens_s':model.stats['decode_tokens_per_second']})
        off=statistics.median(x['generation_seconds'] for x in rows if not x['prefetch'])
        on=statistics.median(x['generation_seconds'] for x in rows if x['prefetch'])
        choice=on<off*.95
        cache_report=calibrate_cache(reader.cache)
        report={'key':runtime_tuning_key(model),'measurements':rows,'warmup_excluded':True,
            'cache_policy':cache_report['selected_policy'],'cache_replay':cache_report,
            'prefetch_enabled':choice,'minimum_gain_fraction':.05,'tokens_per_request':tokens,
            'scope':'actual local checkpoint, short workload; sustained thermal validation separate'}
        atomic_json(model.options.metadata_dir()/(model.checkpoint.fingerprint+'.runtime-tuning.json'),report)
        reader.prefetch_enabled=choice
        return report
    except BaseException:
        reader.prefetch_enabled=old;raise


def calibrate_cache(cache):
    """Replay real requests and load costs; service sums are not wall-time gains.

    Unknown first-request costs use the actually observed subsequent load of the
    same matrix. All policies see identical sizes and service observations.
    """
    from .cache import WeightCache
    from .adaptive_cache import AdaptiveWeightCache
    from types import SimpleNamespace
    trace=list(cache.trace);rows=[]
    costs={}
    for key,seconds in cache.load_trace:
        costs.setdefault(key,[]).append(seconds)
    import statistics
    costs={key:statistics.median(values) for key,values in costs.items()}
    costs.update({key:value for key,value in cache.costs.items() if key not in costs})
    known=sum(key in costs for key,_ in trace)
    for policy in ('slru','lru','lfu','cost','adaptive'):
        cache_type=AdaptiveWeightCache if policy=='adaptive' else WeightCache
        candidate=cache_type(cache.capacity,policy);miss_bytes=0;service=0.;unknown=0
        for key,n in trace:
            if candidate.get(key,n) is None:
                miss_bytes+=n
                if key in costs:
                    service+=costs[key];candidate.record_load(key,costs[key])
                else:unknown+=1
                candidate.put(key,SimpleNamespace(nbytes=n))
        rows.append({'policy':policy,'replayed_requests':len(trace),'miss_weight_bytes':miss_bytes,
            'replayed_observed_service_seconds':service,'misses_without_cost_observation':unknown})
    # Preserve the existing byte-oriented auto-selection rule. Cost policy stays
    # explicit until end-to-end calibration demonstrates a latency improvement.
    eligible=[row for row in rows if row['policy'] not in ('cost','adaptive')]
    if cache.policy=='adaptive':eligible=[row for row in rows if row['policy']=='adaptive']
    best=min(eligible,key=lambda x:x['miss_weight_bytes'])['policy']
    return {'selected_policy':best,'scope':'recent actual weight requests and observed IO+conversion durations',
        'cost_observation_request_coverage':known/len(trace) if trace else 0.,
        'cost_policy_auto_enabled':False,'wall_time_gain_measured':False,'policies':rows}
