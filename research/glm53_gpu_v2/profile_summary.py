"""Summarize serialized native operation diagnostics; never equate pages to SSD IO."""
import argparse
import json
import math
from pathlib import Path
import re


def _new():
    return {'operations':0,'op_wall_seconds':0.0,'inspection_seconds_excluded':0.0,
            'cpu_seconds':0.0,'cpu_counter_operations':0,'os_read_bytes':0,'io_counter_operations':0,
            'pageins':0,'pagein_counter_operations':0,'logical_selected_weight_bytes':0,
            'residency_operations_ok':0,'residency_operations_error':0,'residency_operations_other':0,
            'inspected_pages_snapshot_sum':0,'resident_pages_snapshot_sum':0,'nonresident_pages_snapshot_sum':0,
            'page_sizes':set(),'residency_status_counts':{}}


def _number(row,key,default=None):
    value=row.get(key,default)
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError(f'invalid nonnegative finite metric {key}')
    return value


def _add(total,row):
    total['operations']+=1
    total['op_wall_seconds']+=_number(row,'seconds',0)
    for key,count in [('cpu_seconds','cpu_counter_operations'),('os_read_bytes','io_counter_operations'),
                      ('pageins','pagein_counter_operations')]:
        value=_number(row,key)
        if value is not None:total[key]+=value;total[count]+=1
    pages=row.get('weight_pages_before')
    if pages is None:return
    status=pages.get('status','missing')
    label=status+('/'+pages['reason'] if pages.get('reason') else '')
    total['residency_status_counts'][label]=total['residency_status_counts'].get(label,0)+1
    total['inspection_seconds_excluded']+=_number(pages,'inspection_seconds',0)
    total['logical_selected_weight_bytes']+=_number(pages,'logical_selected_weight_bytes',0)
    if status=='ok':
        fields=[_number(pages,k) for k in ['inspected_pages','resident_pages','nonresident_pages']]
        if None in fields or any(type(v) is not int for v in fields) or fields[0]!=fields[1]+fields[2]:
            raise ValueError('invalid residency page accounting')
        total['residency_operations_ok']+=1
        for key,value in zip(['inspected_pages_snapshot_sum','resident_pages_snapshot_sum','nonresident_pages_snapshot_sum'],fields):
            total[key]+=value
        size=_number(pages,'page_size')
        if type(size) is not int or size<1:raise ValueError('invalid page size')
        total['page_sizes'].add(size)
    elif status in ('error','unsupported'):
        total['residency_operations_error']+=1
        # Failure is unknown, not zero resident/nonresident pages.
        if any(pages.get(k) is not None for k in ('inspected_pages','resident_pages','nonresident_pages')):
            raise ValueError('failed residency inspection must report null page metrics')
    else:total['residency_operations_other']+=1


def _finish(total):
    out={**total,'page_sizes':sorted(total['page_sizes'])}
    for key,count in [('cpu_seconds','cpu_counter_operations'),('os_read_bytes','io_counter_operations'),
                      ('pageins','pagein_counter_operations')]:
        if not total[count]:out[key]=None
    wall=out['op_wall_seconds']
    out['cpu_seconds_per_op_wall_second']=(out['cpu_seconds']/wall if out['cpu_seconds'] is not None and wall>0 else None)
    inspected=out['inspected_pages_snapshot_sum']
    out['resident_fraction_of_successful_snapshots']=(out['resident_pages_snapshot_sum']/inspected if inspected else None)
    # Counts are not unique pages and may repeat the same weight pages per op.
    if not total['residency_operations_ok']:
        for key in ('inspected_pages_snapshot_sum','resident_pages_snapshot_sum','nonresident_pages_snapshot_sum'):
            out[key]=None
    return out


def summarize(rows):
    total=_new();groups={'by_op':{},'by_layer':{},'by_step':{},'by_phase':{},'by_op_layer_step':{}}
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('op'),str) or type(row.get('step')) is not int or row['step']<0:
            raise ValueError('operation row requires op and nonnegative integer step')
        _add(total,row)
        match=re.search(r'-(\d+)(?:\s*\([^)]*\))?$',row.get('name',''))
        layer=match.group(1) if match else 'unlayered'
        keys={'by_op':row['op'],'by_layer':layer,'by_step':str(row['step']),
              'by_phase':'prefill' if row['step']==0 else 'decode',
              'by_op_layer_step':f'{row["op"]}/layer={layer}/step={row["step"]}'}
        for name,key in keys.items():_add(groups[name].setdefault(key,_new()),row)
    return {'schema':1,'scope':'serialized operation callback diagnostics; not production throughput',
        'totals':_finish(total),**{name:{k:_finish(v) for k,v in group.items()} for name,group in groups.items()},
        'interpretation':[
            'operation wall time includes arithmetic, memory access, page-fault wait and backend scheduling',
            'CPU seconds are summed across threads; CPU/wall ratio may exceed one and is not an IO percentage',
            'residency is inspected before the operation timer/counter snapshots, without dereferencing weights',
            'selected logical bytes and repeated residency snapshots are not physical disk traffic or unique pages',
            'nonresident pages may be satisfied from macOS file cache; they are not proven NAND reads',
            'per-process OS storage read deltas and pageins remain separate counters',
            'step0 is the initial prompt prefill in this helper; later steps decode one forced/generated token',
            'captures serialize the graph and add overhead outside operation timers; compare guarded decode separately']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operations',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    with args.operations.open() as f:
        report=summarize(json.loads(line) for line in f if line.strip())
    if not report['totals']['operations']:
        raise ValueError('empty operation profile; enable AIRLLM_PROFILE_OPS/PAGES=1')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['totals'],indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
