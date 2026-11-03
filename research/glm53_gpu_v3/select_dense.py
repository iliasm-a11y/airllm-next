"""Metadata-only fixed dense retention plan; never reads checkpoint payloads."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

REVISION='119034bc3d9d76eb41f91508e6f9fb19fde42672'
MODEL_SHA={
 'GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf':'22e3599ff1cac02e854269b610030800e319151e1ed01838d7087f20b187091d',
 'GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf':'61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4'}
FAMILIES={'hc_attn_fn.weight','hc_ffn_fn.weight','ffn_gate_inp.weight','ssm_beta.weight',
    'ssm_f_a.weight','ssm_f_b.weight','ssm_g_a.weight','ssm_g_b.weight','attn_k_b.weight',
    'attn_v_b.weight','attn_kv_a_mqa.weight','indexer.attn_q_b.weight','indexer.attn_k.weight',
    'indexer.proj.weight','indexer_compressor_gate.weight'}
CODECS={'F32':0,'BF16':30,'Q8_0':8}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat(path):
    s=Path(path).stat()
    return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def rows(path):
    path=Path(path)
    if path.stat().st_size>128*1024**2:raise ValueError('metadata input exceeds128MiB')
    with path.open() as f:
        for count,line in enumerate(f):
            if count>1000000:raise ValueError('too many metadata rows')
            if line.strip():yield json.loads(line)


def union(spans):
    groups=defaultdict(list)
    for file,start,end in spans:groups[file].append((start,end))
    merged=[]
    for file,ranges in sorted(groups.items()):
        out=[]
        for start,end in sorted(ranges):
            if out and start<=out[-1][1]:out[-1]=(out[-1][0],max(end,out[-1][1]))
            else:out.append((start,end))
        merged.extend((file,start,end) for start,end in out)
    return merged


def page_cover(start,length,page):
    if any(type(v) is not int for v in (start,length,page)) or start<0 or length<=0 or page<=0:
        raise ValueError('invalid page interval')
    end=start+length
    return start-start%page,end+(page-end%page)%page


def make_plan(args):
    inventory=json.loads(args.inventory.read_text())
    if inventory['revision']!=REVISION:raise ValueError('wrong pinned model revision')
    expected={t['name']:t for t in inventory['tensors'] if '_exps.' not in t['name'] and
        t['name']!='token_embd.weight' and not t['name'].endswith('indexer_compressor_ape.weight') and
        (len(t['dims'])==2 or t['name'].endswith(('attn_k_b.weight','attn_v_b.weight')))}
    if len(expected)!=684 or sum(t['bytes_with_alignment'] for t in expected.values())!=9444458496:
        raise ValueError('fixed dense inventory does not match684matrices/9444458496bytes')
    ordered=[];source_buffers={}
    for row in rows(args.source_plan):
        if row.get('step')!=args.step or row.get('op')!='MUL_MAT':continue
        source=row.get('source_weight',{});name=source.get('base_name')
        if name not in expected:
            if source.get('buffer_usage')==1:raise ValueError('unexpected original weight in source plan')
            continue
        tensor=expected[name]
        if (source.get('buffer_usage')!=1 or source.get('alias_resolved') is not True or
            source.get('base_op')!='NONE' or source.get('offset_from_base')!=0 or
            source.get('bytes')!=tensor['bytes_with_alignment'] or source.get('type')!=CODECS[tensor['type']]):
            raise ValueError('source descriptor does not prove a complete original matrix: '+name)
        ordered.append(tensor)
        source_buffers[name]=source.get('buffer_name')
    if Counter(t['name'] for t in ordered)!=Counter({name:1 for name in expected}):
        raise ValueError('source step must contain exactlyone ofeach684densematrix')
    times=defaultdict(list);totals=defaultdict(float);sites=defaultdict(list)
    for row in rows(args.operations):
        step=row['step']
        if step<=0:continue
        totals[step]+=row['seconds']
        if row['op']=='MUL_MAT' and row.get('weight_pages_before',{}).get('status')=='ok':sites[step].append(row)
    for step,records in sorted(sites.items()):
        if len(records)!=684:raise ValueError('old profile does not contain684denseoperations')
        for tensor,record in zip(ordered,records):
            if record['weight_pages_before']['logical_selected_weight_bytes']!=tensor['bytes_with_alignment']:
                raise ValueError('oldsite/newsource ordered byte sizes disagree')
            cost=record['seconds']
            if not math.isfinite(cost) or cost<=0:raise ValueError('invalid measured operation wall cost')
            times[tensor['name']].append(cost)
    if not times or len(sites)!=3:raise ValueError('expected three old CPU4 decode steps')
    page=os.sysconf('SC_PAGESIZE');budget=args.budget_mib*1024**2
    files={f['file']:f for f in inventory['files']};candidates=[]
    for name,tensor in expected.items():
        family=name.split('.',2)[-1]
        if tensor['type'] not in ('F32','BF16') or family not in FAMILIES:continue
        if source_buffers[name]!='CPU_Mapped':raise ValueError('eligible fixed source is not original CPU_Mapped: '+name)
        start=files[tensor['file']]['data_offset']+tensor['offset'];end=start+tensor['bytes_with_alignment']
        page_start,page_end=page_cover(start,tensor['bytes_with_alignment'],page)
        candidates.append({'name':name,'file':tensor['file'],'type':tensor['type'],'dimensions':tensor['dims'],
            'tensor_offset':tensor['offset'],'absolute_start':start,'payload_bytes':tensor['bytes_with_alignment'],
            'page_start':page_start,'page_end':page_end,
            'old_profile_mean_op_wall_seconds':sum(times[name])/len(times[name])})
    chosen=[];spans=[];used=0
    while candidates and len(chosen)<128:
        best=None
        for row in candidates:
            trial=union(spans+[(row['file'],row['page_start'],row['page_end'])]);size=sum(b-a for _,a,b in trial)
            marginal=size-used
            if size>budget:continue
            key=(float('inf') if marginal==0 else row['old_profile_mean_op_wall_seconds']/marginal,
                 row['old_profile_mean_op_wall_seconds'],row['name'])
            if best is None or key>best[0]:best=(key,row,trial,size)
        if best is None:break
        _,row,spans,used=best;chosen.append(row);candidates.remove(row)
    names=sorted(row['name'] for row in chosen)
    if not names or len(','.join(names).encode())>16384:raise ValueError('nameplan is empty or oversized')
    if used>budget:raise ValueError('pageunion exceeded the strict budget')
    expected_files={f['file'] for f in inventory['files']}
    current={p.name:stat(p) for p in args.model_directory.glob('*.gguf')}
    if set(current)!=expected_files:raise ValueError('model file set differs from pinned inventory')
    for name,f in files.items():
        if current[name]['size']!=f['file_bytes']:raise ValueError('current shard size differs from inventory')
    identity_sha=sha(args.identity_manifest)
    if args.identity_sha256 and identity_sha!=args.identity_sha256:raise ValueError('prior verification manifest digest differs')
    prior=json.loads(args.identity_manifest.read_text())
    if prior.get('integrity_verified') is not True or prior.get('revision')!=REVISION or prior.get('quantization')!='Q2_K_S':raise ValueError('prior hash evidence is not verified at pinned revision/quantization')
    digests={};identity_changes={};identity_complete=True
    aliases={'device':('device','dev','st_dev'),'inode':('inode','ino','st_ino'),'size':('size','st_size'),
             'mtime_ns':('mtime_ns','st_mtime_ns'),'ctime_ns':('ctime_ns','st_ctime_ns')}
    for f in prior['files']:
        if f.get('integrity_verified') is not True or f.get('actual_sha256')!=f.get('sha256'):
            raise ValueError('prior expected/actual SHA differs')
        digests[f['name']]=f['actual_sha256']
        old=f.get('identity',{})
        if isinstance(old,list) and len(old)==5:old=dict(zip(aliases,old))
        if not isinstance(old,dict):raise ValueError('prior identity must be five values or a field mapping')
        changes={}
        for key,options in aliases.items():
            value=next((old[k] for k in options if k in old),None)
            if value is None:identity_complete=False
            elif value!=current[f['name']][key]:changes[key]={'prior':value,'current':current[f['name']][key]}
        if changes:identity_changes[f['name']]=changes
    if digests!=MODEL_SHA or set(digests)!=expected_files:raise ValueError('prior hashes are not the original pinned checkpoint')
    total_wall=sum(totals.values())/len(totals);selected_wall=sum(r['old_profile_mean_op_wall_seconds'] for r in chosen)
    return {'schema_version':1,'feature':'bounded_dense_cpu_mlock_v1','model_revision':REVISION,
        'budget_mib':args.budget_mib,'page_size':page,'names':names,'selected_raw_bytes':sum(r['payload_bytes'] for r in chosen),
        'selected_page_union_bytes':used,'selected_tensors':sorted(chosen,key=lambda r:r['name']),
        'page_unions':[{'file':file,'start':a,'end':b} for file,a,b in spans],
        'model_file_stats':current,'model_sha256':digests,'identity_manifest_path':str(args.identity_manifest.resolve()),
        'identity_manifest_sha256':identity_sha,'prior_integrity_checked_at':prior['checked_at'],
        'model_sha256_scope':'prior manifest verified SHA; selector never rehashes current checkpoint payload; historical/current identity comparison is separate',
        'historical_identity_changes':identity_changes,'historical_full_identity_equal':not identity_changes if identity_complete else None,
        'current_full_hash_reverified':False,'inventory_sha256':sha(args.inventory),
        'source_alignment':{'source_plan_sha256':sha(args.source_plan),'operations_sha256':sha(args.operations),'source_step':args.step,
            'names_count':684,'old_decode_steps':sorted(sites),'ordered_matrix_sizes_verified':True,
            'scope':'oldsite timing attached to newsource identity by unchanged graphorder+exactsizecheck; not simultaneous pointer/name timing proof'},
        'cost_model':{'method':'deterministic greedy observed opwall per marginal full page, <=128names; not global knapsack optimum',
            'profile_mean_total_op_wall_seconds':total_wall,'selected_mean_op_wall_seconds_upper_bound':selected_wall,
            'zero_selected_op_wall_throughput_gain_percent_upper_bound':100*selected_wall/(total_wall-selected_wall),
            'scope':'old serialized diagnostic, including compute and wait. Cache cannot eliminate all selected opwall; no measured gain or current residency claim.'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inventory','source-plan','operations','model-directory','identity-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-identity-manifest-sha256','--identity-sha256',dest='identity_sha256')
    p.add_argument('--budget-mib',type=int,choices=(128,256,512),default=256)
    p.add_argument('--step',type=int,default=1);args=p.parse_args()
    if args.output.exists():p.error('refusing existing plan')
    result=make_plan(args);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
