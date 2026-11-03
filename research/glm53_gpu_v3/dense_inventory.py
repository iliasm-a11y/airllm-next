"""Saved-data reconciliation only: no model, checkpoint payload, device or build."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path


def selected_matrices(inventory):
    selected=[]
    for tensor in inventory['tensors']:
        name=tensor['name']
        if '_exps.' in name or name=='token_embd.weight' or name.endswith('indexer_compressor_ape.weight'):
            continue
        if len(tensor['dims'])==2 or name.endswith(('attn_k_b.weight','attn_v_b.weight')):
            selected.append(tensor)
    return selected


def analyze(inventory_path,operations_path):
    inventory=json.loads(Path(inventory_path).read_text());selected=selected_matrices(inventory)
    dense=sum(t['bytes_with_alignment'] for t in selected)
    expected=Counter(t['bytes_with_alignment'] for t in selected)
    ops=[json.loads(line) for line in Path(operations_path).open()]
    decode_steps=sorted({r['step'] for r in ops if r['step']>0})
    per_step=[];buckets=defaultdict(lambda:dict(calls=0,wall_seconds=0,process_os_read_bytes=0,pageins=0))
    for step in decode_steps:
        rows=[r for r in ops if r['step']==step and r['op']=='MUL_MAT' and r.get('weight_pages_before',{}).get('status')=='ok']
        observed=Counter(r['weight_pages_before']['logical_selected_weight_bytes'] for r in rows)
        per_step.append({'step':step,'calls':len(rows),'logical_dense_bytes':sum(size*count for size,count in observed.items()),
                         'inventory_size_multiset_matches':observed==expected})
        for r in rows:
            size=r['weight_pages_before']['logical_selected_weight_bytes'];b=buckets[size]
            b['calls']+=1;b['wall_seconds']+=r['seconds'];b['process_os_read_bytes']+=r['os_read_bytes'];b['pageins']+=r['pageins']
    families=defaultdict(lambda:dict(matrices=0,packed_bytes=0))
    for t in selected:
        family=t['name'].split('.',2)[-1] if t['name'].startswith('blk.') else t['name']
        families[family]['matrices']+=1;families[family]['packed_bytes']+=t['bytes_with_alignment']
    # Expert banks are equal-size in this pinned checkpoint; route identity still varies.
    expert_bytes=42*8*8257536
    total=dense+expert_bytes
    bounds=[]
    for mib in (0,512,1024,2048,3072):
        retained=mib*1024**2;fraction=retained/total
        bounds.append({'retained_mib':mib,'retained_bytes':retained,
                       'optimistic_linear_IO_time_reduction_fraction':fraction,
                       'optimistic_linear_IO_throughput_gain_percent':100*fraction/(1-fraction),
                       'remaining_logical_bytes':total-retained})
    return {'schema_version':1,'model_revision':inventory['revision'],
        'input_sha256':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (inventory_path,operations_path)},
        'dense_matrix_count':len(selected),'dense_selected_packed_bytes':dense,'selected_expert_bytes_per_decode':expert_bytes,
        'total_selected_matmul_bytes_per_decode':total,'dense_fraction':dense/total,
        'matrices':[{k:t[k] for k in ('name','file','offset','dims','type','bytes_with_alignment')} for t in selected],
        'families':dict(sorted(families.items(),key=lambda x:-x[1]['packed_bytes'])),
        'saved_profile_per_step_reconciliation':per_step,
        'profile_cost_by_matrix_byte_size':[{'matrix_bytes':size,**row} for size,row in sorted(buckets.items(),key=lambda x:-x[1]['wall_seconds'])],
        'partial_retention_bounds':bounds,
        'required_byte_retention_for_20percent_linear_IO_throughput_gain':total/6,
        'scope':[
            'Unique fixed source-matrix inventory inferred from owning graph source; saved profile independently matches the count and byte-size multiset, not source pointer identities.',
            'Exclude token embedding whole table: decode selects one row. Exclude indexer compressor APE: ADD rather than MUL_MAT. Include all 22 three-dimensional MLA K/V projection matrices.',
            'Profile operation wall includes calculation, paging and scheduling. Process OS reads and pageins are not physical NAND reads or isolated wait time.',
            'Retention bounds assume every retained byte eliminates equal-cost IO and no extra copies/memory pressure; they are optimistic conditional models, not measurements.',
            'No repeated dense matrix within one token is demonstrated. The major potential reuse is between tokens; bounded lookahead changes timing rather than required dense byte volume.'
        ]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--operations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('refusing to overwrite evidence')
    report=analyze(args.inventory,args.operations);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
