"""Read pristine pinned GGML sources and the separately fingerprinted V4 tree."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from research.glm53_ultra_decode.build import REVISION, sha

BASE = '6cfa76930626520c15246286b3baa2a58c0cf952'
REPO = Path(__file__).resolve().parents[2]
ANCHORS = {
    'ggml/src/ggml-metal/ggml-metal-device.cpp':[
        ('native Q8/Q2 dispatch',r'case GGML_TYPE_Q8_0:',23)],
    'ggml/src/ggml-metal/ggml-metal-device.m':[
        ('opportunistic offload threshold',r'op_offload_min_batch_size\s*=',2),
        ('mHC support constraints',r'case GGML_OP_DSV4_HC_POST:',15),
        ('KDA and matrix support constraints',r'case GGML_OP_GATED_DELTA_NET:',19)],
    'ggml/src/ggml-metal/ggml-metal.cpp':[
        ('buffer backend compatibility',r'static bool ggml_backend_metal_device_supports_buft',11),
        ('opportunistic matrix offload',r'static bool ggml_backend_metal_device_offload_op',9)],
    'ggml/src/ggml-metal/ggml-metal-common.cpp':[
        ('few-row MMA constraints',r'bool ggml_metal_op_mul_mat_use_mma\(',16)],
    'ggml/src/ggml-metal/ggml-metal-ops.cpp':[
        ('native graph matrix dispatch',r'^int ggml_metal_op_mul_mat\(',22)],
    'ggml/src/ggml-metal/kernels/mul_mv.metal':[
        ('Q8 native FP32 activation',r'void kernel_mul_mv_q8_0_f32_impl\(',73),
        ('Q2 native FP32 activation',r'void kernel_mul_mv_q2_K_f32_impl\(',103),
        ('Q2 ID entry',r'host_name\("kernel_mul_mv_id_q2_K_f32"',1)],
    'ggml/src/ggml-cpu/ggml-cpu.c':[
        ('CPU Q8 activation and accumulation',r'\[GGML_TYPE_Q8_0\] =',11),
        ('CPU Q2 requires Q8_K',r'\[GGML_TYPE_Q2_K\] =',7)],
    'src/models/glm5-next.cpp':[
        ('mHC fused graph',r'^ggml_tensor \* llama_model_glm5_next::graph::build_hc_post\(',22),
        ('KDA dependencies',r'^ggml_tensor \* llama_model_glm5_next::graph::build_kda_layer\(',25)],
}


def run(source, restored_v4, output):
    source,restored_v4,output = (Path(p).resolve() for p in (source,restored_v4,output))
    output.mkdir(parents=True,exist_ok=False)
    pristine, excerpts = {}, []
    for name, anchors in ANCHORS.items():
        data = subprocess.check_output(['git','-C',str(source),'show',REVISION+':'+name])
        lines = data.decode().splitlines()
        evidence = []
        for label, pattern, count in anchors:
            positions = [i for i,line in enumerate(lines) if re.search(pattern,line)]
            if not positions:
                raise ValueError('missing pinned anchor: '+name+': '+label)
            start = positions[0]; end = min(start+count,len(lines))
            evidence.append({'label':label,'first_line':start+1,'last_line':end})
            excerpts.append(name+' @ '+REVISION+' / '+label+'\n'+'\n'.join(
                f'{i+1}: {lines[i]}'.rstrip() for i in range(start,end)))
        pristine[name] = {'revision':REVISION,'sha256':hashlib.sha256(data).hexdigest(),'evidence':evidence}
    manifest_path = REPO/'research/glm53_gpu_v3/evidence/v4-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    restored = {name:(restored_v4/name).is_file() and sha(restored_v4/name)==row['sha256'] for name,row in
        manifest['native_modified_and_new_files'].items()}
    for name,pattern,count in (
        ('src/llama-context.cpp',r'^static void airllm_glm_q8_only_assign\(',52),
        ('ggml/src/ggml-metal/ggml-metal-device.cpp',r'const char \* cpu_arith =',29),
    ):
        data=(restored_v4/name).read_bytes();lines=data.decode().splitlines()
        start=next(i for i,line in enumerate(lines) if re.search(pattern,line))
        excerpts.append(name+' / V4 restored, not pristine upstream\n'+'\n'.join(
            f'{i+1}: {lines[i]}'.rstrip() for i in range(start,min(start+count,len(lines)))))
    local_selections = {
        'air_llm/airllm/nextgen/glm_native.py':[
            ('production CPU command',r'def _build_command\(',42)],
    }
    local = {}
    for name,anchors in local_selections.items():
        data=(REPO/name).read_bytes(); lines=data.decode().splitlines()
        local[name]={'sha256':hashlib.sha256(data).hexdigest()}
        for label,pattern,count in anchors:
            start=next(i for i,line in enumerate(lines) if re.search(pattern,line))
            excerpts.append(name+' / '+label+'\n'+'\n'.join(
                f'{i+1}: {lines[i]}'.rstrip() for i in range(start,min(start+count,len(lines)))))
    weights_path = REPO/'research/glm53_m5/sources/gguf-headers.json'
    dense_path = REPO/'research/glm53_gpu_v3/evidence/dense-inventory.json'
    cache_path = REPO/'research/glm53_gpu_v2/evidence/cache-policy-phase-replay.json'
    inventory = json.loads(weights_path.read_text())
    dense = json.loads(dense_path.read_text())
    cache = json.loads(cache_path.read_text())
    codecs=defaultdict(lambda:{'matrices':0,'logical_bytes':0})
    for row in dense['matrices']:
        codecs[row['type']]['matrices']+=1
        codecs[row['type']]['logical_bytes']+=row['bytes_with_alignment']
    d,m,total=(dense[key] for key in ('dense_selected_packed_bytes','selected_expert_bytes_per_decode','total_selected_matmul_bytes_per_decode'))
    if d+m != total or sum(c['logical_bytes'] for c in codecs.values()) != d:
        raise ValueError('inconsistent saved logical inventory')
    q2 = [r for r in inventory['tensors'] if r['type']=='Q2_K']
    shader=subprocess.check_output(['git','-C',str(source),'show',REVISION+':ggml/src/ggml-metal/kernels/mul_mv.metal']).decode()
    report = {'schema':1,'utc':datetime.now(timezone.utc).isoformat(),
        'airllm_base':BASE,'llama_revision':REVISION,
        'scope':'source and historical metadata audit only; no device execution or new model fidelity',
        'pristine_sources':pristine,'local_sources':local,'v4_source_manifest_sha256':sha(manifest_path),
        'v4_current_source_matches':restored,'all_v4_sources_match':all(restored.values()),
        'native_entrypoints_present':{name:name in shader for name in (
            'kernel_mul_mv_q2_K_f32','kernel_mul_mv_q8_0_f32',
            'kernel_mul_mv_id_q2_K_f32','kernel_mul_mv_id_q8_0_f32')},
        'checkpoint_tensor_types':dict(Counter(r['type'] for r in inventory['tensors'])),
        'q2_banks':len(q2),'q2_banks_all_experts':all('_exps.weight' in r['name'] for r in q2),
        'historical_dense_codecs':dict(codecs),
        'logical_bytes_per_decode':{'dense':d,'selected_experts':m,'total':total},
        'conditional_perfect_moe_cache_io_speedup':total/d,
        'bound_assumptions':'uniform service cost per logical byte, unchanged accesses, zero cache overhead; not a bound on NAND or complete model runtime',
        'historical_decode_cache_replay':[{'policy':r['policy'],'capacity_bytes':r['capacity_bytes'],
            **r['by_phase']['decode']} for r in cache['results']],
        'historical_cache_scope':cache['scope'],
        'cross_agent_cache_reuse_measured_this_run':False,
        'new_pipeline_implemented':False,'production_changed':False,
        'saved_evidence_sha256':{str(p.relative_to(REPO)):sha(p) for p in (weights_path,dense_path,cache_path)},
        'phase_order':'native complete-model fidelity and speed precede new inter-agent cache; stable memory precedes pipeline'}
    (output/'source-excerpts.txt').write_text('\n\n'.join(excerpts)+'\n')
    write_path=output/'audit.json'
    write_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_v4_sources_match':all(restored.values()),'native_kernels_present':report['native_entrypoints_present'],
        'dense_codecs':dict(codecs),'q2_banks':len(q2)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--restored-v4',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source,args.restored_v4,args.output)
