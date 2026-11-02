"""Export exactly the MLX variant body to bounded standalone Metal timing jobs."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from .q2_kernels import source_for


def emit_job(directory, packed, width, qs, scales, variant, *, repetitions=5, simd_groups=1):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    experts, rows, _ = packed.shape
    count = qs.size//width
    blocks = width//256
    buffers = []
    def add_input(name, value):
        path=root/f'{name}.bin'
        # Memoryview avoids an additional whole packed-matrix bytes allocation.
        with path.open('wb') as output:
            output.write(memoryview(value).cast('B'))
        buffers.append({'bytes': value.nbytes, 'input': str(path.resolve())})
        return len(buffers)-1
    p=add_input('packed',packed)
    q=add_input('qs',qs)
    s=add_input('scales',scales)
    output=root/'output.bin'
    buffers.append({'bytes':experts*rows*4, 'output':str(output.resolve())})
    o=3
    entries=[]
    sources=['#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n']
    types={'packed':'uchar','qs':'char','scales':'float','dots':'int','mins':'int','out':'float'}
    def stage(name, ins, outs, bindings, grid, group, tile=1, groups=1):
        arguments=[]
        for i, item in enumerate(ins+outs):
            const='const ' if item in ins else ''
            arguments.append(f'device {const}{types[item]}* {item} [[buffer({i})]]')
        arguments.append('uint3 thread_position_in_grid [[thread_position_in_grid]]')
        body=f'const uint packed_shape[3] = {{{experts},{rows},{packed.shape[2]}}};\n'
        body+=source_for(name,width,count,tile,groups).replace('uint8_t','uchar')
        entry=f'q2_{name}'
        sources.append('kernel void '+entry+'('+','.join(arguments)+') {\n'+body+'\n}\n')
        entries.append({'entry':entry,'bindings':bindings,'grid':grid,'threadgroup':group})
    if variant=='twopass':
        buffers += [{'bytes':experts*rows*blocks*4}]*2
        stage('partials',['packed','qs'],['dots','mins'],[p,q,4,5],
              [rows*32,blocks,experts],[32*simd_groups,1,1])
        stage('ordered_reduce',['packed','dots','mins','scales'],['out'],[p,4,5,s,o],
              [rows,experts,1],[32,1,1])
    else:
        tile={'rowtile2':2,'rowtile4':4,'cachedbytes_tile2':2,'cachedbytes_tile4':4}.get(variant,1)
        groups={'blocks4':4,'blocks8':8}.get(variant,1)
        name='coalesced' if variant.startswith('cachedbytes') else ('serial' if groups==1 else 'subgroups')
        stage(name,['packed','qs','scales'],['out'],[p,q,s,o],
              [(rows+tile-1)//tile*32,experts,1],[32*simd_groups,1,1],tile,groups)
    source=root/'kernels.metal'
    source.write_text(''.join(sources))
    manifest=root/'job.json'
    manifest.write_text(json.dumps({'source':str(source.resolve()),'buffers':buffers,
                                    'stages':entries,'repetitions':repetitions},indent=2))
    return manifest, output, hashlib.sha256(source.read_bytes()).hexdigest()


def measure_device(timer, directory, packed, width, qs, scales, variant, *, repetitions=5, simd_groups=1):
    manifest, output, source_hash=emit_job(directory,packed,width,qs,scales,variant,
                                         repetitions=repetitions,simd_groups=simd_groups)
    process=subprocess.run([str(Path(timer).resolve()),str(manifest.resolve())],capture_output=True,text=True,
                           timeout=60,check=False)
    if process.returncode:
        raise RuntimeError(f'Metal device timer failed: {process.stderr[:2000]}')
    report=json.loads(process.stdout)
    result=np.fromfile(output,dtype=np.float32).reshape(packed.shape[:2])
    report['generated_source_sha256']=source_hash
    return report,result


def measure_device_suite(timer,directory,packed,width,qs,scales,variants,*,repetitions=7):
    """Shared buffers/process; randomized command order; paired GPU/wall samples."""
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    experts,rows,_=packed.shape;count=qs.size//width;blocks=width//256
    buffers=[]
    for name,value in [('packed',packed),('qs',qs),('scales',scales)]:
        path=root/f'{name}.bin'
        with path.open('wb') as f:f.write(memoryview(value).cast('B'))
        buffers.append({'bytes':value.nbytes,'input':str(path.resolve())})
    outputs={}
    for variant in variants:
        path=root/f'{variant}.bin';outputs[variant]=path
        buffers.append({'bytes':experts*rows*4,'output':str(path.resolve())})
    dot_index=len(buffers);min_index=dot_index+1
    buffers += [{'bytes':experts*rows*blocks*4}]*2
    sources=['#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n']
    experiments=[]
    types={'packed':'uchar','qs':'char','scales':'float','dots':'int','mins':'int','out':'float'}
    for vi,variant in enumerate(variants):
        stages=[]
        def stage(name,ins,outs,bindings,grid,group,tile=1,groups=1):
            arguments=[f'device {"const " if item in ins else ""}{types[item]}* {item} [[buffer({i})]]'
                       for i,item in enumerate(ins+outs)]
            arguments+=['uint3 thread_position_in_grid [[thread_position_in_grid]]']
            body=f'const uint packed_shape[3] = {{{experts},{rows},{packed.shape[2]}}};\n'
            body+=source_for(name,width,count,tile,groups).replace('uint8_t','uchar')
            entry=f'q2_{variant}_{name}'
            sources.append('kernel void '+entry+'('+','.join(arguments)+') {\n'+body+'\n}\n')
            stages.append({'entry':entry,'bindings':bindings,'grid':grid,'threadgroup':group})
        output=3+vi
        if variant=='twopass':
            stage('partials',['packed','qs'],['dots','mins'],[0,1,dot_index,min_index],
                  [rows*32,blocks,experts],[32,1,1])
            stage('ordered_reduce',['packed','dots','mins','scales'],['out'],[0,dot_index,min_index,2,output],
                  [rows,experts,1],[32,1,1])
        else:
            tile={'rowtile2':2,'rowtile4':4,'cachedbytes_tile2':2,'cachedbytes_tile4':4}.get(variant,1)
            groups={'blocks4':4,'blocks8':8}.get(variant,1)
            name='coalesced' if variant.startswith('cachedbytes') else ('serial' if groups==1 else 'subgroups')
            stage(name,['packed','qs','scales'],['out'],[0,1,2,output],
                  [(rows+tile-1)//tile*32,experts,1],[32,1,1],tile,groups)
        experiments.append({'name':variant,'stages':stages})
    source=root/'suite.metal';source.write_text(''.join(sources))
    manifest=root/'suite.json';manifest.write_text(json.dumps({'source':str(source.resolve()),
        'buffers':buffers,'experiments':experiments,'repetitions':repetitions,'warmups':4},indent=2))
    process=subprocess.run([str(Path(timer).resolve()),str(manifest.resolve())],capture_output=True,text=True,timeout=60)
    if process.returncode:raise RuntimeError(f'Metal suite failed: {process.stderr[:2500]}')
    report=json.loads(process.stdout)
    actual={variant:np.fromfile(path,dtype='f4').reshape(experts,rows) for variant,path in outputs.items()}
    report['generated_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    return report,actual
