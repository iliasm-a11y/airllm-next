"""Use the shared bounded Metal timestamp adapter for Q8 compatible kernels."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np


def measure(timer, directory, packed, width, x, *, fused, shared=False, repetitions=5):
    if fused and shared:
        raise ValueError('select fused inline or shared activation')
    if shared and width != 4096:
        raise ValueError('shared activation requires width4096')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    kernels = Path(__file__).parent / 'kernels_q8'
    buffers = []
    def input_buffer(name, values):
        path = root/(name+'.bin')
        with path.open('wb') as target:
            target.write(memoryview(values).cast('B'))
        buffers.append({'bytes': values.nbytes, 'input': str(path.resolve())})
    input_buffer('packed', packed)
    input_buffer('x', x)
    output = root/'output.bin'
    buffers.append({'bytes': packed.shape[0]*4, 'output': str(output.resolve())})
    sources = ['#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n']
    stages = []
    if shared:
        body = (kernels/'q8_matvec_shared.metal').read_text()
        sources.append('kernel void q8_shared(device const uchar* packed [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n'+f'const uint packed_shape[2]={{{packed.shape[0]},{packed.shape[1]}}};\n'+body+'\n}\n')
        stages.append({'entry':'q8_shared','bindings':[0,1,2],'grid':[(packed.shape[0]+7)//8*32,1,1],'threadgroup':[32,1,1]})
    elif fused:
        sources.append((kernels/'q8_inline_helpers.metal').read_text())
        body = (kernels/'q8_matvec_inline.metal').read_text().replace('WIDTH', str(width))
        sources.append('kernel void q8_fused(device const uchar* packed [[buffer(0)]],device const float* x [[buffer(1)]],device float* out [[buffer(2)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n'+f'const uint packed_shape[2]={{{packed.shape[0]},{packed.shape[1]}}};\n'+body+'\n}\n')
        stages.append({'entry': 'q8_fused', 'bindings': [0, 1, 2], 'grid': [packed.shape[0]*4, 1, 1], 'threadgroup': [32, 1, 1]})
    else:
        buffers += [{'bytes': width}, {'bytes': width//32*2}]
        body = (kernels/'q8_activation.metal').read_text()
        sources.append('kernel void q8_activation(device const float* x [[buffer(0)]],device char* qs [[buffer(1)]],device ushort* scale_bits [[buffer(2)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n'+f'const uint x_shape[1]={{{width}}};\n'+body+'\n}\n')
        body = (kernels/'q8_matvec.metal').read_text().replace('WIDTH', str(width))
        sources.append('kernel void q8_dot(device const uchar* packed [[buffer(0)]],device const char* qs [[buffer(1)]],device const ushort* scale_bits [[buffer(2)]],device float* out [[buffer(3)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n'+f'const uint packed_shape[2]={{{packed.shape[0]},{packed.shape[1]}}};\n'+body+'\n}\n')
        stages += [{'entry': 'q8_activation', 'bindings': [1, 3, 4], 'grid': [width, 1, 1], 'threadgroup': [32, 1, 1]},
                   {'entry': 'q8_dot', 'bindings': [0, 3, 4, 2], 'grid': [packed.shape[0]*4, 1, 1], 'threadgroup': [32, 1, 1]}]
    source = root/'kernels.metal'
    source.write_text(''.join(sources))
    manifest = root/'job.json'
    manifest.write_text(json.dumps({'source': str(source.resolve()), 'buffers': buffers, 'stages': stages,
                                    'repetitions': repetitions}, indent=2)+'\n')
    result = subprocess.run([str(Path(timer).resolve()), str(manifest.resolve())], text=True,
                            capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Metal timestamp probe failed: '+result.stderr[:2000])
    report = json.loads(result.stdout)
    report['declared_static_threadgroup_bytes'] = 4608 if shared else 0
    report.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Command-buffer device timestamps on prepared small Q8 matrix; no SSD or GLM throughput claim')
    return report, np.fromfile(output, dtype=np.float32)
