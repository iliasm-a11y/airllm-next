#!/usr/bin/env python3
"""Generate an additive shared-activation head patch; no runtime mutation."""
from pathlib import Path
import argparse
import difflib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_stage', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = ['ggml/src/ggml-metal/ggml-metal-device.cpp', 'ggml/src/ggml-metal/kernels/mul_mv.metal']
    old = {p:(args.source_stage/p).read_text() for p in paths}
    new = dict(old)
    p = paths[0]
    needle = '        const char * kernel = "kernel_mul_mv_q8_0_f32_cpuarith";'
    replacement = '''        const char * shared_flag = std::getenv("AIRLLM_METAL_Q8_SHARED");
        const bool shared_head = std::strcmp(cpu_arith, "1") == 0 && shared_flag && std::strcmp(shared_flag, "1") == 0 &&
            ne00 == 4096 && ne01 == 154880 && std::strcmp(op->src[0]->name, "output.weight") == 0;
        const char * kernel = shared_head ? "kernel_mul_mv_q8_0_f32_cpuarith_shared" : "kernel_mul_mv_q8_0_f32_cpuarith";'''
    assert new[p].count(needle)==1
    new[p]=new[p].replace(needle,replacement,1)
    p=paths[1]
    body=(root/'kernels_q8/q8_matvec_shared.metal').read_text()
    body=body.replace('const uint lane = thread_position_in_grid.x % 32;',
                      'device const uchar * packed = (device const uchar *) src0;\n    device const float * x = (device const float *) src1;\n    device float * out = (device float *) dst;\n    const uint lane = tiisg;')
    body=body.replace('const uint row = thread_position_in_grid.x / 4;', 'const uint row = tgpig.x * 8 + tiisg / 4;')
    body=body.replace('const uint part = thread_position_in_grid.x % 4;', 'const uint part = tiisg % 4;')
    body=body.replace('if (row >= packed_shape[0]) return;', 'if (row >= args.ne01) return;')
    body=body.replace('(row * 128 + block) * 34', 'uint64_t(row) * args.nb01 + block * 34')
    kernel='''// Private head alternative: one Q8 activation tile reused by eight rows.
[[host_name("kernel_mul_mv_q8_0_f32_cpuarith_shared")]]
kernel void kernel_mul_mv_q8_0_f32_cpuarith_shared(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
'''+''.join('    '+line+'\n' if line.strip() else '\n' for line in body.splitlines())+'}\n\n'
    marker='// mat-vec kernel processing in chunks of float4\n'
    assert new[p].count(marker)==1
    new[p]=new[p].replace(marker,kernel+marker,1)
    # One unchanged context line keeps the getter hunk applicable with/without selective mode 2.
    patch=''.join(''.join(difflib.unified_diff(old[p].splitlines(True),new[p].splitlines(True),fromfile='a/'+p,tofile='b/'+p,n=1)) for p in paths)
    output=root/'patches/llama-q8-shared-activation.patch';output.write_text(patch)
    subprocess.run(['git','apply','--check',str(output)],cwd=args.source_stage,check=True)
    print(f'Shared head patch applies, {len(patch.splitlines())} lines')


if __name__=='__main__':
    main()
