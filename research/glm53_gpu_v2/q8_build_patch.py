#!/usr/bin/env python3
"""Generate the private pinned-runtime patch without editing that runtime."""
from pathlib import Path
import argparse
import difflib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = ['ggml/src/ggml-metal/ggml-metal-device.cpp', 'ggml/src/ggml-metal/ggml-metal-device.m',
             'ggml/src/ggml-metal/kernels/mul_mv.metal']
    old = {p: subprocess.check_output(['git', 'show', 'HEAD:'+p], cwd=args.runtime, text=True) for p in paths}
    new = dict(old)
    p = paths[0]
    new[p] = new[p].replace('#include <cassert>\n', '#include <cassert>\n#include <cstdlib>\n#include <cstring>\n', 1)
    start = new[p].index('ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(')
    end = new[p].index('ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(', start)
    section = new[p][start:end]
    needle = '    const char * suffix = "";'
    replacement = '''    const char * cpu_arith = std::getenv("AIRLLM_METAL_Q8_CPU_ARITH");
    if (cpu_arith && std::strcmp(cpu_arith, "1") == 0 && op->op == GGML_OP_MUL_MAT &&
            tsrc0 == GGML_TYPE_Q8_0 && tsrc1 == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            std::strcmp(op->src[0]->name, "output.weight") == 0 &&
            ne00 == 4096 && ne01 == 154880 && ne02 == 1 && ne03 == 1 &&
            ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
        const char * kernel = "kernel_mul_mv_q8_0_f32_cpuarith";
        ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, kernel);
        if (!res.pipeline) {
            GGML_LOG_INFO("%s: experimental CPU-compatible Q8_0 output matvec\\n", __func__);
            res = ggml_metal_library_compile_pipeline(lib, kernel, kernel, nullptr);
        }
        res.nr0 = 8;
        res.nr1 = 1;
        res.nsg = 1;
        res.smem = 0;
        return res;
    }

    const char * suffix = "";'''
    assert section.count(needle) == 1
    new[p] = new[p][:start] + section.replace(needle, replacement, 1) + new[p][end:]
    p = paths[1]
    needle = '''                ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(res->dev)->has_tensor);

                lib = [device newLibraryWithSource:src options:options error:&error];'''
    replacement = '''                ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(res->dev)->has_tensor);
                const char * cpu_arith = getenv("AIRLLM_METAL_Q8_CPU_ARITH");
                if (kind == GGML_METAL_LIB_MUL_MV && cpu_arith && strcmp(cpu_arith, "1") == 0) {
                    options.fastMathEnabled = NO;
                    GGML_LOG_INFO("%s: experimental Q8_0 MUL_MV safe math enabled\\n", __func__);
                }

                lib = [device newLibraryWithSource:src options:options error:&error];'''
    assert new[p].count(needle) == 1
    new[p] = new[p].replace(needle, replacement, 1)
    p = paths[2]
    helper = (root/'kernels_q8/q8_inline_helpers.metal').read_text().replace('uint part) {\n', 'uint part) {\n#pragma clang fp contract(off)\n', 1)
    body = (root/'kernels_q8/q8_matvec_inline.metal').read_text()
    body = body.replace('const uint row = thread_position_in_grid.x / 4;', 'const uint row = tgpig.x * 8 + tiisg / 4;')
    body = body.replace('const uint part = thread_position_in_grid.x % 4;', 'const uint part = tiisg % 4;')
    body = body.replace('const uint blocks = WIDTH / 32;', 'const uint blocks = args.ne00 / 32;\n    device const uchar * packed = (device const uchar *) src0;\n    device const float * x = (device const float *) src1;\n    device float * out = (device float *) dst;')
    body = body.replace('if (row >= packed_shape[0]) return;', 'if (row >= args.ne01) return;')
    body = body.replace('(row * blocks + block) * 34', 'uint64_t(row) * args.nb01 + block * 34')
    kernel = '\n// Private output-head experiment: original Q8_0 bytes and ARM nrc=1 arithmetic.\n'+helper+'''
[[host_name("kernel_mul_mv_q8_0_f32_cpuarith")]]
kernel void kernel_mul_mv_q8_0_f32_cpuarith(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
'''+body+'}\n\n'
    needle = '// mat-vec kernel processing in chunks of float4\n'
    assert new[p].count(needle) == 1
    new[p] = new[p].replace(needle, kernel+needle, 1)
    patch = ''.join(''.join(difflib.unified_diff(old[p].splitlines(True), new[p].splitlines(True), fromfile='a/'+p, tofile='b/'+p)) for p in paths)
    output = root/'patches/llama-q8-cpuarith.patch'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patch)
    subprocess.run(['git', 'apply', '--check', str(output)], cwd=args.runtime, check=True)
    print(f'Patch applies, {len(patch.splitlines())} lines')


if __name__ == '__main__':
    main()
