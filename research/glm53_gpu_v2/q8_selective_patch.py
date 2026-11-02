#!/usr/bin/env python3
"""Generate an incremental patch against an isolated Q8+PR18 source stage."""
from pathlib import Path
import argparse
import difflib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_stage', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = ['ggml/src/ggml-metal/ggml-metal-device.cpp', 'ggml/src/ggml-metal/ggml-metal-device.m', 'src/llama-context.cpp']
    old = {p: (args.source_stage/p).read_text() for p in paths}
    new = dict(old)
    p = paths[0]
    before = '''    const char * cpu_arith = std::getenv("AIRLLM_METAL_Q8_CPU_ARITH");
    if (cpu_arith && std::strcmp(cpu_arith, "1") == 0 && op->op == GGML_OP_MUL_MAT &&
            tsrc0 == GGML_TYPE_Q8_0 && tsrc1 == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            std::strcmp(op->src[0]->name, "output.weight") == 0 &&
            ne00 == 4096 && ne01 == 154880 && ne02 == 1 && ne03 == 1 &&
            ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {'''
    after = '''    const char * cpu_arith = std::getenv("AIRLLM_METAL_Q8_CPU_ARITH");
    const bool mode_head = cpu_arith && std::strcmp(cpu_arith, "1") == 0;
    const bool mode_selective = cpu_arith && std::strcmp(cpu_arith, "2") == 0;
    const bool head = ne00 == 4096 && ne01 == 154880 && std::strcmp(op->src[0]->name, "output.weight") == 0;
    if ((mode_selective || (mode_head && head)) && op->op == GGML_OP_MUL_MAT &&
            tsrc0 == GGML_TYPE_Q8_0 && tsrc1 == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            ne00 >= 64 && ne00 <= 65536 && ne00 % 64 == 0 && ne01 > 0 && ne01 % 8 == 0 &&
            ne02 == 1 && ne03 == 1 && ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {'''
    assert new[p].count(before) == 1
    new[p] = new[p].replace(before, after, 1)
    new[p] = new[p].replace('experimental CPU-compatible Q8_0 output matvec', 'experimental CPU-compatible Q8_0 matvec')
    p = paths[1]
    before = 'if (kind == GGML_METAL_LIB_MUL_MV && cpu_arith && strcmp(cpu_arith, "1") == 0) {'
    after = 'if (kind == GGML_METAL_LIB_MUL_MV && cpu_arith && (strcmp(cpu_arith, "1") == 0 || strcmp(cpu_arith, "2") == 0)) {'
    assert new[p].count(before) == 1
    new[p] = new[p].replace(before, after, 1)
    p = paths[2]
    helper = '''// Private inference policy: pin every graph operation before scheduling.
static void airllm_glm_q8_only_assign(
        const llama_model & model, ggml_backend_sched_t sched, ggml_backend_t cpu,
        const std::vector<ggml_backend_t> & backends, ggml_cgraph * gf, bool training) {
    const char * enabled = std::getenv("AIRLLM_GLM_Q8_ONLY");
    if (!enabled || std::strcmp(enabled, "1") != 0 || gf == nullptr) {
        return;
    }
    const char * mode = std::getenv("AIRLLM_METAL_Q8_CPU_ARITH");
    if (model.arch != LLM_ARCH_GLM5_NEXT || model.n_gpu_layers() != 0 || !cpu || training ||
            !mode || std::strcmp(mode, "2") != 0) {
        GGML_ABORT("AIRLLM_GLM_Q8_ONLY requires GLM inference, GPU layers 0, CPU backend and Q8 mode 2");
    }
    size_t n_cpu = 0;
    size_t n_gpu = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE) {
            continue;
        }
        ggml_backend_t target = cpu;
        const ggml_tensor * w = node->src[0];
        const ggml_tensor * x = node->src[1];
        const bool eligible = node->op == GGML_OP_MUL_MAT && w && x &&
            w->type == GGML_TYPE_Q8_0 && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
            w->ne[0] >= 64 && w->ne[0] <= 65536 && w->ne[0] % 64 == 0 && w->ne[1] > 0 && w->ne[1] % 8 == 0 &&
            w->ne[2] == 1 && w->ne[3] == 1 && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
            ggml_is_contiguous(w) && ggml_is_contiguous(x);
        if (eligible && w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer));
            if (device && std::strncmp(ggml_backend_dev_name(device), "MTL", 3) == 0) {
                target = nullptr;
                for (auto * backend : backends) {
                    if (ggml_backend_get_device(backend) == device) {
                        target = backend;
                        break;
                    }
                }
            }
        }
        if (!target || !ggml_backend_supports_op(target, node)) {
            GGML_ABORT("AIRLLM_GLM_Q8_ONLY cannot place %s (%s) on its required backend", node->name, ggml_op_name(node->op));
        }
        ggml_backend_sched_set_tensor_backend(sched, node, target);
        if (target == cpu) {
            ++n_cpu;
        } else {
            ++n_gpu;
        }
    }
    LLAMA_LOG_INFO("%s: experimental selective Q8 policy: %zu CPU ops, %zu Metal Q8 matvecs\\n", __func__, n_cpu, n_gpu);
}

'''
    marker = '//\n// llama_context\n//\n'
    assert new[p].count(marker) == 1
    new[p] = new[p].replace(marker, helper+marker, 1)
    # Actual freshly built inference graph and the reserve/fused-probe graph.
    before = '        gf = model.build_graph(gparams);\n'
    after = before+'        airllm_glm_q8_only_assign(model, sched.get(), backend_cpu, backend_ptrs, gf, cparams.training);\n'
    assert new[p].count(before) == 1
    new[p] = new[p].replace(before, after, 1)
    before = '    auto * gf = model.build_graph(gparams);\n'
    after = before+'    airllm_glm_q8_only_assign(model, sched.get(), backend_cpu, backend_ptrs, gf, cparams.training);\n'
    # There are reserve and optimizer sites. Rejecting training also avoids an uncontrolled optimizer graph.
    assert new[p].count(before) == 2
    new[p] = new[p].replace(before, after)
    new[p] = new[p].replace('            auto * gf = model.build_graph(gparams);\n    airllm_glm_q8_only_assign', '            auto * gf = model.build_graph(gparams);\n            airllm_glm_q8_only_assign')
    patch = ''.join(''.join(difflib.unified_diff(old[p].splitlines(True), new[p].splitlines(True), fromfile='a/'+p, tofile='b/'+p)) for p in paths)
    output = root/'patches/llama-q8-selective.patch'
    output.write_text(patch)
    subprocess.run(['git','apply','--check',str(output)],cwd=args.source_stage,check=True)
    print(f'Incremental selective patch applies: {len(patch.splitlines())} lines')


if __name__ == '__main__':
    main()
