// CPU-only file-mapped Q2_K matmul_id fixture. No model or GPU is loaded.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    constexpr int width = 256, rows = 1024, experts = 4;
    const size_t stride = rows * 84, bytes = experts * stride;
    const size_t page = sysconf(_SC_PAGESIZE);
    const size_t mapped = (bytes + page - 1) / page * page;
    std::vector<unsigned char> original(mapped, 0);
    for (size_t block = 0; block < bytes / 84; ++block) {
        unsigned char * p = original.data() + block * 84;
        for (int i = 0; i < 16; ++i) { p[i] = 1; }
        for (int i = 16; i < 80; ++i) { p[i] = static_cast<unsigned char>(i + block); }
        p[80] = 0; p[81] = 0x3c;
    }
    char path[] = "/tmp/airllm-expert-drop-fixture-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, mapped) || pwrite(fd, original.data(), mapped, 0) != ssize_t(mapped)) { return 2; }
    void * data = mmap(nullptr, mapped, PROT_READ, MAP_SHARED, fd, 0);
    unlink(path); close(fd);
    if (data == MAP_FAILED) { return 3; }
    auto cpu = ggml_backend_cpu_init();
    if (!cpu) { return 4; }
    ggml_backend_cpu_set_n_threads(cpu, 4);
    auto ctx = ggml_init({4 << 20, nullptr, true});
    auto weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, width, rows, experts);
    ggml_set_name(weight, "fixture.ffn_gate_exps.weight");
    auto weights = ggml_backend_cpu_buffer_from_ptr(data, mapped);
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight->data = data; weight->buffer = weights;
    auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    auto output = ggml_mul_mat_id(ctx, weight, x, ids);
    auto pre_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 2);
    auto pre_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    auto pre_output = ggml_mul_mat_id(ctx, weight, pre_x, pre_ids);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    auto pre_graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(pre_graph, pre_output);
    auto tensors = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> activation(width * 2);
    for (int i = 0; i < width * 2; ++i) { activation[i] = float(i % 31 - 15) * .03125f; }
    ggml_backend_tensor_set(x, activation.data(), 0, width * sizeof(float));
    ggml_backend_tensor_set(pre_x, activation.data(), 0, activation.size() * sizeof(float));
    int checks = 0;
    for (const auto selected : {std::vector<int>{1, 3}, std::vector<int>{3, 3}}) {
        ggml_backend_tensor_set(ids, selected.data(), 0, 2 * sizeof(int));
        std::vector<float> before(rows * 2), after(rows * 2);
        unsetenv("AIRLLM_EXPERT_DROP_AFTER_USE");
        if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) { return 5; }
        ggml_backend_tensor_get(output, before.data(), 0, before.size() * sizeof(float));
        setenv("AIRLLM_EXPERT_DROP_AFTER_USE", "1", 1);
        if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) { return 6; }
        ggml_backend_tensor_get(output, after.data(), 0, after.size() * sizeof(float));
        if (std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) ||
                std::memcmp(data, original.data(), bytes)) { return 7; }
        ++checks;
    }
    const int pre_selected[] = {1, 3, 3, 1};
    ggml_backend_tensor_set(pre_ids, pre_selected, 0, sizeof(pre_selected));
    std::fprintf(stderr, "AIRLLM_FIXTURE_PREFILL_BEGIN\n");
    if (ggml_backend_graph_compute(cpu, pre_graph) != GGML_STATUS_SUCCESS) { return 8; }
    std::fprintf(stderr, "AIRLLM_FIXTURE_PREFILL_END\n");
    std::printf("{\"status\":\"passed\",\"checks\":%d,\"cpu_threads\":4,\"mapped_original_bytes_unchanged\":true,\"decode_outputs_bit_identical\":true}\n", checks);
    ggml_backend_buffer_free(tensors);
    ggml_backend_buffer_free(weights);
    ggml_free(ctx); ggml_backend_free(cpu); munmap(data, mapped);
    return 0;
}
