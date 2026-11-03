// MIT. Component oracle using the pinned CPU graph dispatcher.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstring>
#include <vector>

extern "C" int airllm_q8_cpu_graph(const void * weights, const float * input, float * output,
                                  int width, int rows, int tokens, int threads) {
    if (!weights || !input || !output || width < 32 || width > 65536 || width % 32 ||
            rows < 2 || rows > 8192 || rows % 2 || tokens < 2 || tokens > 128 || tokens % 2 ||
            threads < 1 || threads > 10) return -1;
    const size_t weight_bytes = size_t(width / 32) * 34 * rows;
    const size_t payload = weight_bytes + size_t(tokens) * (width + rows) * sizeof(float);
    if (payload > (32 << 20)) return -2;
    ggml_init_params params = { payload + ggml_graph_overhead() + 16 * ggml_tensor_overhead() + (64 << 10), nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return -3;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    memcpy(w->data, weights, weight_bytes);
    memcpy(x->data, input, size_t(tokens) * width * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_cplan plan = ggml_graph_plan(graph, threads, nullptr);
    if (plan.work_size > (32 << 20)) { ggml_free(ctx); return -5; }
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status == GGML_STATUS_SUCCESS) memcpy(output, y->data, size_t(tokens) * rows * sizeof(float));
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS ? 0 : -4;
}
