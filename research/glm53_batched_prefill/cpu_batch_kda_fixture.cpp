// MIT. Component fixture only: native fused KDA with identical fixed inputs.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static constexpr int D = 128, H = 2, T = 32;
static constexpr size_t STEP = D * H, STATE = D * D * H;

struct inputs {
    std::vector<float> q, k, v, g, beta, state;
    inputs() : q(STEP*T), k(STEP*T), v(STEP*T), g(STEP*T), beta(H*T), state(STATE) {
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = float(int(i % 53) - 26) / 128.0f;
            k[i] = float(int(i % 47) - 23) / 128.0f;
            v[i] = float(int(i % 31) - 15) / 64.0f;
            g[i] = -float(1 + i % 17) / 64.0f;
        }
        for (size_t i = 0; i < beta.size(); ++i) beta[i] = float(1 + i % 15) / 16.0f;
        for (size_t i = 0; i < state.size(); ++i) state[i] = float(int(i % 7) - 3) / 256.0f;
    }
};

struct outputs { std::vector<float> attention, state; };

static outputs run(const inputs & input, int batch) {
    outputs result{std::vector<float>(STEP*T), input.state};
    for (int offset = 0; offset < T; offset += batch) {
        ggml_init_params init{4 << 20, nullptr, false};
        ggml_context * ctx = ggml_init(init);
        if (!ctx) throw std::runtime_error("fixture context allocation failed");
        auto activation = [&](const std::vector<float> & values) {
            auto * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, batch, 1);
            memcpy(tensor->data, values.data() + size_t(offset)*STEP, size_t(batch)*STEP*sizeof(float));
            return tensor;
        };
        auto * q = activation(input.q);
        auto * k = activation(input.k);
        auto * v = activation(input.v);
        auto * g = activation(input.g);
        auto * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, batch, 1);
        auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, 1);
        memcpy(beta->data, input.beta.data() + size_t(offset)*H, size_t(batch)*H*sizeof(float));
        memcpy(state->data, result.state.data(), STATE*sizeof(float));
        auto * output = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto plan = ggml_graph_plan(graph, 10, nullptr);
        if (plan.work_size > (1 << 20)) { ggml_free(ctx); throw std::runtime_error("fixture workspace exceeds1MiB"); }
        std::vector<unsigned char> scratch(plan.work_size);
        plan.work_data = scratch.data();
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx); throw std::runtime_error("native KDA operation failed");
        }
        const auto * values = static_cast<const float *>(output->data);
        const size_t attention_elements = size_t(batch)*STEP;
        memcpy(result.attention.data() + size_t(offset)*STEP, values, attention_elements*sizeof(float));
        memcpy(result.state.data(), values + attention_elements, STATE*sizeof(float));
        ggml_free(ctx);
    }
    return result;
}

static bool finite(const outputs & value) {
    for (auto v : value.attention) if (!std::isfinite(v)) return false;
    for (auto v : value.state) if (!std::isfinite(v)) return false;
    return true;
}

static bool exact(const outputs & a, const outputs & b) {
    return finite(a) && finite(b) &&
        memcmp(a.attention.data(), b.attention.data(), a.attention.size()*sizeof(float)) == 0 &&
        memcmp(a.state.data(), b.state.data(), a.state.size()*sizeof(float)) == 0;
}

int main() {
    try {
        inputs input;
        const auto a = run(input, 8), b = run(input, 16), c = run(input, 32);
        const bool passed = exact(a, b) && exact(a, c);
        std::cout << "{\"component_only\":true,\"model_loaded\":false,\"tokens\":32,\"head_dim\":128,"
            "\"heads\":2,\"threads\":10,\"attention_elements\":" << STEP*T << ",\"state_elements\":" << STATE <<
            ",\"batch8_vs16_exact\":" << (exact(a,b)?"true":"false") <<
            ",\"batch8_vs32_exact\":" << (exact(a,c)?"true":"false") <<
            ",\"passed\":" << (passed?"true":"false") << "}\n";
        return passed ? 0 : 3;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
