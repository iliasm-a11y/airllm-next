// MIT. Component adapter to existing GGML backends, not a model engine.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

extern "C" void dequantize_row_q8_K(const void *, float *, int64_t);

using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

struct component {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;
    ggml_type activation_type = GGML_TYPE_F32;
    int width = 0;
    bool metal = false;
    std::vector<unsigned char> quantized;
    std::vector<float> decoded;
    ~component() {
        if (backend) ggml_backend_synchronize(backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

extern "C" void * native_quant_create(int format, const void * packed,
        int width, int rows, int gpu, int threads, size_t * allocated) {
    if (!packed || !allocated || (format != 2 && format != 8) ||
        width < 256 || width > 16384 || width % 256 || rows < 8 || rows > 8192 ||
        rows % 8 || (gpu != 0 && gpu != 1) || threads < 1 || threads > 10) return nullptr;
    const ggml_type type = format == 2 ? GGML_TYPE_Q2_K : GGML_TYPE_Q8_0;
    if (ggml_row_size(type, width)*rows > (16 << 20)) return nullptr;
    try {
        ggml_cpu_init();
        auto c = std::make_unique<component>();
        c->width = width; c->metal = gpu;
        c->activation_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
        c->quantized.resize(ggml_row_size(c->activation_type, width));
        c->decoded.resize(width);
        c->backend = gpu ? ggml_backend_metal_init() : ggml_backend_cpu_init();
        if (!c->backend) return nullptr;
        if (!gpu) ggml_backend_cpu_set_n_threads(c->backend, threads);
        ggml_init_params params = {ggml_graph_overhead()+8*ggml_tensor_overhead()+(64 << 10), nullptr, true};
        c->ctx = ggml_init(params);
        if (!c->ctx) return nullptr;
        auto * w = ggml_new_tensor_2d(c->ctx, type, width, rows);
        ggml_set_name(w, "component.weight");
        c->x = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, width, 1);
        c->y = ggml_mul_mat(c->ctx, w, c->x);
        if (!ggml_backend_dev_supports_op(ggml_backend_get_device(c->backend), c->y)) return nullptr;
        c->graph = ggml_new_graph(c->ctx);
        ggml_build_forward_expand(c->graph, c->y);
        auto buft = ggml_backend_get_default_buffer_type(c->backend);
        *allocated = ggml_backend_alloc_ctx_tensors_from_buft_size(c->ctx, buft);
        if (!*allocated || *allocated > (32 << 20)) return nullptr;
        c->buffer = ggml_backend_alloc_ctx_tensors(c->ctx, c->backend);
        if (!c->buffer) return nullptr;
        ggml_backend_tensor_set(w, packed, 0, ggml_nbytes(w));
        ggml_backend_synchronize(c->backend);
        return c.release();
    } catch (...) { return nullptr; }
}

// Times include input preparation/import, submit, terminal wait and output read.
// Only the activation is dequantized by mode 1; packed weights never change.
extern "C" int native_quant_run(void * handle, const float * x, float * y,
        int adapted, double * timing) {
    if (!handle || !x || !y || !timing || (adapted != 0 && adapted != 1)) return -1;
    auto * c = static_cast<component *>(handle);
    if (adapted && !c->metal) return -2;
    for (int i=0; i<c->width; ++i) if (!std::isfinite(x[i])) return -3;
    const auto t0 = clock_type::now();
    const float * input = x;
    if (adapted) {
        auto quantize = ggml_get_type_traits_cpu(c->activation_type)->from_float;
        if (!quantize) return -4;
        quantize(x, c->quantized.data(), c->width);
        if (c->activation_type == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(c->quantized.data(), c->decoded.data(), c->width);
        } else {
            auto decode = ggml_get_type_traits(c->activation_type)->to_float;
            if (!decode) return -5;
            decode(c->quantized.data(), c->decoded.data(), c->width);
        }
        input = c->decoded.data();
    }
    ggml_backend_tensor_set(c->x, input, 0, ggml_nbytes(c->x));
    const auto t1 = clock_type::now();
    const auto status = ggml_backend_graph_compute_async(c->backend, c->graph);
    const auto t2 = clock_type::now();
    ggml_backend_synchronize(c->backend);
    const auto t3 = clock_type::now();
    if (status != GGML_STATUS_SUCCESS) return -6;
    ggml_backend_tensor_get(c->y, y, 0, ggml_nbytes(c->y));
    const auto t4 = clock_type::now();
    timing[0]=seconds(t0,t4); timing[1]=seconds(t0,t1);
    timing[2]=seconds(t1,t2); timing[3]=seconds(t2,t3); timing[4]=seconds(t3,t4);
    return 0;
}

extern "C" void native_quant_destroy(void * handle) { delete static_cast<component *>(handle); }
