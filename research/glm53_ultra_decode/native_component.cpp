// MIT. Persistent single-token CPU graph, not a production engine replacement.
// Borrow immutable packed weights and FP32 activations. No weight copy/cache.
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

struct component {
    ggml_context* ctx=nullptr;
    ggml_threadpool* pool=nullptr;
    ggml_cgraph* graph=nullptr;
    ggml_cplan plan{};
    std::vector<unsigned char> work;
    std::vector<float> result;
    ~component() {
        if(pool) ggml_threadpool_free(pool);
        if(ctx) ggml_free(ctx);
    }
};

extern "C" void* decode_component_create(int type, const void* packed, const void* packed1,
                                         const float* input, int width, int rows, int banks, int threads) {
    if((type!=2&&type!=8)||!packed||!input||width<256||width>16384||width%256||
       rows<1||rows>8192||(banks!=1&&banks!=2)||(banks==2&&!packed1)||threads<1||threads>10) return nullptr;
    const size_t weight_bytes=size_t(rows)*(width/(type==2?256:32))*(type==2?84:34);
    if(weight_bytes*banks>(32<<20)) return nullptr;
    try {
        auto c=std::make_unique<component>();
        ggml_cpu_init();
        ggml_init_params params={ggml_graph_overhead()+12*ggml_tensor_overhead()+(64<<10),nullptr,true};
        c->ctx=ggml_init(params); if(!c->ctx) return nullptr;
        auto* x=ggml_new_tensor_2d(c->ctx,GGML_TYPE_F32,width,1);
        x->data=const_cast<float*>(input);c->result.resize(rows*banks);
        c->graph=ggml_new_graph(c->ctx);
        for(int b=0;b<banks;++b) {
            auto* w=ggml_new_tensor_2d(c->ctx,type==2?GGML_TYPE_Q2_K:GGML_TYPE_Q8_0,width,rows);
            w->data=const_cast<void*>(b?packed1:packed);
            auto* y=ggml_mul_mat(c->ctx,w,x);y->data=c->result.data()+b*rows;
            ggml_build_forward_expand(c->graph,y);
        }
        // A pair uses one pool, as in the model. Two independently spinning
        // CPU10 pools would turn scheduler contention into a false GPU gain.
        auto pool_params=ggml_threadpool_params_default(threads);
        c->pool=ggml_threadpool_new(&pool_params);if(!c->pool) return nullptr;
        c->plan=ggml_graph_plan(c->graph,threads,c->pool);
        if(c->plan.work_size>(32<<20)) return nullptr;
        c->work.resize(c->plan.work_size);c->plan.work_data=c->work.data();
        return c.release();
    } catch(...) { return nullptr; }
}

extern "C" int decode_component_run(void* handle,float* output,double* seconds) {
    if(!handle||!output||!seconds) return -1;
    auto* c=static_cast<component*>(handle);
    const auto begin=std::chrono::steady_clock::now();
    const auto status=ggml_graph_compute(c->graph,&c->plan);
    *seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    if(status!=GGML_STATUS_SUCCESS) return -2;
    std::memcpy(output,c->result.data(),c->result.size()*sizeof(float));
    return 0;
}
extern "C" void decode_component_destroy(void* handle) { delete static_cast<component*>(handle); }
