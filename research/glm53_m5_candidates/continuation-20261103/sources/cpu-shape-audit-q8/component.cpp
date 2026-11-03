// Native CPU10 component diagnostic. Synthetic resident data, no model/GPU.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>
using json=nlohmann::json;
static uint32_t state;
static float random_float() {
    state=state*1664525u+1013904223u;
    return float(int32_t(state>>8)-8388608)*0x1.000002p-23f;
}
static json compare(const float *a,const float *b,size_t count) {
    size_t different=0;double error=0;bool finite=true;
    for(size_t i=0;i<count;++i) {
        different+=std::memcmp(a+i,b+i,4)!=0;
        finite=finite&&std::isfinite(a[i])&&std::isfinite(b[i]);
        error=std::max(error,std::abs(double(a[i])-double(b[i])));
    }
    if(!finite)throw std::runtime_error("nonfinite component output");
    return {{"outputs",count},{"different_float32_bits",different},{"max_abs_error",error},{"all_finite",finite}};
}
int main() {try {
    auto backend=ggml_backend_cpu_init();if(!backend)throw std::runtime_error("CPU init");
    ggml_backend_cpu_set_n_threads(backend,10);ggml_backend_cpu_set_use_ref(backend,false);
    json rows=json::array();
    struct Shape {int k,m;ggml_type type;bool norm;const char *role;};
    const Shape shapes[]={{4096,8192,GGML_TYPE_Q8_0,false,"attn_q_k_v"}};
    for(const auto &s:shapes)for(int n:{16,32,64,128})for(unsigned seed:{5305u,713962u}) {
        // Allocation includes full and split outputs; one case is freed before next.
        auto ctx=ggml_init({2u<<20,nullptr,true});if(!ctx)throw std::runtime_error("metadata");
        auto w=ggml_new_tensor_2d(ctx,s.type,s.k,s.m);
        auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s.k,n);
        auto normalized=s.norm?ggml_rms_norm(ctx,x,1e-6f):x;
        auto full=ggml_mul_mat(ctx,w,normalized);
        auto fg=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(fg,full);
        std::vector<ggml_tensor *> parts,norms;std::vector<ggml_cgraph *> graphs;
        for(int start=0;start<n;start+=8) {
            auto view=ggml_view_2d(ctx,x,s.k,8,x->nb[1],size_t(start)*x->nb[1]);
            auto norm=s.norm?ggml_rms_norm(ctx,view,1e-6f):view;
            auto y=ggml_mul_mat(ctx,w,norm);auto g=ggml_new_graph_custom(ctx,16,false);
            ggml_build_forward_expand(g,y);parts.push_back(y);norms.push_back(norm);graphs.push_back(g);
        }
        auto memory=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!memory)throw std::runtime_error("tensor allocation");
        size_t payload=ggml_backend_buffer_get_size(memory)+(2u<<20);
        if(payload>64u<<20)throw std::runtime_error("logical payload cap");
        state=seed;std::vector<float> weights(s.k);
        auto quantize=ggml_get_type_traits_cpu(s.type)->from_float;
        if(!quantize)throw std::runtime_error("native quantizer");
        for(int row=0;row<s.m;++row) {
            for(float &v:weights)v=random_float()*.05f;
            quantize(weights.data(),(char *)w->data+size_t(row)*w->nb[1],s.k);
        }
        for(size_t i=0;i<size_t(s.k)*n;++i)((float *)x->data)[i]=random_float();
        if(ggml_backend_graph_compute(backend,fg)!=GGML_STATUS_SUCCESS)throw std::runtime_error("full compute");
        std::vector<float> split(size_t(s.m)*n),split_norm(s.norm?size_t(s.k)*n:0);
        for(size_t i=0;i<graphs.size();++i) {
            if(ggml_backend_graph_compute(backend,graphs[i])!=GGML_STATUS_SUCCESS)throw std::runtime_error("split compute");
            std::memcpy(split.data()+i*s.m*8,parts[i]->data,size_t(s.m)*8*4);
            if(s.norm)std::memcpy(split_norm.data()+i*s.k*8,norms[i]->data,size_t(s.k)*8*4);
        }
        json row={{"role",s.role},{"K",s.k},{"M",s.m},{"N",n},{"seed",seed},
            {"weight_type",ggml_type_name(s.type)},{"tensor_payload_and_metadata_bytes",payload},
            {"full_vs_split8",compare((float *)full->data,split.data(),split.size())}};
        if(s.norm)row["rms_full_vs_split8"]=compare((float *)normalized->data,split_norm.data(),split_norm.size());
        rows.push_back(row);ggml_backend_buffer_free(memory);ggml_free(ctx);
    }
    rusage usage={};if(getrusage(RUSAGE_SELF,&usage)!=0||usage.ru_maxrss>(128u<<20))throw std::runtime_error("native lifetime RSS cap");
    ggml_backend_free(backend);
    std::cout<<json({{"status","completed_component_diagnostic_only"},{"threads",10},{"uses_reference_mode",false},
        {"model_loaded",false},{"gpu_backend_initialized",false},{"gpu_matrix_dispatches",0},
        {"native_lifetime_peak_rss_bytes",usage.ru_maxrss},{"rows",rows},{"whole_model_causality_proven",false}}).dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}}
