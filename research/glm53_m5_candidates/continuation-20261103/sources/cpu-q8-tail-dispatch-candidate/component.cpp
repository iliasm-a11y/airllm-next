// Direct native CPU10 Q8 dispatch diagnostic. No scheduler/BLAS/GPU/model.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <poll.h>
#include <unistd.h>
#include <vector>
using json=nlohmann::json;
static constexpr size_t RSS_CAP=128u<<20;
static const std::vector<int> sizes={14,18,20,22,24,26,28,30,32,38,44,48,64,72,80,88,96,104,112,120,128};
static uint32_t random_state;
static float next_float(){random_state=random_state*1664525u+1013904223u;return float(int32_t(random_state>>8)-8388608)*0x1.000002p-23f;}
static size_t peak_rss(){rusage u{};if(getrusage(RUSAGE_SELF,&u)||u.ru_maxrss<0||size_t(u.ru_maxrss)>RSS_CAP)throw std::runtime_error("native lifetime128MiB cap");return size_t(u.ru_maxrss);}
static std::string digest(const void *data,size_t bytes){unsigned char h[32];CC_SHA256(data,CC_LONG(bytes),h);const char hex[]="0123456789abcdef";std::string s;for(auto c:h){s+=hex[c>>4];s+=hex[c&15];}return s;}
static json compare(const float *reference,const float *candidate,size_t count){size_t diff=0,first=count;double error=0;
    for(size_t i=0;i<count;++i){if(!std::isfinite(reference[i])||!std::isfinite(candidate[i]))throw std::runtime_error("nonfinite output");
        if(std::memcmp(reference+i,candidate+i,4)){++diff;if(first==count)first=i;}error=std::max(error,std::abs(double(reference[i])-double(candidate[i])));}
    return {{"outputs",count},{"different_float32_bits",diff},{"cpu_bitexact",diff==0},{"all_finite",true},{"max_abs_error",error},
        {"first_different_output",first==count?json(nullptr):json(first)},{"reference_sha256",digest(reference,count*4)},{"candidate_sha256",digest(candidate,count*4)}};}
static std::vector<int> partition8(int n){std::vector<int> out;while(n){const int take=std::min(8,n);out.push_back(take);n-=take;}return out;}
static std::vector<int> partition_v4(int n){std::vector<int> out;while(n){int take=n;
    if(take>16 && take%16)take-=take%16;
    else if(take>8 && take%2)take=8;
    if(take<=0||take>n)throw std::runtime_error("invalid v4 partition");out.push_back(take);n-=take;}return out;}
// Derived metadata only, using the exact pinned ggml-cpu.c loop. It does not
// instrument actual per-kernel calls, and never changes the real native traits.
static json derived_dispatch(int m,int n,int64_t native_nrows){const int chunk_size=(m==1||n==1)?64:16;
    int nc0=(m+chunk_size-1)/chunk_size,nc1=(n+chunk_size-1)/chunk_size;
    const bool rechunk=nc0*nc1<10*4||ggml_is_numa();if(rechunk){nc0=m>n?10:1;nc1=m>n?1:10;}
    const int dr0=(m+nc0-1)/nc0,dr1=(n+nc1-1)/nc1;json columns=json::array();
    for(int i=0;i<nc1;++i){const int first=dr1*i,end=std::min(first+dr1,n);if(first>=end)continue;
        const int64_t nrc=(m%2||n%2||dr0%2||(end-first)%2)?1:native_nrows;
        columns.push_back({{"first",first},{"end",end},{"derived_nrc",nrc}});}
    return {{"source","pinned ggml-cpu.c:1476-1524"},{"scope","derived from actual traits/shape; no per-kernel instrumentation"},
        {"native_vec_dot_nrows",native_nrows},{"numa",ggml_is_numa()},{"rechunked_by_threads",rechunk},
        {"nchunk0",nc0},{"nchunk1",nc1},{"dr0",dr0},{"dr1",dr1},{"column_chunks",columns},{"actual_called_nrc",nullptr}};}
int main(int argc,char **argv){try{
    if(argc!=2)throw std::runtime_error("usage: COMPONENT NEW_OUTPUT_DIRECTORY");
    const char *gate_fd=std::getenv("AIRLLM_FIXTURE_START_FD");
    if(!gate_fd||!*gate_fd||std::strspn(gate_fd,"0123456789")!=std::strlen(gate_fd)||std::strlen(gate_fd)>6)throw std::runtime_error("owned start pipe required");
    const int fd=std::stoi(gate_fd);struct stat st{};if(fd<3||fstat(fd,&st)||!S_ISFIFO(st.st_mode))throw std::runtime_error("inherited start pipe required");
    pollfd p={fd,POLLIN,0};char go=0;if(poll(&p,1,10000)!=1||read(fd,&go,1)!=1||go!='G')throw std::runtime_error("owned GO admission missing within10s");close(fd);
    const std::filesystem::path root=argv[1];if(std::filesystem::exists(root))throw std::runtime_error("fresh output required");std::filesystem::create_directories(root);
    auto cpu=ggml_backend_cpu_init();if(!cpu)throw std::runtime_error("CPU init failed");
    ggml_backend_cpu_set_n_threads(cpu,10);ggml_backend_cpu_set_use_ref(cpu,false);
    if(ggml_backend_dev_type(ggml_backend_get_device(cpu))!=GGML_BACKEND_DEVICE_TYPE_CPU)throw std::runtime_error("ordinary CPU device required");
    const auto *traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);if(!traits||!traits->from_float||!traits->vec_dot||traits->nrows<1)throw std::runtime_error("native Q8 traits required");
    const int k=4096,m=8192;json rows=json::array();
    for(int n:sizes)for(unsigned seed:{5305u,713962u}){
        peak_rss();auto ctx=ggml_init({2u<<20,nullptr,true});if(!ctx)throw std::runtime_error("metadata allocation");
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,m),x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        ggml_set_name(w,"fixture.original.q8.weight");ggml_set_name(x,"fixture.input");
        auto full=ggml_mul_mat(ctx,w,x);auto fg=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(fg,full);
        const auto ref_parts=partition8(n),v4_parts=partition_v4(n);
        std::vector<ggml_cgraph *>rg,vg;std::vector<ggml_tensor *>ry,vy;
        auto create_parts=[&](const std::vector<int>&parts,std::vector<ggml_cgraph *>&graphs,std::vector<ggml_tensor *>&ys){int first=0;
            for(int take:parts){auto view=ggml_view_2d(ctx,x,k,take,x->nb[1],size_t(first)*x->nb[1]);auto y=ggml_mul_mat(ctx,w,view);
                auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);graphs.push_back(graph);ys.push_back(y);first+=take;}
            if(first!=n)throw std::runtime_error("partition coverage");};
        create_parts(ref_parts,rg,ry);create_parts(v4_parts,vg,vy);
        // Quantized weights are generated row by row; no full F32 weight copy.
        auto memory=ggml_backend_alloc_ctx_tensors(ctx,cpu);if(!memory)throw std::runtime_error("tensor allocation");
        const size_t allocated=ggml_backend_buffer_get_size(memory);if(allocated+(2u<<20)+(size_t(m)*n*8)+(size_t(k)*4)>(64u<<20))throw std::runtime_error("logical payload64MiB cap");
        random_state=seed;std::vector<float>line(k);
        for(int r=0;r<m;++r){for(float &v:line)v=next_float()*.05f;traits->from_float(line.data(),(char *)w->data+size_t(r)*w->nb[1],k);}
        for(size_t i=0;i<size_t(k)*n;++i)((float *)x->data)[i]=next_float();
        auto compute=[&](ggml_cgraph *graph){if(ggml_backend_graph_compute(cpu,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("native CPU graph failed");ggml_backend_synchronize(cpu);peak_rss();};
        compute(fg);std::vector<float>reference(size_t(m)*n),v4(size_t(m)*n);
        auto run_parts=[&](const std::vector<int>&parts,const std::vector<ggml_cgraph *>&graphs,const std::vector<ggml_tensor *>&ys,std::vector<float>&out){size_t first=0;
            for(size_t i=0;i<parts.size();++i){compute(graphs[i]);std::memcpy(out.data()+first*m,ys[i]->data,size_t(parts[i])*m*4);first+=parts[i];}};
        run_parts(ref_parts,rg,ry,reference);run_parts(v4_parts,vg,vy,v4);
        json v4_dispatch=json::array();for(int take:v4_parts)v4_dispatch.push_back(derived_dispatch(m,take,traits->nrows));
        rows.push_back({{"K",k},{"M",m},{"N",n},{"seed",seed},{"weight_type","q8_0"},
            {"actual_compute_backend",ggml_backend_name(cpu)},{"native_vec_dot_nrows",traits->nrows},{"native_vec_dot_type",ggml_type_name(traits->vec_dot_type)},
            {"weight_bytes",ggml_nbytes(w)},{"weight_sha256",digest(w->data,ggml_nbytes(w))},{"input_sha256",digest(x->data,ggml_nbytes(x))},
            {"reference8_parts",ref_parts},{"v4_parts",v4_parts},{"tensor_allocation_bytes",allocated},{"metadata_envelope_bytes",2u<<20},
            {"comparison_vectors_bytes",size_t(m)*n*8},{"generation_line_bytes",size_t(k)*4},
            {"fullN_vs_reference8",compare(reference.data(),(const float *)full->data,reference.size())},
            {"v4_vs_reference8",compare(reference.data(),v4.data(),reference.size())},
            {"fullN_dispatch_derived",derived_dispatch(m,n,traits->nrows)},{"v4_dispatch_derived",v4_dispatch},
            {"lifetime_peak_rss_bytes_after_case",peak_rss()},{"raw_float32_exported",false},
            {"scope","synthetic resident direct CPU Q8 only; no model/fence/other tensor proof"}});
        ggml_backend_buffer_free(memory);ggml_free(ctx);
    }
    const auto rss=peak_rss();ggml_backend_free(cpu);
    json result={{"status","completed_q8_tail_dispatch_diagnostic_only"},{"model_loaded",false},{"gpu_backend_initialized",false},{"gpu_matrix_dispatches",0},
        {"BLAS_initialized",false},{"scheduler_used",false},{"uses_reference_mode",false},{"cpu_threads",10},
        {"native_lifetime_peak_rss_bytes",rss},{"native_rss_cap_bytes",RSS_CAP},{"raw_float32_exported",false},
        {"raw_export_scope","bitcounts/maxabs/SHA compared in native memory; no F32 files exported"},
        {"sizes",sizes},{"seeds",{5305,713962}},{"case_count",rows.size()},{"rows",rows},
        {"full_model_qualified",false},{"performance_qualified",false},{"v4_model_fidelity_qualified",false}};
    std::ofstream f(root/"result.json");f.exceptions(std::ios::badbit|std::ios::failbit);f<<result.dump(2)<<'\n';f.close();std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}}
