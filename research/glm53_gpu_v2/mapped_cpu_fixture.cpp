// Tiny real Metal-mapped constant -> CPU read preflight. No GPU compute/model.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "mapped_cpu_guard.inc"

int main(){
    ggml_backend_load_all();
    auto metal_device=ggml_backend_dev_by_name("MTL0");
    if(!metal_device){std::puts("{\"status\":\"skipped\",\"reason\":\"Metal_device_unavailable\"}");return 77;}
    auto cpu=ggml_backend_cpu_init();if(!cpu)return 2;
    const size_t page=sysconf(_SC_PAGESIZE);
    std::vector<unsigned char> packed(68,0);
    for(int row=0;row<2;++row){
        packed[row*34]=0;packed[row*34+1]=0x3c; // IEEE half1.0
        for(int i=0;i<32;++i)packed[row*34+2+i]=static_cast<unsigned char>(int8_t(i-16+row));
    }
    char path[]="/tmp/airllm-mapped-cpu-fixture-XXXXXX";
    const int fd=mkstemp(path);if(fd<0||ftruncate(fd,page)||pwrite(fd,packed.data(),packed.size(),0)!=ssize_t(packed.size()))return 3;
    void* original=mmap(nullptr,page,PROT_READ,MAP_PRIVATE,fd,0);unlink(path);close(fd);
    if(original==MAP_FAILED)return 4;
    auto mapped=ggml_backend_dev_buffer_from_host_ptr(metal_device,original,page,packed.size());
    if(!mapped)return 5;
    ggml_backend_buffer_set_usage(mapped,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_init_params params{1<<20,nullptr,true};auto ctx=ggml_init(params);
    auto weight=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,32,2);
    weight->data=original;weight->buffer=mapped;
    int checks=0;
    auto check=[&](bool value){++checks;if(!value){std::fprintf(stderr,"failed check%d\n",checks);std::exit(6);}};
    unsetenv("AIRLLM_METAL_CPU_MAPPED_WEIGHTS");unsetenv("AIRLLM_GLM_Q8_ONLY");
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));
    setenv("AIRLLM_METAL_CPU_MAPPED_WEIGHTS","1",1);
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));
    setenv("AIRLLM_GLM_Q8_ONLY","1",1);
    check(airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));
    check(ggml_backend_buffer_get_base(mapped)==original);
    check(!ggml_backend_buffer_is_host(mapped));
    ggml_backend_buffer_set_usage(mapped,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));
    ggml_backend_buffer_set_usage(mapped,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight->flags=GGML_TENSOR_FLAG_INPUT;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));weight->flags=0;
    weight->op=GGML_OP_DUP;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));weight->op=GGML_OP_NONE;
    // Exercise the depth boundary explicitly. Normal GGML view constructors
    // flatten aliases; linked tensor metadata lets the real guard see every hop.
    ggml_tensor aliases[17];
    for(int i=0;i<17;++i){
        aliases[i]=*weight;
        aliases[i].op=GGML_OP_VIEW;
        aliases[i].view_src=i ? &aliases[i-1] : weight;
        aliases[i].src[0]=aliases[i].view_src;
    }
    const int before_alias_checks=checks;
    check(airllm_metal_cpu_mapped_weight_supported(cpu,&aliases[14],mapped));
    check(airllm_metal_cpu_mapped_weight_supported(cpu,&aliases[15],mapped));
    const int mutable_flags[]={GGML_TENSOR_FLAG_INPUT,GGML_TENSOR_FLAG_OUTPUT,GGML_TENSOR_FLAG_COMPUTE};
    for(int flag:mutable_flags){
        weight->flags=flag;
        check(!airllm_metal_cpu_mapped_weight_supported(cpu,&aliases[15],mapped));
    }
    weight->flags=0;
    aliases[0].flags=GGML_TENSOR_FLAG_COMPUTE;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,&aliases[15],mapped));
    aliases[0].flags=0;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,&aliases[16],mapped));
    const int alias_depth_checks=checks-before_alias_checks;
    weight->data=static_cast<char*>(original)+page;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,mapped));weight->data=original;
    // Even the mapped type's owned allocation can be writable: reject it.
    auto writable=ggml_backend_buft_alloc_buffer(ggml_backend_buffer_get_type(mapped),page);
    ggml_backend_buffer_set_usage(writable,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight->data=ggml_backend_buffer_get_base(writable);weight->buffer=writable;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,writable));
    // Normal shared/private allocation types are not the mapped type.
    auto allocated=ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(metal_device),page);
    ggml_backend_buffer_set_usage(allocated,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight->data=ggml_backend_buffer_get_base(allocated);weight->buffer=allocated;
    check(!airllm_metal_cpu_mapped_weight_supported(cpu,weight,allocated));
    weight->data=original;weight->buffer=mapped;
    auto copied=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,32,2);
    auto copy_buffer=ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),packed.size());
    copied->buffer=copy_buffer;copied->data=ggml_backend_buffer_get_base(copy_buffer);
    ggml_backend_tensor_set(copied,packed.data(),0,packed.size());
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,1);
    auto xbuffer=ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),32*sizeof(float));
    x->buffer=xbuffer;x->data=ggml_backend_buffer_get_base(xbuffer);
    std::vector<float> values(32,1.0f);ggml_backend_tensor_set(x,values.data(),0,128);
    auto direct=ggml_mul_mat(ctx,weight,x);auto reference=ggml_mul_mat(ctx,copied,x);
    auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,direct);ggml_build_forward_expand(graph,reference);
    auto outputs=ggml_backend_alloc_ctx_tensors(ctx,cpu);
    check(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS);
    float actual[2],expected[2];ggml_backend_tensor_get(direct,actual,0,sizeof(actual));
    ggml_backend_tensor_get(reference,expected,0,sizeof(expected));
    check(std::memcmp(actual,expected,sizeof(actual))==0);
    std::printf("{\"status\":\"passed\",\"checks\":%d,\"alias_depth_checks\":%d,\"sixteen_alias_leaf_flags_rejected\":true,\"seventeen_aliases_rejected\":true,\"buffer\":\"%s\",\"base_is_original_mapping\":true,\"cpu_direct_outputs_bit_identical\":true,\"scheduler_copy_elimination\":\"not_tested\"}\n",checks,alias_depth_checks,ggml_backend_buffer_name(mapped));
    ggml_backend_buffer_free(outputs);ggml_backend_buffer_free(copy_buffer);ggml_backend_buffer_free(xbuffer);
    ggml_backend_buffer_free(allocated);ggml_backend_buffer_free(writable);ggml_backend_buffer_free(mapped);
    ggml_free(ctx);ggml_backend_free(cpu);munmap(original,page);
    return 0;
}
