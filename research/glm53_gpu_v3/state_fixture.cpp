#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <CommonCrypto/CommonDigest.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include "state_capture.inc"
int main(){
 auto cpu=ggml_backend_cpu_init(); if(!cpu)return 2;
 ggml_init_params params{1024*1024,nullptr,true};auto ctx=ggml_init(params);
 auto base=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,4);ggml_set_name(base,"cache_s_l0");
 auto dst=ggml_view_2d(ctx,base,2,2,16,0);auto src=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2,2);auto copy=ggml_cpy(ctx,src,dst);
 auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,copy);
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);float zeros[16]{};float values[4]={1.0f,-0.0f,3.0f,4.0f};
 ggml_backend_tensor_set(base,zeros,0,sizeof(zeros));ggml_backend_tensor_set(src,values,0,sizeof(values));
 if(ggml_backend_graph_compute(cpu,graph)!=GGML_STATUS_SUCCESS)return 3;
 std::puts(state_event(copy,0).c_str());
 values[1]=NAN;ggml_backend_tensor_set(src,values,0,sizeof(values));
 if(ggml_backend_graph_compute(cpu,graph)!=GGML_STATUS_SUCCESS)return 4;
 std::puts(state_event(copy,1).c_str());
 ggml_tensor empty{};empty.type=GGML_TYPE_F32;empty.op=GGML_OP_CPY;empty.ne[0]=4;empty.ne[1]=0;empty.ne[2]=empty.ne[3]=1;empty.nb[0]=4;empty.nb[1]=16;empty.src[1]=base;
 std::puts(state_event(&empty,2).c_str());
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(cpu);return 0;
}
