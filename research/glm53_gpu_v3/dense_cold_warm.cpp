// Fixed real Q8_0[4096,8192] CPU10 diagnostic. No model decoder/GPU.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <libproc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

static rusage_info_v2 usage() {
    rusage_info_v2 r{};
    if(proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t *>(&r)))
        throw std::runtime_error("process IO counters unavailable");
    return r;
}
int main(int argc,char ** argv) {try {
    if(argc!=6)throw std::runtime_error("usage: dense-cold-warm shard absolute_offset activation.f32 output_dir phases");
    constexpr size_t width=4096,rows=8192,weight_bytes=width*rows/32*34;
    const uint64_t offset=std::stoull(argv[2]);const int phases=std::stoi(argv[5]);
    if(phases<2||phases>8)throw std::runtime_error("phases must be2..8");
    const std::filesystem::path directory(argv[4]);
    if(std::filesystem::exists(directory))throw std::runtime_error("refusing existing native output directory");
    std::filesystem::create_directories(directory);
    const int fd=open(argv[1],O_RDONLY);struct stat st{};
    if(fd<0||fstat(fd,&st)||offset>uint64_t(st.st_size)||weight_bytes>uint64_t(st.st_size)-offset)
        throw std::runtime_error("weight range outside source file");
    const size_t page=size_t(sysconf(_SC_PAGESIZE));const uint64_t base=offset-offset%page;
    const size_t delta=size_t(offset-base),mapped_bytes=delta+weight_bytes;
    if(mapped_bytes>(128U<<20))throw std::runtime_error("mapping exceeds128MiB");
    void * mapping=mmap(nullptr,mapped_bytes,PROT_READ,MAP_SHARED,fd,off_t(base));
    if(mapping==MAP_FAILED)throw std::runtime_error("readonly mmap failed");
    void * original=static_cast<char *>(mapping)+delta;
    std::vector<float> activation(width);
    if(std::filesystem::file_size(argv[3])!=width*sizeof(float))throw std::runtime_error("activation byte count mismatch");
    std::ifstream input(argv[3],std::ios::binary);input.read(reinterpret_cast<char *>(activation.data()),width*sizeof(float));
    if(!input||!std::all_of(activation.begin(),activation.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("activation missing or nonfinite");
    auto cpu=ggml_backend_cpu_init();if(!cpu)throw std::runtime_error("CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(cpu,10);
    auto pool_params=ggml_threadpool_params_default(10);auto pool=ggml_threadpool_new(&pool_params);
    if(!pool)throw std::runtime_error("CPU10 threadpool unavailable");
    ggml_backend_cpu_set_threadpool(cpu,pool);
    ggml_init_params params{1<<20,nullptr,true};auto ctx=ggml_init(params);
    auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,width,rows);
    // Wrap the page-aligned mapping base, preserving the tensor's exact GGUF
    // byte offset; tensor alignment need not equal the buffer-base alignment.
    auto weights=ggml_backend_cpu_buffer_from_ptr(mapping,mapped_bytes);
    ggml_backend_buffer_set_usage(weights,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);w->buffer=weights;w->data=original;
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,1);ggml_set_input(x);
    auto result=ggml_mul_mat(ctx,w,x);auto graph=ggml_new_graph_custom(ctx,8,false);ggml_build_forward_expand(graph,result);
    auto allocation=ggml_backend_alloc_ctx_tensors(ctx,cpu);if(!allocation)throw std::runtime_error("CPU allocation failed");
    ggml_backend_tensor_set(x,activation.data(),0,width*sizeof(float));
    std::vector<float> baseline(rows),actual(rows);
    std::cout<<std::setprecision(17);
    for(int phase=0;phase<phases;++phase){
        std::cout<<"{\"ready\":"<<phase<<",\"mapped_weight_bytes\":"<<mapped_bytes<<"}"<<std::endl;
        std::string command;if(!std::getline(std::cin,command)||command!="run")throw std::runtime_error("phase authorization absent");
        const auto before=usage();const auto start=std::chrono::steady_clock::now();
        const auto status=ggml_backend_graph_compute(cpu,graph);
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const auto after=usage();if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU graph failed");
        ggml_backend_tensor_get(result,actual.data(),0,rows*sizeof(float));
        if(!std::all_of(actual.begin(),actual.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("nonfinite output");
        const bool equal=phase==0||!std::memcmp(actual.data(),baseline.data(),rows*sizeof(float));
        if(phase==0)baseline=actual;
        const std::string name="outputs-"+std::to_string(phase)+".f32";
        std::ofstream output(directory/name,std::ios::binary);output.exceptions(std::ios::failbit|std::ios::badbit);
        output.write(reinterpret_cast<const char *>(actual.data()),rows*sizeof(float));output.close();
        std::cout<<"{\"phase\":"<<phase<<",\"kernel_wall_seconds\":"<<wall
            <<",\"process_os_read_bytes\":"<<after.ri_diskio_bytesread-before.ri_diskio_bytesread
            <<",\"process_pageins\":"<<after.ri_pageins-before.ri_pageins
            <<",\"output_file\":\""<<name<<"\",\"outputs_equal_first\":"<<(equal?"true":"false")<<"}"<<std::endl;
        if(!equal)throw std::runtime_error("warm output bit mismatch");
    }
    ggml_backend_buffer_free(allocation);ggml_backend_buffer_free(weights);ggml_free(ctx);ggml_backend_free(cpu);ggml_threadpool_free(pool);
    munmap(mapping,mapped_bytes);close(fd);return 0;
}catch(const std::exception & error){std::cerr<<error.what()<<'\n';return 2;}}
