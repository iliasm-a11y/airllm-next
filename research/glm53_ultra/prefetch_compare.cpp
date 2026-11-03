// Bounded original Q8_0[4096,8192] CPU10 component; never a model decoder.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <libproc.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t width=4096, rows=8192, weight_bytes=width*rows/32*34;
constexpr size_t scratch_bytes=size_t(8)<<20, range_cap=size_t(64)<<20;
using clock_type=std::chrono::steady_clock;
double seconds(clock_type::time_point a,clock_type::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
struct counters {
    rusage_info_v2 usage{};
    proc_taskinfo task{};
};
counters observe() {
    counters c;
    if(proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t *>(&c.usage)) ||
       proc_pidinfo(getpid(),PROC_PIDTASKINFO,0,&c.task,sizeof(c.task))!=sizeof(c.task))
        throw std::runtime_error("required owned-process counters unavailable");
    return c;
}
uint64_t difference(uint64_t a,uint64_t b) {
    if(b<a)throw std::runtime_error("process counter decreased");
    return b-a;
}
std::string delta_json(const counters & a,const counters & b,double timebase) {
    std::ostringstream o;
    const uint64_t user=difference(a.usage.ri_user_time,b.usage.ri_user_time);
    const uint64_t system=difference(a.usage.ri_system_time,b.usage.ri_system_time);
    o<<std::setprecision(17)<<"{\"os_storage_read_bytes\":"
     <<difference(a.usage.ri_diskio_bytesread,b.usage.ri_diskio_bytesread)
     <<",\"pageins\":"<<difference(a.usage.ri_pageins,b.usage.ri_pageins)
     <<",\"task_faults\":"<<difference(a.task.pti_faults,b.task.pti_faults)
     <<",\"task_pageins\":"<<difference(a.task.pti_pageins,b.task.pti_pageins)
     <<",\"cpu_core_seconds\":"<<(double(user)+double(system))*timebase
     <<",\"resident_bytes_after\":"<<b.usage.ri_resident_size<<"}";
    return o.str();
}
struct resident {
    size_t pages=0,present=0;
};
resident residency(void * mapping,size_t bytes,size_t page) {
    resident r;r.pages=(bytes+page-1)/page;
    std::vector<char> vector(r.pages);
    if(mincore(static_cast<char *>(mapping),bytes,vector.data()))
        throw std::runtime_error("mincore failed");
    for(const auto value:vector)if(value&MINCORE_INCORE)++r.present;
    return r;
}
std::string resident_json(resident r) {
    return "{\"mapping_pages\":"+std::to_string(r.pages)+
           ",\"resident_pages\":"+std::to_string(r.present)+"}";
}
bool identity_equal(const struct stat & a,const struct stat & b) {
    return a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_size==b.st_size &&
        a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec &&
        a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec;
}
void verify_identity(int fd,const char * path,const struct stat & expected) {
    struct stat opened{},current{};
    if(fstat(fd,&opened)||stat(path,&current)||!identity_equal(expected,opened)||!identity_equal(expected,current))
        throw std::runtime_error("original opened/path file identity changed");
}
struct resources {
    int fd=-1;void * mapping=MAP_FAILED;size_t mapped_bytes=0;
    ggml_backend_t cpu=nullptr;
    struct ggml_threadpool * pool=nullptr;
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t weights=nullptr,allocation=nullptr;
    ~resources() {
        if(allocation)ggml_backend_buffer_free(allocation);
        if(weights)ggml_backend_buffer_free(weights);
        if(ctx)ggml_free(ctx);
        if(cpu)ggml_backend_free(cpu);
        if(pool)ggml_threadpool_free(pool);
        if(mapping!=MAP_FAILED)munmap(mapping,mapped_bytes);
        if(fd>=0)close(fd);
    }
};
} // namespace

int main(int argc,char ** argv) {try {
    if(argc!=8)throw std::runtime_error("usage: prefetch-compare shard absolute_offset activation.f32 output_dir none|willneed|pread standard|nocache phases");
    const uint64_t offset=std::stoull(argv[2]);
    const std::string requested_mode(argv[5]),mapping_mode(argv[6]);const int phases=std::stoi(argv[7]);
    if((requested_mode!="none"&&requested_mode!="willneed"&&requested_mode!="pread")||
       (mapping_mode!="standard"&&mapping_mode!="nocache")||phases<2||phases>4||weight_bytes>range_cap)
        throw std::runtime_error("bounded mode/mapping/phases required");
    const std::filesystem::path directory(argv[4]);
    if(std::filesystem::exists(directory))throw std::runtime_error("refusing existing native evidence");
    std::filesystem::create_directories(directory);
    resources r;r.fd=open(argv[1],O_RDONLY);struct stat original{};
    if(r.fd<0||fstat(r.fd,&original)||!S_ISREG(original.st_mode)||original.st_size<0||
       offset>uint64_t(original.st_size)||weight_bytes>uint64_t(original.st_size)-offset)
        throw std::runtime_error("weight range exceeds regular source file");
    const long page_value=sysconf(_SC_PAGESIZE);if(page_value<=0)throw std::runtime_error("page size unavailable");
    const size_t page=size_t(page_value);const uint64_t base=offset-offset%page;
    const size_t displacement=size_t(offset-base);r.mapped_bytes=displacement+weight_bytes;
    if(r.mapped_bytes>range_cap)throw std::runtime_error("mapped range exceeds64MiB");
    const int flags=MAP_SHARED|(mapping_mode=="nocache"?MAP_NOCACHE:0);
    r.mapping=mmap(nullptr,r.mapped_bytes,PROT_READ,flags,r.fd,off_t(base));
    if(r.mapping==MAP_FAILED)throw std::runtime_error("original readonly mapping failed");
    std::vector<float> activation(width),first(rows),actual(rows);
    std::vector<unsigned char> scratch(scratch_bytes); // Same fixed resident scratch in every arm.
    if(std::filesystem::file_size(argv[3])!=width*sizeof(float))throw std::runtime_error("activation byte count mismatch");
    std::ifstream input(argv[3],std::ios::binary);input.read(reinterpret_cast<char *>(activation.data()),width*sizeof(float));
    if(!input||!std::all_of(activation.begin(),activation.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("activation missing or nonfinite");
    r.cpu=ggml_backend_cpu_init();if(!r.cpu)throw std::runtime_error("CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(r.cpu,10);
    auto pool_params=ggml_threadpool_params_default(10);r.pool=ggml_threadpool_new(&pool_params);
    if(!r.pool)throw std::runtime_error("CPU10 pool unavailable");
    ggml_backend_cpu_set_threadpool(r.cpu,r.pool);
    ggml_init_params params{1<<20,nullptr,true};r.ctx=ggml_init(params);
    if(!r.ctx)throw std::runtime_error("GGML context unavailable");
    auto weight=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,width,rows);
    r.weights=ggml_backend_cpu_buffer_from_ptr(r.mapping,r.mapped_bytes);
    if(!r.weights)throw std::runtime_error("weight wrapper unavailable");
    ggml_backend_buffer_set_usage(r.weights,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight->buffer=r.weights;weight->data=static_cast<char *>(r.mapping)+displacement;
    auto x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,width,1);ggml_set_input(x);
    auto result=ggml_mul_mat(r.ctx,weight,x);auto graph=ggml_new_graph_custom(r.ctx,8,false);
    ggml_build_forward_expand(graph,result);r.allocation=ggml_backend_alloc_ctx_tensors(r.ctx,r.cpu);
    if(!r.allocation)throw std::runtime_error("CPU workspace allocation failed");
    ggml_backend_tensor_set(x,activation.data(),0,width*sizeof(float));
    mach_timebase_info_data_t timebase{};
    if(mach_timebase_info(&timebase)||!timebase.denom)throw std::runtime_error("CPU timebase unavailable");
    const double tick_seconds=double(timebase.numer)/double(timebase.denom)/1e9;
    std::cout<<std::setprecision(17);
    for(int phase=0;phase<phases;++phase) {
        const std::string mode=phase==0?requested_mode:"none"; // Later phases are warm exact references.
        std::cout<<"{\"ready\":"<<phase<<",\"mode\":\""<<mode<<"\",\"mapping\":\""<<mapping_mode<<"\"}"<<std::endl;
        std::string command;if(!std::getline(std::cin,command)||command!="run")throw std::runtime_error("phase authorization absent");
        verify_identity(r.fd,argv[1],original);
        const auto before=observe();const auto resident_before=residency(r.mapping,r.mapped_bytes,page);
        size_t completed=0,read_calls=0;
        const auto start=clock_type::now();
        if(mode=="willneed") {
            const int status=posix_madvise(r.mapping,r.mapped_bytes,POSIX_MADV_WILLNEED);
            if(status)throw std::runtime_error("WILLNEED rejected: "+std::to_string(status));
        } else if(mode=="pread") {
            while(completed<weight_bytes) {
                const size_t wanted=std::min(scratch.size(),weight_bytes-completed);
                const ssize_t got=pread(r.fd,scratch.data(),wanted,off_t(offset+completed));
                if(got<0&&errno==EINTR)continue;
                if(got<=0||size_t(got)>wanted)throw std::runtime_error("bounded normal pread failed/short EOF");
                completed+=size_t(got);if(++read_calls>4096)throw std::runtime_error("pread call budget exceeded");
            }
        }
        const auto preparation_end=clock_type::now();const auto after_preparation=observe();
        const auto resident_after_preparation=residency(r.mapping,r.mapped_bytes,page);
        const auto compute_start=clock_type::now();const auto status=ggml_backend_graph_compute(r.cpu,graph);
        const auto end=clock_type::now();const auto after_compute=observe();
        const auto resident_after_compute=residency(r.mapping,r.mapped_bytes,page);
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML compute failed");
        verify_identity(r.fd,argv[1],original);
        ggml_backend_tensor_get(result,actual.data(),0,rows*sizeof(float));
        if(!std::all_of(actual.begin(),actual.end(),[](float value){return std::isfinite(value);}))
            throw std::runtime_error("nonfinite output");
        const bool equal=phase==0||!std::memcmp(first.data(),actual.data(),rows*sizeof(float));
        if(phase==0)first=actual;
        const std::string name="outputs-"+std::to_string(phase)+".f32";
        std::ofstream output(directory/name,std::ios::binary);output.exceptions(std::ios::failbit|std::ios::badbit);
        output.write(reinterpret_cast<const char *>(actual.data()),rows*sizeof(float));output.close();
        std::cout<<"{\"phase\":"<<phase<<",\"mode\":\""<<mode<<"\",\"mapping\":\""<<mapping_mode
            <<"\",\"logical_weight_bytes\":"<<weight_bytes<<",\"mapped_bytes\":"<<r.mapped_bytes
            <<",\"page_size_bytes\":"<<page
            <<",\"scratch_bytes\":"<<scratch_bytes<<",\"pread_completed_bytes\":"<<completed
            <<",\"pread_calls\":"<<read_calls<<",\"preparation_wall_seconds\":"<<seconds(start,preparation_end)
            <<",\"compute_wall_seconds\":"<<seconds(compute_start,end)
            <<",\"preparation_plus_compute_seconds\":"<<seconds(start,preparation_end)+seconds(compute_start,end)
            <<",\"observation_gap_seconds\":"<<seconds(preparation_end,compute_start)
            <<",\"total_observed_sequence_seconds\":"<<seconds(start,end)
            <<",\"residency_before\":"<<resident_json(resident_before)
            <<",\"residency_after_preparation\":"<<resident_json(resident_after_preparation)
            <<",\"residency_after_compute\":"<<resident_json(resident_after_compute)
            <<",\"preparation_process_delta\":"<<delta_json(before,after_preparation,tick_seconds)
            <<",\"compute_process_delta\":"<<delta_json(after_preparation,after_compute,tick_seconds)
            <<",\"total_process_delta\":"<<delta_json(before,after_compute,tick_seconds)
            <<",\"all_outputs_finite\":true,\"outputs_equal_first\":"<<(equal?"true":"false")
            <<",\"file_identity_unchanged\":true,\"output_file\":\""<<name<<"\"}"<<std::endl;
        if(!equal)throw std::runtime_error("warm reference differs in output bits");
    }
    verify_identity(r.fd,argv[1],original);
    std::cout<<"{\"complete\":true,\"outputs_equal_warm_reference\":true,\"file_identity_unchanged\":true,\"all_outputs_finite\":true}"<<std::endl;
    return 0;
} catch(const std::exception & error) {std::cerr<<error.what()<<'\n';return 2;}}
