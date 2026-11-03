// Tiny real GGML readonly mmap/user-lock lifecycle fixture; no model/GPU.
#include "dense_lock.h"
#include "ggml-cpu.h"
#include <fcntl.h>
#include <stdexcept>
#include <thread>

using namespace airllm_dense_lock;
static int checks=0;
static void check(bool ok){++checks;if(!ok){std::fprintf(stderr,"dense_lock check%d failed\n",checks);std::exit(6);}}
struct mapping {
    void * data=nullptr;size_t bytes=0;bool * alive=nullptr;
    explicit mapping(size_t length,bool * observed=nullptr):bytes(length),alive(observed){
        char path[]="/tmp/airllm-dense-lock-XXXXXX";const int fd=mkstemp(path);
        check(fd>=0&&ftruncate(fd,length)==0);data=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);
        unlink(path);close(fd);check(data!=MAP_FAILED);if(alive)*alive=true;
    }
    ~mapping(){munmap(data,bytes);if(alive)*alive=false;}
};
struct buffer {
    ggml_backend_buffer_t value;bool * alive;
    buffer(void * data,size_t bytes,bool * observed=nullptr):value(ggml_backend_cpu_buffer_from_ptr(data,bytes)),alive(observed){
        ggml_backend_buffer_set_usage(value,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);if(alive)*alive=true;
    }
    ~buffer(){ggml_backend_buffer_free(value);if(alive)*alive=false;}
};
static unsigned wired(void * data){
    mach_vm_address_t address=reinterpret_cast<uintptr_t>(data);mach_vm_size_t length=0;
    vm_region_basic_info_data_64_t info{};mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64;mach_port_t object=MACH_PORT_NULL;
    const auto rc=mach_vm_region(mach_task_self(),&address,&length,VM_REGION_BASIC_INFO_64,reinterpret_cast<vm_region_info_t>(&info),&count,&object);
    if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
    check(rc==KERN_SUCCESS);return info.user_wired_count;
}
struct injected {
    int lock_calls=0,unlock_calls=0,admit_calls=0,cancel_calls=0;
    int fail_lock_on=0,cancel_on=0,deny_admit_on=0;
    bool throw_after_lock=false,fail_unlock=false;
    bool * map_alive=nullptr,* buffer_alive=nullptr;
};
static bool admit_hook(size_t,void * data){auto & s=*static_cast<injected *>(data);return ++s.admit_calls!=s.deny_admit_on;}
static bool cancel_hook(void * data){auto & s=*static_cast<injected *>(data);return s.cancel_on&&++s.cancel_calls>=s.cancel_on;}
static int lock_hook(void * data,size_t bytes,void * user){
    auto & s=*static_cast<injected *>(user);++s.lock_calls;
    if(s.lock_calls==s.fail_lock_on){errno=ENOMEM;return -1;}
    const int rc=mlock(data,bytes);if(!rc&&s.throw_after_lock)throw std::runtime_error("injected after real mlock");return rc;
}
static int unlock_hook(void * data,size_t bytes,void * user){
    auto & s=*static_cast<injected *>(user);++s.unlock_calls;
    if(s.map_alive)check(*s.map_alive);if(s.buffer_alive)check(*s.buffer_alive);
    if(s.fail_unlock){errno=EIO;return -1;}return munlock(data,bytes);
}
static hooks injected_hooks(injected & s){return {admit_hook,cancel_hook,lock_hook,unlock_hook,&s};}
static void names(const char * value){setenv("AIRLLM_DENSE_LOCK_NAMES",value,1);}

int main(int argc,char ** argv){
    const size_t page=size_t(sysconf(_SC_PAGESIZE));struct rlimit before{},after{};check(getrlimit(RLIMIT_MEMLOCK,&before)==0);
    mapping file(page*6);buffer original(file.data,file.bytes);ggml_init_params params{65536,nullptr,true};auto ctx=ggml_init(params);
    auto w0=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2),w1=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,32,2);
    auto other=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2);
    w0->data=static_cast<char *>(file.data)+32;w1->data=static_cast<char *>(file.data)+512;other->data=static_cast<char *>(file.data)+4*page+32;
    for(auto * w:{w0,w1,other})w->buffer=original.value;
    std::vector<std::pair<std::string,ggml_tensor *>> tensors={{"blk.0.hc_attn_fn.weight",w0},{"blk.0.ssm_f_a.weight",w1},{"blk.0.hc_ffn_fn.weight",other}};
    std::vector<range> owned={{reinterpret_cast<uintptr_t>(file.data),reinterpret_cast<uintptr_t>(file.data)+file.bytes}};
    names("blk.0.hc_attn_fn.weight,blk.0.ssm_f_a.weight");setenv("AIRLLM_DENSE_LOCK_MIB","128",1);
    injected s;unsetenv("AIRLLM_DENSE_LOCK");{owner disabled;check(!disabled.acquire(tensors,owned,injected_hooks(s))&&s.lock_calls==0);}
    setenv("AIRLLM_DENSE_LOCK","1",1);
    if(argc==2&&!std::strcmp(argv[1],"--poison")){
        names("blk.0.hc_attn_fn.weight");s.fail_unlock=true;
        {owner held;check(held.acquire(tensors,owned,injected_hooks(s)));}
        check(poisoned());names("blk.0.hc_ffn_fn.weight");injected next;
        {owner denied;check(!denied.acquire(tensors,owned,injected_hooks(next))&&next.lock_calls==0);check(!std::strcmp(denied.result().status,"prior_unlock_incomplete"));}
        check(munlock(file.data,page)==0);check(getrlimit(RLIMIT_MEMLOCK,&after)==0&&before.rlim_cur==after.rlim_cur&&before.rlim_max==after.rlim_max);
        std::printf("{\"status\":\"passed\",\"mode\":\"poison\",\"checks\":%d,\"new_budget_declined\":true,\"rlimit_unchanged\":true}\n",checks);ggml_free(ctx);return 0;
    }
    {
        owner held;check(held.acquire(tensors,owned,injected_hooks(s)));const auto r=held.result();
        check(r.retained&&r.raw_bytes==384&&r.union_bytes==page&&r.locked_bytes==page&&r.lock_calls==1&&r.resident_pages==1);
        check(wired(file.data)>0);names("blk.0.hc_ffn_fn.weight");
        std::thread concurrent([&]{injected next;owner denied;check(!denied.acquire(tensors,owned,injected_hooks(next))&&next.lock_calls==0);});concurrent.join();
        held.release();check(held.result().locked_bytes==0&&!held.result().retained&&wired(file.data)==0);
    }
    names("blk.0.hc_attn_fn.weight,blk.0.ssm_f_a.weight");
    {injected denied;denied.deny_admit_on=1;owner held;check(!held.acquire(tensors,owned,injected_hooks(denied))&&denied.lock_calls==0);}
    {injected denied;denied.deny_admit_on=2;owner held;check(!held.acquire(tensors,owned,injected_hooks(denied))&&denied.lock_calls==1&&held.result().locked_bytes==0);}
    {injected cancel;cancel.cancel_on=1;owner held;check(!held.acquire(tensors,owned,injected_hooks(cancel))&&held.result().cancelled&&cancel.lock_calls==0);}
    w1->data=static_cast<char *>(file.data)+2*page+32;
    {injected fail;fail.fail_lock_on=2;owner held;check(!held.acquire(tensors,owned,injected_hooks(fail)));check(fail.lock_calls==2&&fail.unlock_calls==2&&held.result().locked_bytes==0&&wired(file.data)==0);}
    {injected fail;fail.throw_after_lock=true;owner held;check(!held.acquire(tensors,owned,injected_hooks(fail)));check(fail.unlock_calls==1&&held.result().locked_bytes==0&&wired(file.data)==0);}
    {injected cancel;cancel.cancel_on=2;owner held;check(!held.acquire(tensors,owned,injected_hooks(cancel))&&held.result().cancelled&&cancel.lock_calls==1&&wired(file.data)==0);}
    {owner bad;check(!bad.acquire(tensors,{},injected_hooks(s))&&bad.result().lock_calls==0);}
    names("blk.0.hc_attn_fn.weight,blk.0.hc_attn_fn.weight");{owner bad;check(!bad.acquire(tensors,owned,injected_hooks(s))&&bad.result().lock_calls==0);}
    names("blk.0.attn_q.weight");{owner bad;check(!bad.acquire(tensors,owned,injected_hooks(s))&&bad.result().lock_calls==0);}
    names("blk.0.hc_attn_fn.weight");
    w0->flags=GGML_TENSOR_FLAG_INPUT;{owner bad;check(!bad.acquire(tensors,owned,injected_hooks(s))&&bad.result().lock_calls==0);}w0->flags=0;
    void * writable=mmap(nullptr,page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    {buffer modified(writable,page);auto saved=*w0;w0->data=writable;w0->buffer=modified.value;
     std::vector<range> view={{reinterpret_cast<uintptr_t>(writable),reinterpret_cast<uintptr_t>(writable)+page}};
     owner bad;check(!bad.acquire(tensors,view,injected_hooks(s))&&bad.result().lock_calls==0);*w0=saved;}
    munmap(writable,page);
    // Sparse virtual mapping: raw128MiB fits but its edge pages exceed128MiB.
    {mapping sparse((size_t(128)<<20)+2*page);buffer large(sparse.data,sparse.bytes);
     auto huge=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,8192);huge->data=static_cast<char *>(sparse.data)+32;huge->buffer=large.value;
     std::vector<std::pair<std::string,ggml_tensor *>> one={{"blk.0.hc_attn_fn.weight",huge}};
     std::vector<range> view={{reinterpret_cast<uintptr_t>(sparse.data),reinterpret_cast<uintptr_t>(sparse.data)+sparse.bytes}};
     injected probe;owner bad;check(!bad.acquire(one,view,injected_hooks(probe))&&probe.lock_calls==0);check(!std::strcmp(bad.result().status,"page_union_over_budget"));}
    bool map_alive=false,buffer_alive=false;injected lifetime;lifetime.map_alive=&map_alive;lifetime.buffer_alive=&buffer_alive;
    {struct model {mapping pages;buffer buf;owner retention;model(size_t p,bool * m,bool * b):pages(p,m),buf(pages.data,p,b){}} tiny(page,&map_alive,&buffer_alive);
     auto copy=*w0;copy.data=static_cast<char *>(tiny.pages.data)+32;copy.buffer=tiny.buf.value;
     std::vector<std::pair<std::string,ggml_tensor *>> one={{"blk.0.hc_attn_fn.weight",&copy}};
     std::vector<range> view={{reinterpret_cast<uintptr_t>(tiny.pages.data),reinterpret_cast<uintptr_t>(tiny.pages.data)+page}};
     check(tiny.retention.acquire(one,view,injected_hooks(lifetime)));}
    check(!map_alive&&!buffer_alive&&lifetime.unlock_calls==1);
    check(getrlimit(RLIMIT_MEMLOCK,&after)==0&&before.rlim_cur==after.rlim_cur&&before.rlim_max==after.rlim_max);
    std::printf("{\"status\":\"passed\",\"mode\":\"normal\",\"checks\":%d,\"real_mlock_mincore_release\":true,\"page_union_budget_enforced\":true,\"lifetime_order_verified\":true,\"rlimit_unchanged\":true,\"model_or_gpu\":false}\n",checks);
    ggml_free(ctx);return 0;
}
