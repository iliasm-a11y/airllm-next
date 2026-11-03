// Default-off GLM dense CPU retention: original readonly mmap pages only.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>

namespace airllm_dense_lock {
struct range {uintptr_t begin=0,end=0;};
struct outcome {
    bool retained=false,cancelled=false;
    size_t names=0,raw_bytes=0,union_bytes=0,locked_bytes=0,lock_calls=0,unlock_calls=0,resident_pages=0;
    int error=0,unlock_error=0;
    const char * status="disabled";
};
struct hooks {
    bool (*admit)(size_t,void *)=nullptr;
    bool (*cancel)(void *)=nullptr;
    int (*lock)(void *,size_t,void *)=nullptr;
    int (*unlock)(void *,size_t,void *)=nullptr;
    void * data=nullptr;
};
class owner;
inline std::mutex & mutex(){static std::mutex value;return value;}
inline owner * & active(){static owner * value=nullptr;return value;}
inline bool & poisoned(){static bool value=false;return value;}
inline bool enabled(){const char * p=std::getenv("AIRLLM_DENSE_LOCK");return p&&!std::strcmp(p,"1");}

inline bool readonly_unwired(uintptr_t begin,uintptr_t end) {
    if(!begin||begin>=end)return false;
    for(int i=0;begin<end&&i<64;++i){
        mach_vm_address_t address=begin;mach_vm_size_t length=0;
        vm_region_basic_info_data_64_t info{};mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object=MACH_PORT_NULL;
        const auto rc=mach_vm_region(mach_task_self(),&address,&length,VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info),&count,&object);
        if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
        if(rc!=KERN_SUCCESS||address>begin||!length||address>UINTPTR_MAX-length||address+length<=begin||
           !(info.protection&VM_PROT_READ)||(info.protection&VM_PROT_WRITE)||info.user_wired_count)return false;
        begin=std::min<uintptr_t>(end,address+length);
    }
    return begin==end;
}
inline bool admissible(size_t bytes) {
    int pressure=0;size_t length=sizeof(pressure);
    if(sysctlbyname("kern.memorystatus_vm_pressure_level",&pressure,&length,nullptr,0)||pressure!=1)return false;
    vm_statistics64_data_t vm{};mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    const auto host=mach_host_self();const auto rc=host_statistics64(host,HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&vm),&count);
    mach_port_deallocate(mach_task_self(),host);const long page=sysconf(_SC_PAGESIZE);
    return rc==KERN_SUCCESS&&page>0&&(uint64_t(vm.free_count)+vm.inactive_count)*uint64_t(page)>=(uint64_t(2)<<30)+bytes;
}
inline bool dense_name(const std::string & name) {
    if(name.rfind("blk.",0)!=0)return false;
    size_t dot=name.find('.',4);if(dot==std::string::npos||dot==4)return false;
    for(size_t i=4;i<dot;++i)if(name[i]<'0'||name[i]>'9')return false;
    const char * families[]={"hc_attn_fn.weight","hc_ffn_fn.weight","ffn_gate_inp.weight",
        "ssm_beta.weight","ssm_f_a.weight","ssm_f_b.weight","ssm_g_a.weight","ssm_g_b.weight",
        "attn_k_b.weight","attn_v_b.weight","attn_kv_a_mqa.weight","indexer.attn_q_b.weight",
        "indexer.attn_k.weight","indexer.proj.weight","indexer_compressor_gate.weight"};
    const auto family=name.substr(dot+1);
    return std::any_of(std::begin(families),std::end(families),[&](const char * p){return family==p;});
}
inline bool tensor_range(const ggml_tensor * tensor,const std::vector<range> & mappings,size_t page,range & out) {
    if(!tensor||!tensor->data||!tensor->buffer||!ggml_is_contiguous(tensor)||ggml_n_dims(tensor)<2||ggml_n_dims(tensor)>3||
       (tensor->type!=GGML_TYPE_F32&&tensor->type!=GGML_TYPE_BF16)||
       std::strcmp(ggml_backend_buffer_name(tensor->buffer),"CPU_Mapped")||
       ggml_backend_buffer_get_usage(tensor->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS)return false;
    const ggml_tensor * leaf=tensor;
    for(int i=0;i<16;++i){
        if(leaf->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT|GGML_TENSOR_FLAG_COMPUTE))return false;
        if(leaf->op==GGML_OP_NONE)break;
        if(!leaf->view_src||(leaf->op!=GGML_OP_VIEW&&leaf->op!=GGML_OP_RESHAPE&&leaf->op!=GGML_OP_PERMUTE&&leaf->op!=GGML_OP_TRANSPOSE))return false;
        leaf=leaf->view_src;
    }
    if(leaf->op!=GGML_OP_NONE||(leaf->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT|GGML_TENSOR_FLAG_COMPUTE)))return false;
    const auto data=reinterpret_cast<uintptr_t>(tensor->data),base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
    const size_t size=ggml_backend_buffer_get_size(tensor->buffer),bytes=ggml_nbytes(tensor);
    if(!base||data<base||data-base>size||!bytes||bytes>(size_t(512)<<20)||bytes>size-(data-base)||data>UINTPTR_MAX-bytes)return false;
    const uintptr_t end=data+bytes;const size_t pad=(page-end%page)%page;
    if(end>UINTPTR_MAX-pad)return false;
    out={data-data%page,end+pad};
    return std::any_of(mappings.begin(),mappings.end(),[&](range r){return r.begin<=out.begin&&out.end<=r.end;})&&readonly_unwired(out.begin,out.end);
}

class owner {
    std::vector<range> locked_;
    hooks hooks_;
    bool reserved_=false;
    outcome result_;
    void release_reservation(){if(reserved_){std::lock_guard<std::mutex> lock(mutex());if(active()==this)active()=nullptr;reserved_=false;}}
    bool unlock_all() noexcept {
        size_t keep=0;
        for(auto r:locked_){
            ++result_.unlock_calls;
            int rc=-1;
            errno=0;
            try {rc=hooks_.unlock?hooks_.unlock(reinterpret_cast<void *>(r.begin),r.end-r.begin,hooks_.data):munlock(reinterpret_cast<void *>(r.begin),r.end-r.begin);}
            catch(...){errno=EIO;}
            if(rc){result_.unlock_error=errno?errno:EIO;locked_[keep++]=r;}
        }
        locked_.resize(keep);result_.locked_bytes=0;
        for(auto r:locked_)result_.locked_bytes+=r.end-r.begin;
        result_.retained=false;if(locked_.empty())release_reservation();
        return locked_.empty();
    }
    bool decline(const char * status,int error=0){result_.status=status;result_.error=error;unlock_all();return false;}
public:
    owner()=default;owner(const owner &)=delete;owner & operator=(const owner &)=delete;
    const outcome & result() const{return result_;}
    bool acquire(const std::vector<std::pair<std::string,ggml_tensor *>> & tensors,const std::vector<range> & mappings,hooks injected={}) {
        if(!enabled())return false;
        if(reserved_||!locked_.empty())return false;
        hooks_=injected;result_={};
        try {
        const char * budget=std::getenv("AIRLLM_DENSE_LOCK_MIB");size_t mib=256;
        if(budget){if(!std::strcmp(budget,"128"))mib=128;else if(!std::strcmp(budget,"256"))mib=256;else if(!std::strcmp(budget,"512"))mib=512;else return decline("invalid_budget");}
        const char * names=std::getenv("AIRLLM_DENSE_LOCK_NAMES");
        if(!names||!names[0]||strnlen(names,16385)>16384)return decline("invalid_names");
        std::set<std::string> wanted;std::string text(names);
        for(size_t begin=0;begin<=text.size();){
            const size_t comma=text.find(',',begin);const auto name=text.substr(begin,comma==std::string::npos?std::string::npos:comma-begin);
            if(name.empty()||!dense_name(name)||!wanted.insert(name).second||wanted.size()>128)return decline("invalid_names");
            if(comma==std::string::npos)break;begin=comma+1;
        }
        const long page_value=sysconf(_SC_PAGESIZE);if(page_value<=0)return decline("unknown_page_size");const size_t page=size_t(page_value);
        std::vector<range> spans;spans.reserve(wanted.size());
        for(const auto & name:wanted){
            auto found=std::find_if(tensors.begin(),tensors.end(),[&](const auto & row){return row.first==name;});
            range r;
            if(found==tensors.end()||!tensor_range(found->second,mappings,page,r))return decline("invalid_original_weight");
            spans.push_back(r);result_.raw_bytes+=ggml_nbytes(found->second);
        }
        std::sort(spans.begin(),spans.end(),[](range a,range b){return a.begin<b.begin;});
        std::vector<range> merged;for(auto r:spans){if(!merged.empty()&&r.begin<=merged.back().end)merged.back().end=std::max(merged.back().end,r.end);else merged.push_back(r);}
        for(auto r:merged){if(r.end-r.begin>(mib<<20)-result_.union_bytes)return decline("page_union_over_budget");result_.union_bytes+=r.end-r.begin;}
        result_.names=wanted.size();
        struct rlimit limit{};if(getrlimit(RLIMIT_MEMLOCK,&limit))return decline("rlimit_unavailable",errno);
        if(limit.rlim_cur!=RLIM_INFINITY&&result_.union_bytes>limit.rlim_cur)return decline("rlimit_insufficient");
        // All allocations precede the first VM lock; rollback/destruction allocate nothing.
        locked_.reserve(merged.size());size_t largest=0;
        for(auto r:merged)largest=std::max(largest,(r.end-r.begin)/page);
        std::vector<unsigned char> resident(largest);
        {std::unique_lock<std::mutex> lock(mutex(),std::try_to_lock);
         if(!lock.owns_lock())return decline("concurrent_owner");
         if(poisoned())return decline("prior_unlock_incomplete");
         if(active())return decline("concurrent_owner");active()=this;reserved_=true;}
        size_t remaining=result_.union_bytes;
        for(auto r:merged){
            if(hooks_.cancel&&hooks_.cancel(hooks_.data)){result_.cancelled=true;return decline("cancelled");}
            if(!(hooks_.admit?hooks_.admit(remaining,hooks_.data):admissible(remaining)))return decline("admission_declined");
            // Revalidate after reservation, immediately before touching the VM lock.
            if(!readonly_unwired(r.begin,r.end))return decline("mapping_changed_or_prelocked");
            ++result_.lock_calls;errno=0;
            locked_.push_back(r);
            const int rc=hooks_.lock?hooks_.lock(reinterpret_cast<void *>(r.begin),r.end-r.begin,hooks_.data):mlock(reinterpret_cast<void *>(r.begin),r.end-r.begin);
            // Include an attempted failed range in rollback: partial kernel failure
            // must not leave a silent wired reservation behind.
            if(rc)return decline("mlock_failed",errno?errno:EIO);
            if(mincore(reinterpret_cast<void *>(r.begin),r.end-r.begin,reinterpret_cast<char *>(resident.data())))return decline("residency_unknown",errno);
            const size_t count=(r.end-r.begin)/page;
            const auto pages=std::count_if(resident.begin(),resident.begin()+count,[](unsigned char value){return value&MINCORE_INCORE;});
            result_.resident_pages+=size_t(pages);
            if(size_t(pages)!=count)return decline("residency_incomplete");
            result_.locked_bytes+=r.end-r.begin;remaining-=r.end-r.begin;
        }
        if(hooks_.cancel&&hooks_.cancel(hooks_.data)){result_.cancelled=true;return decline("cancelled");}
        if(!(hooks_.admit?hooks_.admit(0,hooks_.data):admissible(0)))return decline("postlock_admission_declined");
        result_.retained=true;result_.status="retained";return true;
        }catch(...){return decline("exception",ENOMEM);}
    }
    void release() noexcept {const bool was=result_.retained;const bool released=unlock_all();if(was)result_.status=released?"released":"release_incomplete";}
    void log() const {
        std::fprintf(stderr,"AIRLLM_DENSE_LOCK status=%s retained=%d names=%zu raw_bytes=%zu page_union_bytes=%zu locked_bytes=%zu lock_calls=%zu unlock_calls=%zu verified_pages_at_lock=%zu error=%d unlock_error=%d\n",
            result_.status,int(result_.retained),result_.names,result_.raw_bytes,result_.union_bytes,result_.locked_bytes,
            result_.lock_calls,result_.unlock_calls,result_.resident_pages,result_.error,result_.unlock_error);
    }
    ~owner(){
        release();
        if(reserved_){
            // Mapping destruction follows this member, but its completion cannot
            // be proved here. Never admit another budget after failed cleanup.
            std::lock_guard<std::mutex> lock(mutex());poisoned()=true;
            if(active()==this)active()=nullptr;reserved_=false;
            std::fprintf(stderr,"AIRLLM_DENSE_LOCK status=admission_poisoned_after_unlock_failure retained=0 uncertain_locked_bytes=%zu\n",result_.locked_bytes);
        }
    }
};
} // namespace airllm_dense_lock
