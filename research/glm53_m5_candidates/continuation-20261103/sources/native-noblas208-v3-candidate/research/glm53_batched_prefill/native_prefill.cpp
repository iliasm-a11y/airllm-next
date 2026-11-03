// ISOLATED UNQUALIFIED v3: same fences/tails; unregister BLAS before model creation.
// Research diagnostic: real multi-token prefill and full sequence snapshots.
// No production routing or arithmetic changes. The Python owner enforces guards.
#include "llama.h"
#include "ggml-backend.h"
#include "snapshot.h"
#include <nlohmann/json.hpp>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libproc.h>
#include <map>
#include <mach/mach_time.h>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
using steady = std::chrono::steady_clock;
static const char * const prefill_chunk_policy = "cpu8-tail-global-fences208-no-blas-v3";
static const std::vector<int> prefill_fences = {128,208,256,384,512};
static const char * const cpu_accelerator_policy = "unregister-BLAS-before-model-v1";
static int prefill_fence_end(int offset, int end) {
    int stop=end;
    for(const int fence:prefill_fences)if(fence>offset && fence<stop)stop=fence;
    return stop;
}
// Fences are absolute native-token positions, independent of full/suffix mode.
// Their multiples-of8 boundaries preserve CPU8's odd1/3/5/7 remainder.
// This proves chunk parity eligibility only, not native numerical equivalence.
static int prefill_take(int ub, int offset, int end) {
    if ((ub != 8 && ub != 16 && ub != 32 && ub != 64 && ub != 128) || offset<0 || end<=offset)
        throw std::runtime_error("candidate requires an admitted positive microbatch");
    const int take = std::min(ub, prefill_fence_end(offset,end)-offset);
    return take > 8 && take % 2 != 0 ? take - take % 8 : take;
}
#include "../glm53_gpu_v3/state_capture.inc"
static double seconds(steady::time_point a) { return std::chrono::duration<double>(steady::now()-a).count(); }
static std::string hash_bytes(const void * data, size_t bytes) {
    CC_SHA256_CTX ctx; CC_SHA256_Init(&ctx);
    const auto * p=static_cast<const unsigned char *>(data);
    for(size_t offset=0;offset<bytes;offset+=1024*1024) CC_SHA256_Update(&ctx,p+offset,CC_LONG(std::min<size_t>(1024*1024,bytes-offset)));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest,&ctx);
    std::ostringstream out; for(auto v:digest) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v); return out.str();
}
static json usage() {
    rusage_info_v2 u{}; mach_timebase_info_data_t t{}; mach_timebase_info(&t);
    if(proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t *>(&u))) throw std::runtime_error("self usage unavailable");
    struct rusage peak{};
    if(getrusage(RUSAGE_SELF,&peak))throw std::runtime_error("lifetime RSS peak unavailable");
    if(uint64_t(peak.ru_maxrss)>uint64_t(8.5*1024*1024*1024))throw std::runtime_error("lifetime RSS peak exceeded8.5GiB");
    return {{"rss_bytes",u.ri_resident_size},{"footprint_bytes",u.ri_phys_footprint},
        {"lifetime_peak_rss_bytes",peak.ru_maxrss},
        {"cpu_seconds",(u.ri_user_time+u.ri_system_time)*double(t.numer)/t.denom/1e9},
        {"os_storage_read_bytes",u.ri_diskio_bytesread},{"pageins",u.ri_pageins}};
}
struct observer {
    std::ofstream operations, captures, batches, states, checkpoints, continuation_captures, continuation_states;
    std::filesystem::path root;
    bool diagnostic=false;
    size_t serial=0;
    int request=-1, offset=0;
    int input_tokens=0, batch_index=0;
    std::map<int,int> next_batch;
    std::string phase="load";
};
static bool evaluate(ggml_tensor * node,bool ask,void * user) {
    auto & o=*static_cast<observer *>(user);
    const std::string name=ggml_get_name(node);
    const bool matrix=node->op==GGML_OP_MUL_MAT || node->op==GGML_OP_MUL_MAT_ID;
    const bool continuation=o.phase=="continuation";
    const bool selected=name.rfind("ffn_moe_topk-",0)==0 || name.rfind("ffn_moe_logits-",0)==0 || name.rfind("kda_scan_out-",0)==0 || (continuation && name=="result_output");
    auto & captures=continuation?o.continuation_captures:o.captures;
    auto & states=continuation?o.continuation_states:o.states;
    const bool write=o.diagnostic && state_target(node);
    if(ask && matrix) {
        json row={{"request",o.request},{"offset",o.offset},{"phase",o.phase},{"name",name},{"op",ggml_op_name(node->op)},
            {"shape",{node->ne[0],node->ne[1],node->ne[2],node->ne[3]}}};
        if(node->src[1]) row["activation_shape"]={node->src[1]->ne[0],node->src[1]->ne[1],node->src[1]->ne[2],node->src[1]->ne[3]};
        if(node->src[0]) {row["weight_type"]=ggml_type_name(node->src[0]->type);row["weight_name"]=ggml_get_name(node->src[0]);}
        o.operations<<row.dump()<<'\n';
    }
    if(ask) return o.diagnostic && (selected || write);
    if(write) {
        const size_t rss=usage().at("rss_bytes"), cap=size_t(8.5*1024*1024*1024);
        if(rss>cap || 1024*1024>cap-rss)throw std::runtime_error("state hash scratch RSS admission refused");
        auto event=json::parse(state_event(node,o.offset));
        event["request"]=o.request;event["offset"]=o.offset;event["phase"]=o.phase;
        event["input_tokens"]=o.input_tokens;event["batch_index"]=o.batch_index;
        states<<event.dump()<<'\n';states.flush();
    }
    if(!selected || !o.diagnostic) return true;
    const size_t bytes=ggml_nbytes(node);
    if(bytes>64*1024*1024) throw std::runtime_error("diagnostic tensor exceeds64MiB");
    const size_t rss=usage().at("rss_bytes"), cap=size_t(8.5*1024*1024*1024);
    if(rss>cap || bytes>cap-rss)throw std::runtime_error("diagnostic tensor RSS admission refused");
    std::vector<unsigned char> data(bytes);ggml_backend_tensor_get(node,data.data(),0,bytes);
    const auto file="tensor-"+std::to_string(o.serial++)+".bin";
    std::ofstream binary(o.root/file,std::ios::binary);binary.exceptions(std::ios::badbit|std::ios::failbit);
    binary.write(reinterpret_cast<const char *>(data.data()),bytes);
    captures<<json({{"request",o.request},{"offset",o.offset},{"input_tokens",o.input_tokens},{"batch_index",o.batch_index},
        {"phase",o.phase},{"name",name},{"type",int(node->type)},
        {"shape",{node->ne[0],node->ne[1],node->ne[2],node->ne[3]}},{"strides",{node->nb[0],node->nb[1],node->nb[2],node->nb[3]}},
        {"bytes",bytes},{"file",file},{"sha256",hash_bytes(data.data(),bytes)}}).dump()<<'\n';
    captures.flush(); return true;
}
static void save(const std::filesystem::path & p,const json & j) {
    std::ofstream f(p); f.exceptions(std::ios::badbit|std::ios::failbit); f<<j.dump(2)<<'\n';
}
static llama_context_params context_parameters(int ub) {
    auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=cp.n_ubatch=ub;cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=10;cp.op_offload=false;cp.offload_kqv=false;cp.moe_cache_size=0;
    return cp;
}
static json parameter_description(int ub) {
    const auto cp=context_parameters(ub);
    return {{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_seq_max",cp.n_seq_max},
        {"n_rs_seq",cp.n_rs_seq},{"n_outputs_max",cp.n_outputs_max},{"n_outputs_max_per_seq",cp.n_outputs_max_per_seq},
        {"n_threads",cp.n_threads},{"n_threads_batch",cp.n_threads_batch},{"type_k",int(cp.type_k)},{"type_v",int(cp.type_v)},
        {"flash_attn",int(cp.flash_attn_type)},{"offload_kqv",cp.offload_kqv},{"op_offload",cp.op_offload},{"n_gpu_layers",0},
        {"rope_scaling_type",int(cp.rope_scaling_type)},{"rope_freq_base",cp.rope_freq_base},{"rope_freq_scale",cp.rope_freq_scale},
        {"yarn_ext_factor",cp.yarn_ext_factor},{"yarn_attn_factor",cp.yarn_attn_factor},{"yarn_beta_fast",cp.yarn_beta_fast},
        {"yarn_beta_slow",cp.yarn_beta_slow},{"yarn_orig_ctx",cp.yarn_orig_ctx},{"swa_full",cp.swa_full},{"kv_unified",cp.kv_unified},
        {"ctx_type",int(cp.ctx_type)},{"pooling_type",int(cp.pooling_type)},{"attention_type",int(cp.attention_type)},
        {"embeddings",cp.embeddings},{"moe_cache_size",cp.moe_cache_size},{"n_samplers",cp.n_samplers},
        {"prefill_chunk_policy",prefill_chunk_policy},{"prefill_chunk_policy_version",3},{"tail_reference_microbatch",8},
        {"prefill_fences",prefill_fences},{"cpu_accelerator_policy",cpu_accelerator_policy}};
}

static json enforce_backend_policy() {
    json before=json::array(),after=json::array();
    for(size_t i=0;i<ggml_backend_reg_count();++i)before.push_back(ggml_backend_reg_name(ggml_backend_reg_get(i)));
    auto blas=ggml_backend_reg_by_name("BLAS");
    if(!blas)throw std::runtime_error("expected original BLAS registry is absent before explicit candidate policy");
    ggml_backend_unload(blas);
    if(ggml_backend_reg_by_name("BLAS"))throw std::runtime_error("BLAS removal failed before model creation");
    for(size_t i=0;i<ggml_backend_dev_count();++i)
        if(ggml_backend_dev_type(ggml_backend_dev_get(i))==GGML_BACKEND_DEVICE_TYPE_ACCEL)
            throw std::runtime_error("unexpected accelerator remains after BLAS removal");
    for(size_t i=0;i<ggml_backend_reg_count();++i)after.push_back(ggml_backend_reg_name(ggml_backend_reg_get(i)));
    if(!ggml_backend_reg_by_name("CPU"))throw std::runtime_error("ordinary CPU registry is absent");
    return {{"policy",cpu_accelerator_policy},{"registry_before",before},{"registry_after",after},
        {"BLAS_registered_after",false},{"accelerator_devices_after",0},{"CPU_registered_after",true},
        {"model_loaded",false},{"bridge_matrix_dispatches",0},
        {"scope","registration only; legacy backend initialization may use the device"}};
}

int main(int argc,char ** argv) {
    llama_model * model=nullptr; llama_context * ctx=nullptr;
    try {
        if(argc==2 && std::string(argv[1])=="--backend-policy-selftest") {
            ggml_backend_load_all();std::cout<<enforce_backend_policy().dump()<<'\n';return 0;
        }
        if(argc==3 && std::string(argv[1])=="--state-parameters") {
            const int ub=std::stoi(argv[2]);
            if(ub!=8 && ub!=16 && ub!=32 && ub!=64 && ub!=128)throw std::runtime_error("invalid microbatch");
            std::cout<<parameter_description(ub).dump()<<'\n';return 0;
        }
        if(argc!=6) throw std::runtime_error("usage: native-prefill FIRST_GGUF PROMPTS_JSON OUTPUT_DIR MICROBATCH full|split|restore");
        const int ub=std::stoi(argv[4]); const std::string mode=argv[5];
        if((ub!=8 && ub!=16 && ub!=32 && ub!=64 && ub!=128) || (mode!="full" && mode!="split" && mode!="restore")) throw std::runtime_error("invalid bounded profile");
        const std::filesystem::path root=argv[3];if(std::filesystem::exists(root))throw std::runtime_error("output exists");
        std::ifstream input(argv[2]);if(!input || std::filesystem::file_size(argv[2])>16*1024*1024)throw std::runtime_error("invalid prompt file");
        const auto prompts=json::parse(input);
        std::vector<std::vector<llama_token>> ids;
        for(int agent:{1,600,1}) {
            const auto it=std::find_if(prompts.at("agents").begin(),prompts.at("agents").end(),[&](const auto & a){return a.at("agent_id")==agent;});
            if(it==prompts.at("agents").end())throw std::runtime_error("missing agent");
            ids.push_back(it->at("token_ids").template get<std::vector<llama_token>>());
            if(ids.back().size()<=208 || ids.back().size()>1800)throw std::runtime_error("invalid full prompt size");
        }
        if(!std::equal(ids[0].begin(),ids[0].begin()+208,ids[1].begin()))throw std::runtime_error("prefix differs");
        std::filesystem::create_directories(root);
        observer o; o.root=root;o.diagnostic=std::getenv("RONIN_PREFILL_DIAGNOSTIC")!=nullptr;
        o.operations.open(root/"operations.jsonl");o.captures.open(root/"tensors.jsonl");o.batches.open(root/"batches.jsonl");o.states.open(root/"state_hashes.jsonl");o.checkpoints.open(root/"state_checkpoints.jsonl");
        o.continuation_captures.open(root/"continuation_tensors.jsonl");o.continuation_states.open(root/"continuation_states.jsonl");
        for(auto * stream:{&o.operations,&o.captures,&o.batches,&o.states,&o.checkpoints,&o.continuation_captures,&o.continuation_states})stream->exceptions(std::ios::badbit|std::ios::failbit);
        json report={{"schema_version",1},{"mode",mode},{"microbatch",ub},{"logical_batch",ub},{"threads",10},
            {"prefill_chunk_policy",prefill_chunk_policy},{"prefill_chunk_policy_version",3},{"prefill_fences",prefill_fences},
            {"prefill_chunk_policy_qualified",false},{"tail_reference_microbatch",8},
            {"gpu_layers",0},{"physical_slots",1},{"model_loads",0},{"diagnostic",o.diagnostic},{"complete",false},
            {"generation_requested",o.diagnostic},{"generation_performed",false},{"functional_agents_qualified",false},
            {"continuation_policy","fresh native greedy sampler; max4; diagnostic only"},
            {"requests",json::array()},{"state_parameters",parameter_description(ub)},
            {"output_policy","logits only for the final token of each complete private prompt; no unused vocabulary products per microbatch"},
            {"scope","prefill/snapshot and bounded greedy continuity diagnostic; no functional agent EOS qualification"}};
        save(root/"report.json",report);
        ggml_backend_load_all();
        report["backend_policy_observation"]=enforce_backend_policy();save(root/"report.json",report);
        llama_backend_init(); auto mp=llama_model_default_params();
        mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.use_extra_bufts=false;
        ggml_backend_dev_t devices[]={nullptr};mp.devices=devices;
        const auto load_start=steady::now();model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load failed");
        report["model_loads"]=1;report["model_load_seconds"]=seconds(load_start);
        auto cp=context_parameters(ub);
        cp.cb_eval=evaluate;cp.cb_eval_user_data=&o;ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context load failed");
        const auto * vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
        report["vocab_size"]=nv;
        for(const auto & tokens:ids)for(auto token:tokens)if(token<0 || token>=nv)throw std::runtime_error("invalid prompt token");
        const std::string key=prompts.at("snapshot_key_sha256");
        report["snapshot_key_sha256"]=key;
        glm53_prefill::prefix_snapshot snapshot;
        auto snapshot_metrics=[&]() {
            const auto t=snapshot.metrics();
            return json({{"bytes",t.bytes},{"reserved_bytes",t.reserved_bytes},{"size_seconds",t.size_seconds},
                {"save_seconds",t.save_seconds},{"restore_seconds_total",t.restore_seconds},{"successful_restores",t.successful_restores},
                {"sha256",snapshot.ready()?json(snapshot.checksum_sha256()):json(nullptr)}});
        };
        // Caller supplied conservative native read/write scratch bound, audited
        // separately; zero or unavailable means restore mode cannot be admitted.
        const size_t scratch=prompts.value("native_snapshot_scratch_bound_bytes",size_t(0));
        bool snapshot_resident=false;
        const auto budget=[&](size_t bytes,const char * phase) {
            const auto u=usage();const size_t rss=u.at("rss_bytes");
            // Snapshot payload is not counted twice once resident. The owner
            // also watches available reserve, pressure, swap and PID identity.
            if(std::string(phase)=="after_snapshot_write")snapshot_resident=true;
            const size_t additional=snapshot_resident?scratch:bytes;
            if(rss>size_t(8.5*1024*1024*1024) || additional>size_t(8.5*1024*1024*1024)-rss)throw std::runtime_error("snapshot RSS admission refused");
            o.batches<<json({{"event",phase},{"reserved_bytes",bytes},{"self",u}}).dump()<<'\n';o.batches.flush();
        };
        auto decode=[&](int request,int first,int end,const char * phase) {
            o.request=request;o.phase=phase;
            for(int offset=first;offset<end;) {
                o.offset=offset;const int unfenced_take=std::min(ub,end-offset);
                const int fence_end=prefill_fence_end(offset,end);
                const int unadjusted_take=std::min(ub,fence_end-offset);
                const int take=prefill_take(ub,offset,end);auto start=steady::now();const auto before=usage();
                o.input_tokens=take;o.batch_index=o.next_batch[request]++;
                auto batch=llama_batch_get_one(ids[request].data()+offset,take);
                std::vector<int8_t> output_flags(take,0);
                if(offset+take==int(ids[request].size()))output_flags.back()=1;
                batch.logits=output_flags.data();
                const int rc=llama_decode(ctx,batch);llama_synchronize(ctx);
                o.batches<<json({{"request",request},{"phase",phase},{"offset",offset},{"tokens",take},
                    {"unfenced_take",unfenced_take},{"fence_end",fence_end},{"decode_end",end},
                    {"fence_limited",unadjusted_take!=unfenced_take},{"unadjusted_take",unadjusted_take},
                    {"odd_tail_split",take!=unadjusted_take},{"prefill_chunk_policy",prefill_chunk_policy},
                    {"seconds",seconds(start)},{"returncode",rc},{"before",before},{"after",usage()}}).dump()<<'\n';o.batches.flush();
                if(rc)throw std::runtime_error("native decode failed");offset+=take;
                if(o.diagnostic && (offset%128==0 || offset==208 || offset==int(ids[request].size()))) {
                    const auto checkpoint_start=steady::now();
                    const size_t n=llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_NONE);
                    const size_t rss=usage().at("rss_bytes"),cap=size_t(8.5*1024*1024*1024);
                    if(!n || n>256*1024*1024 || rss>cap || scratch>cap || n>cap-scratch || n+scratch>cap-rss)throw std::runtime_error("checkpoint admission refused");
                    std::vector<uint8_t> data(n);
                    if(llama_state_seq_get_data_ext(ctx,data.data(),n,0,LLAMA_STATE_SEQ_FLAGS_NONE)!=n)throw std::runtime_error("short state checkpoint");
                    o.checkpoints<<json({{"request",request},{"token_end",offset},{"bytes",n},{"sha256",hash_bytes(data.data(),n)},
                        {"state_seq_flags",0},{"status","hashed"},{"next_position",llama_memory_seq_pos_max(llama_get_memory(ctx),0)+1},
                        {"seconds",seconds(checkpoint_start)},{"self",usage()}}).dump()<<'\n';o.checkpoints.flush();
                }
            }
        };
        if(mode=="restore") {
            auto start=steady::now();decode(0,0,208,"common_prefix");report["common_prefix_seconds"]=seconds(start);
            save(root/"report.json",report);
            start=steady::now();
            snapshot.capture(ctx,key,208,scratch,budget);
            report["snapshot_capture_total_seconds"]=seconds(start);
            report["snapshot"]=snapshot_metrics();save(root/"report.json",report);
        }
        for(int r=0;r<3;++r) {
            const auto start=steady::now();const auto before=usage();size_t restored=0;
            const auto restore_start=steady::now();
            if(mode=="restore") restored=snapshot.restore(ctx,key,ids[r].size()-208,budget);
            else llama_memory_clear(llama_get_memory(ctx),true);
            const double reset_restore_seconds=seconds(restore_start);
            if(mode=="restore") {report["snapshot"]=snapshot_metrics();save(root/"report.json",report);}
            if(mode=="split") {decode(r,0,208,"prefix");decode(r,208,ids[r].size(),"suffix");}
            else decode(r,restored,ids[r].size(),mode=="restore"?"suffix":"full");
            const float * logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("missing continuation logits");
            if(!std::all_of(logits,logits+nv,[](float x){return std::isfinite(x);}))throw std::runtime_error("nonfinite logits");
            std::ofstream binary(root/("logits-"+std::to_string(r)+".f32"),std::ios::binary);
            binary.exceptions(std::ios::badbit|std::ios::failbit);binary.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));binary.close();
            const size_t state_bytes=llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_NONE);
            json row={{"request",r},{"agent_id",r==1?600:1},{"prompt_tokens",ids[r].size()},{"computed_tokens",ids[r].size()-restored},
                {"prefill_with_reset_restore_seconds",seconds(start)},{"before",before},{"after",usage()},{"restored_prefix_tokens",restored},
                {"reset_restore_total_seconds",reset_restore_seconds},
                {"logits_sha256",hash_bytes(logits,nv*sizeof(float))},{"sequence_bytes",state_bytes},{"next_position",llama_memory_seq_pos_max(llama_get_memory(ctx),0)+1}};
            row["prompt_ids_sha256"]=hash_bytes(ids[r].data(),ids[r].size()*sizeof(llama_token));
            row["n_vocab"]=nv;
            if(o.diagnostic) {
                // A second full-state buffer is explicitly charged. Diagnostic
                // payload is freed before the next request, never per identity.
                if(state_bytes>256*1024*1024)throw std::runtime_error("final-state diagnostic exceeds256MiB");
                const size_t rss=usage().at("rss_bytes");
                const size_t cap=size_t(8.5*1024*1024*1024);
                if(rss>cap || scratch>cap || state_bytes>cap-scratch || state_bytes+scratch>cap-rss)throw std::runtime_error("final-state diagnostic RSS admission refused");
                std::vector<uint8_t> state(state_bytes);
                if(llama_state_seq_get_data_ext(ctx,state.data(),state.size(),0,LLAMA_STATE_SEQ_FLAGS_NONE)!=state_bytes)throw std::runtime_error("short final state");
                row["sequence_sha256"]=hash_bytes(state.data(),state.size());
                std::ofstream state_file(root/("final-state-"+std::to_string(r)+".bin"),std::ios::binary);
                state_file.exceptions(std::ios::badbit|std::ios::failbit);
                state_file.write(reinterpret_cast<const char *>(state.data()),state.size());
            }
            row["continuation_complete"]=false;row["continuation"]=json::array();
            report["requests"].push_back(row);save(root/"report.json",report);
            if(o.diagnostic) {
                const auto continuation_start=steady::now();
                std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
                if(!sampler)throw std::runtime_error("fresh native sampler unavailable");
                llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
                for(int step=0;step<4;++step) {
                    const float * next_logits=llama_get_logits_ith(ctx,-1);
                    if(!next_logits || !std::all_of(next_logits,next_logits+nv,[](float x){return std::isfinite(x);}))throw std::runtime_error("continuation logits unavailable or nonfinite");
                    const size_t rss=usage().at("rss_bytes"),cap=size_t(8.5*1024*1024*1024);
                    if(rss>cap || size_t(nv)*sizeof(llama_token_data)>cap-rss)throw std::runtime_error("sampler scratch RSS admission refused");
                    const auto token=llama_sampler_sample(sampler.get(),ctx,-1);
                    if(token<0 || token>=nv)throw std::runtime_error("invalid native greedy token");
                    const bool eog=llama_vocab_is_eog(vocab,token);
                    std::ofstream vector_file(root/("continuation-logits-"+std::to_string(r)+"-"+std::to_string(step)+".f32"),std::ios::binary);
                    vector_file.exceptions(std::ios::badbit|std::ios::failbit);vector_file.write(reinterpret_cast<const char *>(next_logits),nv*sizeof(float));vector_file.close();
                    row["continuation"].push_back({{"step",step},{"token",token},{"native_eog",eog},{"logits_sha256",hash_bytes(next_logits,nv*sizeof(float))}});
                    report["generation_performed"]=true;
                    row["continuation_stop"]=eog?"eos":"diagnostic_budget";
                    report["requests"][r]=row;save(root/"report.json",report);
                    if(eog || step==3)break;
                    auto input_token=token;auto one=llama_batch_get_one(&input_token,1);
                    o.request=r;o.offset=step+1;o.batch_index=step+1;o.input_tokens=1;o.phase="continuation";
                    if(llama_decode(ctx,one))throw std::runtime_error("continuation decode failed");llama_synchronize(ctx);
                }
                row["continuation_seconds"]=seconds(continuation_start);
                row["continuation_complete"]=true;
                const size_t n=llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_NONE);
                const size_t rss=usage().at("rss_bytes"),cap=size_t(8.5*1024*1024*1024);
                if(!n || n>256*1024*1024 || rss>cap || scratch>cap || n>cap-scratch || n+scratch>cap-rss)throw std::runtime_error("continuation-state admission refused");
                std::vector<uint8_t> state(n);
                if(llama_state_seq_get_data_ext(ctx,state.data(),n,0,LLAMA_STATE_SEQ_FLAGS_NONE)!=n)throw std::runtime_error("short continuation state");
                row["continuation_state_bytes"]=n;row["continuation_state_sha256"]=hash_bytes(state.data(),n);
                row["continuation_next_position"]=llama_memory_seq_pos_max(llama_get_memory(ctx),0)+1;
                std::ofstream state_file(root/("continuation-state-"+std::to_string(r)+".bin"),std::ios::binary);
                state_file.exceptions(std::ios::badbit|std::ios::failbit);state_file.write(reinterpret_cast<const char *>(state.data()),n);
                report["requests"][r]=row;save(root/"report.json",report);
            }
        }
        report["snapshot"]=snapshot_metrics();
        report["complete"]=true;save(root/"report.json",report);
        llama_free(ctx);ctx=nullptr;llama_model_free(model);model=nullptr;llama_backend_free();return 0;
    } catch(const std::exception & e) {
        std::cerr<<e.what()<<'\n';if(ctx)llama_free(ctx);if(model)llama_model_free(model);return 2;
    }
}
