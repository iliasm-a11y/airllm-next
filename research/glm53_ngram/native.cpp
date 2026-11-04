// Research only. Original llama.cpp n-gram proposer, one CPU target context.
// Audit work is deliberately separate from timed generation.
#include "llama.h"
#include "ggml-backend.h"
#include "ngram-map.h"
#include <nlohmann/json.hpp>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libproc.h>
#include <mach/mach_time.h>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

using json=nlohmann::json;
using clock_type=std::chrono::steady_clock;
constexpr size_t MIB=1ull<<20, RSS_CAP=17ull<<29, STATE_CAP=256*MIB, TRACE_CAP=64*MIB;
static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now()-start).count();
}
static size_t rss() {
    rusage_info_v2 u{}; struct rusage peak{};
    if(proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t*>(&u)) ||
       getrusage(RUSAGE_SELF,&peak)) throw std::runtime_error("self memory counters unavailable");
    if(uint64_t(peak.ru_maxrss)>RSS_CAP || u.ri_resident_size>RSS_CAP)
        throw std::runtime_error("native lifetime RSS exceeded 8.5GiB");
    return u.ri_resident_size;
}
static void admit(size_t bytes) {
    const auto resident=rss();
    if(bytes>RSS_CAP-resident)throw std::runtime_error("additional native allocation exceeds RSS budget");
}
static std::string hash(const void * data,size_t bytes) {
    CC_SHA256_CTX h;CC_SHA256_Init(&h);
    const auto * p=static_cast<const unsigned char*>(data);
    for(size_t offset=0;offset<bytes;offset+=MIB)
        CC_SHA256_Update(&h,p+offset,CC_LONG(std::min(MIB,bytes-offset)));
    unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(d,&h);
    std::ostringstream s;for(auto b:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);
    return s.str();
}
static void save(const std::filesystem::path & p,const json & value) {
    std::ofstream f(p);f.exceptions(std::ios::badbit|std::ios::failbit);f<<value.dump(2)<<'\n';
}
#include "../glm53_gpu_v3/state_capture.inc"
#include "../glm53_ultra_decode/timing.inc"

static std::vector<uint8_t> snapshot(llama_context * ctx) {
    llama_synchronize(ctx);const auto n=llama_state_get_size(ctx);
    if(!n || n>STATE_CAP)throw std::runtime_error("full snapshot exceeds 256MiB admission");
    admit(n);std::vector<uint8_t> out(n);
    if(llama_state_get_data(ctx,out.data(),out.size())!=n)throw std::runtime_error("short full snapshot");
    return out;
}
static std::string sequence_hash(llama_context * ctx) {
    llama_synchronize(ctx);const auto n=llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_NONE);
    if(!n || n>STATE_CAP)throw std::runtime_error("sequence state exceeds 256MiB admission");
    admit(n);std::vector<uint8_t> out(n);
    if(llama_state_seq_get_data_ext(ctx,out.data(),n,0,LLAMA_STATE_SEQ_FLAGS_NONE)!=n)
        throw std::runtime_error("short complete sequence snapshot");
    return hash(out.data(),n);
}
static void restore(llama_context * ctx,const std::vector<uint8_t>& data) {
    if(llama_state_set_data(ctx,data.data(),data.size())!=data.size())
        throw std::runtime_error("short full snapshot restore");
    llama_synchronize(ctx);
}
static int axis(const std::string & name) {
    for(const auto * prefix:{"ffn_moe_topk-","ffn_moe_logits-","kda_scan_out-"}) {
        const std::string p=prefix;
        if(name.rfind(p,0)==0 && name.size()>p.size() &&
           std::all_of(name.begin()+p.size(),name.end(),[](char c){return c>='0'&&c<='9';}))
            return p=="kda_scan_out-"?2:1;
    }
    return -1;
}
using trace_key=std::pair<int,std::string>;
static std::string canonical_target(std::string name) {
    // GGML names nested views with repeated " (view)" suffixes. Count the
    // underlying state bank, while preserving every raw event in the receipt.
    const auto suffix=name.find(" (view)");
    if(suffix!=std::string::npos)name.resize(suffix);
    return name;
}
static std::set<std::string> expected_targets() {
    std::set<std::string> names;
    for(int layer=0;layer<45;++layer) {
        const auto i=std::to_string(layer);
        if(layer%4!=3) {names.insert("cache_r_l"+i);names.insert("cache_s_l"+i);}
        else {names.insert("cache_k_l"+i);names.insert("cache_idx_k_l"+i);}
    }
    return names;
}
struct observer {
    bool enabled=false,reference=true,failed=false;
    int offset=0,tokens=1;
    size_t bytes=0;
    std::string phase;
    std::map<trace_key,std::vector<uint32_t>> values;
    std::set<trace_key> seen;
    std::set<std::string> writes;
    json comparisons=json::array();
    std::ofstream events;
};
static bool evaluate(ggml_tensor * node,bool ask,void * user) {
    auto & o=*static_cast<observer*>(user);
    if(!o.enabled)return ask?false:true;
    const std::string name=ggml_get_name(node);const int a=axis(name);
    const auto * target=state_target(node);
    if(ask)return a>=0 || target;
    if(target) {
        admit(MIB);auto event=json::parse(state_event(node,o.offset));
        event["phase"]=o.phase;event["input_tokens"]=o.tokens;
        if(event.value("status","")!="hashed" || !event.value("finite",false))o.failed=true;
        o.writes.insert(canonical_target(ggml_get_name(target)));o.events<<event.dump()<<'\n';o.events.flush();
    }
    if(a<0)return true;
    const size_t n=ggml_nbytes(node);
    if(n>TRACE_CAP || node->ne[a]!=o.tokens || node->ne[3]!=1 ||
       (node->type!=GGML_TYPE_F32 && node->type!=GGML_TYPE_I32))
        throw std::runtime_error("unsupported named GLM trace shape/type");
    admit(2*n);std::vector<uint8_t> raw(n);ggml_backend_tensor_get(node,raw.data(),0,n);
    for(int token=0;token<o.tokens;++token) {
        size_t count=1;for(int d=0;d<4;++d)if(d!=a)count*=size_t(node->ne[d]);
        if(count*4>TRACE_CAP-o.bytes)throw std::runtime_error("reference trace exceeds 64MiB bound");
        std::vector<uint32_t> logical;logical.reserve(count);
        for(int64_t i3=0;i3<node->ne[3];++i3)for(int64_t i2=0;i2<node->ne[2];++i2)
        for(int64_t i1=0;i1<node->ne[1];++i1)for(int64_t i0=0;i0<node->ne[0];++i0) {
            int64_t index[]={i0,i1,i2,i3};if(index[a]!=token)continue;
            size_t off=0;for(int d=0;d<4;++d)off+=size_t(index[d])*node->nb[d];
            if(off>n || 4>n-off)throw std::runtime_error("GLM trace stride out of bounds");
            uint32_t value;std::memcpy(&value,raw.data()+off,4);logical.push_back(value);
        }
        trace_key key{o.offset+token,name};
        if(!o.seen.insert(key).second)throw std::runtime_error("duplicate canonical GLM trace");
        if(o.reference) {o.bytes+=logical.size()*4;o.values.emplace(key,std::move(logical));}
        else {
            const auto found=o.values.find(key);
            if(found==o.values.end() || found->second.size()!=logical.size())
                throw std::runtime_error("missing serial trace for batched GLM tensor");
            size_t mismatches=0;double max_abs=0;bool finite=true;
            for(size_t i=0;i<logical.size();++i) {
                mismatches+=logical[i]!=found->second[i];
                if(node->type==GGML_TYPE_F32) {
                    float x,y;std::memcpy(&x,&found->second[i],4);std::memcpy(&y,&logical[i],4);
                    finite=finite&&std::isfinite(x)&&std::isfinite(y);
                    if(std::isfinite(x)&&std::isfinite(y))max_abs=std::max(max_abs,std::abs(double(x)-y));
                }
            }
            const bool passed=finite && mismatches==0;o.failed|=!passed;
            o.comparisons.push_back({{"position",key.first},{"name",name},{"elements",logical.size()},
                {"bit_mismatches",mismatches},{"max_abs",max_abs},{"finite",finite},{"passed",passed}});
        }
    }
    return true;
}
static std::vector<float> logits(llama_context * ctx,int row,int vocab) {
    const float * p=llama_get_logits_ith(ctx,row);
    if(!p || !std::all_of(p,p+vocab,[](float v){return std::isfinite(v);}))
        throw std::runtime_error("missing or nonfinite full logits");
    admit(size_t(vocab)*4);return {p,p+vocab};
}
static llama_token best(const std::vector<float>& values) {
    return llama_token(std::max_element(values.begin(),values.end())-values.begin());
}
static void decode(llama_context * ctx,const std::vector<llama_token>& ids,bool all=true) {
    if(ids.empty() || ids.size()>8)throw std::runtime_error("invalid verification batch");
    auto batch=llama_batch_init(int(ids.size()),0,1);
    const auto position=llama_memory_seq_pos_max(llama_get_memory(ctx),0)+1;
    batch.n_tokens=int(ids.size());
    for(size_t i=0;i<ids.size();++i) {
        batch.token[i]=ids[i];batch.pos[i]=position+int(i);batch.n_seq_id[i]=1;
        batch.seq_id[i][0]=0;batch.logits[i]=all || i+1==ids.size();
    }
    const int status=llama_decode(ctx,batch);llama_synchronize(ctx);llama_batch_free(batch);
    if(status)throw std::runtime_error("native decode failed: "+std::to_string(status));
    rss();
}
static json audit_case(llama_context * ctx,observer & o,llama_token pending,
                       const std::vector<llama_token>& draft,int vocab,const std::string & label) {
    const auto before=snapshot(ctx);const auto before_hash=sequence_hash(ctx);
    std::vector<llama_token> inputs{pending};inputs.insert(inputs.end(),draft.begin(),draft.end());
    std::vector<std::vector<float>> reference;std::vector<std::string> states;
    o.enabled=true;o.reference=true;o.failed=false;o.offset=0;o.tokens=1;o.bytes=0;
    o.values.clear();o.seen.clear();o.comparisons=json::array();o.writes.clear();o.phase=label+"/serial";
    for(size_t i=0;i<inputs.size();++i) {
        o.offset=int(i);decode(ctx,{inputs[i]});reference.push_back(logits(ctx,-1,vocab));
        states.push_back(sequence_hash(ctx));
    }
    const auto serial_write_count=o.writes.size();
    const bool serial_writes=o.writes==expected_targets();
    restore(ctx,before);const bool initial_restore=sequence_hash(ctx)==before_hash;
    o.reference=false;o.offset=0;o.tokens=int(inputs.size());o.seen.clear();o.writes.clear();o.phase=label+"/batch";
    decode(ctx,inputs);
    json out={{"label",label},{"draft_tokens",draft.size()},{"verification_tokens",inputs.size()},
        {"snapshot_bytes",before.size()},{"initial_restore_exact",initial_restore},
        {"tensor_comparisons",o.comparisons},{"full_logits",json::array()},
        {"serial_write_target_coverage",serial_write_count},{"batched_write_target_coverage",o.writes.size()}};
    bool passed=!o.failed && initial_restore && serial_writes && o.writes==expected_targets() &&
                o.seen.size()==inputs.size()*118 && o.values.size()==inputs.size()*118;
    std::vector<llama_token> decisions;
    for(size_t i=0;i<inputs.size();++i) {
        const auto values=logits(ctx,int(i),vocab);size_t mismatch=0;double max_abs=0;
        for(int j=0;j<vocab;++j) {
            mismatch+=std::memcmp(&reference[i][j],&values[j],4)!=0;
            max_abs=std::max(max_abs,std::abs(double(reference[i][j])-values[j]));
        }
        decisions.push_back(best(values));passed&=mismatch==0;
        out["full_logits"].push_back({{"position",i},{"elements",vocab},{"bit_mismatches",mismatch},
            {"max_abs",max_abs},{"argmax_equal",best(values)==best(reference[i])},{"passed",mismatch==0}});
    }
    out["full_sequence_state_exact"]=sequence_hash(ctx)==states.back();
    passed&=out["full_sequence_state_exact"].get<bool>();
    o.enabled=false;
    size_t accepted=0;while(accepted<draft.size() && draft[accepted]==decisions[accepted])++accepted;
    out["accepted_draft_tokens"]=accepted;out["replay_calls"]=0;
    if(accepted<draft.size()) {
        restore(ctx,before);
        for(size_t i=0;i<=accepted;++i)decode(ctx,{inputs[i]});
        out["replay_calls"]=accepted+1;
        out["rejected_prefix_state_exact"]=sequence_hash(ctx)==states[accepted];
        out["rejected_prefix_next_token_exact"]=best(logits(ctx,-1,vocab))==decisions[accepted];
        passed&=out["rejected_prefix_state_exact"].get<bool>() && out["rejected_prefix_next_token_exact"].get<bool>();
    }
    out["passed"]=passed;return out;
}
static json parameters() {
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=8;
    cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=10;cp.op_offload=false;cp.offload_kqv=false;cp.moe_cache_size=0;
    return {{"n_ctx",512},{"n_batch",8},{"n_ubatch",8},{"physical_slots",1},{"threads",10},
        {"gpu_layers",0},{"n_rs_seq",cp.n_rs_seq},{"second_model",false}};
}
int main(int argc,char ** argv) {
    llama_model * model=nullptr;llama_context * ctx=nullptr;std::filesystem::path root;json report;
    try {
        if(argc==2 && std::string(argv[1])=="--parameters") {std::cout<<parameters().dump()<<'\n';return 0;}
        if(argc==3 && std::string(argv[1])=="--draft-fixture") {
            std::ifstream file(argv[2]);const auto j=json::parse(file);
            const common_ngram_simple_config cfg{j.at("n").get<uint16_t>(),j.at("m").get<uint16_t>()};
            if(cfg.size_ngram<1 || cfg.size_ngram>4 || cfg.size_mgram<1 || cfg.size_mgram>4)
                throw std::runtime_error("fixture sizes out of range");
            std::cout<<json(common_ngram_simple_draft(cfg,j.at("history").get<llama_tokens>(),j.at("sampled"))).dump()<<'\n';return 0;
        }
        if(argc!=4)throw std::runtime_error("usage: ngram-probe MODEL_GGUF CONFIG_JSON OUTPUT_DIR");
        std::ifstream file(argv[2]);const auto config=json::parse(file);
        const std::string mode=config.at("mode");const int count=config.at("tokens"),n=config.at("ngram_size"),maximum=config.at("draft_tokens");
        if((mode!="audit" && mode!="reference" && mode!="ngram") || (count!=16 && count!=32 && count!=64 && count!=128) ||
           (n<1 || n>4) || (maximum!=2 && maximum!=4))throw std::runtime_error("unadmitted experiment parameters");
        root=argv[3];if(std::filesystem::exists(root))throw std::runtime_error("output already exists");
        std::filesystem::create_directories(root);
        report={{"complete",false},{"config",config},{"parameters",parameters()},
            {"performance_measured",false},{"production_changed",false},{"cases",json::array()}};save(root/"report.json",report);
        observer o;o.events.open(root/"state_writes.jsonl");o.events.exceptions(std::ios::badbit|std::ios::failbit);
        std::ofstream timing(root/"timing.jsonl");timing.exceptions(std::ios::badbit|std::ios::failbit);
        ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();
        mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.use_extra_bufts=false;ggml_backend_dev_t devices[]={nullptr};mp.devices=devices;
        model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=8;cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=10;cp.op_offload=false;cp.offload_kqv=false;cp.moe_cache_size=0;
        cp.cb_eval=evaluate;cp.cb_eval_user_data=&o;ctx=llama_init_from_model(model,cp);
        if(!ctx)throw std::runtime_error("context initialization failed");
        const auto * vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
        const std::string prompt=config.at("prompt");
        const int size=-llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,true,true);
        if(size<1 || size>400 || size+count>512)throw std::runtime_error("prompt exceeds unchanged context budget");
        std::vector<llama_token> history(size);
        if(llama_tokenize(vocab,prompt.data(),int(prompt.size()),history.data(),size,true,true)!=size)
            throw std::runtime_error("prompt tokenization changed size");
        save(root/"prompt_tokens.json",history);
        const auto prefill=timing_snapshot();
        for(int start=0;start<size;start+=8) {
            std::vector<llama_token> ids(history.begin()+start,history.begin()+std::min(size,start+8));decode(ctx,ids,false);
        }
        timing<<timing_delta("prefill",-1,prefill,timing_snapshot())<<'\n';timing.flush();
        auto pending=best(logits(ctx,-1,nv));
        if(mode=="audit") {
            const auto initial=snapshot(ctx);
            for(int d:{maximum}) {
                restore(ctx,initial);auto current=pending;std::vector<llama_token> correct;
                for(int i=0;i<d;++i) {decode(ctx,{current});current=best(logits(ctx,-1,nv));correct.push_back(current);}
                for(const std::string label:{"all-accepted","reject-first","reject-middle"}) {
                    if(label=="reject-middle" && d==1)continue;
                    restore(ctx,initial);auto draft=correct;
                    if(label!="all-accepted") {const int at=label=="reject-first"?0:d/2;draft[at]=(draft[at]+1)%nv;}
                    const auto result=audit_case(ctx,o,pending,draft,nv,"draft"+std::to_string(d)+"/"+label);
                    report["cases"].push_back(result);save(root/"report.json",report);
                    if(!result.at("passed").get<bool>())throw std::runtime_error("strict CPU fidelity failed; timed speculation is forbidden");
                }
            }
            report["qualification_passed"]=true;
            report["scope"]="forced oracle/mismatch diagnostic fixtures, not n-gram acceptance or performance";
        } else {
            const common_ngram_simple_config cfg{uint16_t(n),uint16_t(maximum)};
            std::vector<llama_token> generated;json blocks=json::array();
            int proposals=0,accepted_total=0,replay_calls=0,target_calls=0,target_positions=0;
            const auto begin=timing_snapshot();
            while(generated.size()<size_t(count)) {
                generated.push_back(pending);
                if(llama_vocab_is_eog(vocab,pending) || generated.size()==size_t(count))break;
                const auto block_begin=clock_type::now();
                auto draft=mode=="ngram"?common_ngram_simple_draft(cfg,history,pending):llama_tokens{};
                const size_t remaining=size_t(count)-generated.size();
                // Only the exact qualified shape is allowed. Short proposals
                // and the end of a request use serial N1, never an untested N2/N4.
                if(draft.size()!=size_t(maximum) || draft.size()>=remaining)draft.clear();
                std::vector<uint8_t> before;
                if(!draft.empty())before=snapshot(ctx);
                std::vector<llama_token> inputs{pending};inputs.insert(inputs.end(),draft.begin(),draft.end());
                const auto call_begin=timing_snapshot();decode(ctx,inputs);++target_calls;target_positions+=int(inputs.size());
                timing<<timing_delta("verify",int(generated.size()),call_begin,timing_snapshot())<<'\n';timing.flush();
                std::vector<llama_token> decisions;for(size_t i=0;i<inputs.size();++i)decisions.push_back(best(logits(ctx,int(i),nv)));
                size_t accepted=0;while(accepted<draft.size() && draft[accepted]==decisions[accepted] && !llama_vocab_is_eog(vocab,draft[accepted]))++accepted;
                const bool rejected=accepted<draft.size();
                if(rejected) {
                    restore(ctx,before);
                    for(size_t i=0;i<=accepted;++i) {decode(ctx,{inputs[i]});++replay_calls;++target_calls;++target_positions;}
                    if(best(logits(ctx,-1,nv))!=decisions[accepted])throw std::runtime_error("replay next-token fidelity failed");
                }
                history.push_back(pending);
                for(size_t i=0;i<accepted;++i) {generated.push_back(draft[i]);history.push_back(draft[i]);}
                pending=decisions[accepted];proposals+=int(draft.size());accepted_total+=int(accepted);
                blocks.push_back({{"draft",draft},{"accepted",accepted},{"rejected",rejected},
                    {"snapshot_bytes",before.size()},{"total_seconds",elapsed(block_begin)}});
            }
            const auto end=timing_snapshot();
            timing<<timing_delta("generation",-1,begin,end)<<'\n';timing.flush();
            const double seconds=std::chrono::duration<double>(end.stamp-begin.stamp).count();
            save(root/"generated.json",generated);save(root/"blocks.json",blocks);
            report.update({{"generated_tokens",generated.size()},{"decode_intervals",generated.size()-1},
                {"generation_seconds_excluding_prefill_first_token",seconds},
                {"decode_intervals_per_second",(generated.size()-1)/seconds},{"draft_proposals",proposals},
                {"accepted_draft_tokens",accepted_total},{"replay_calls",replay_calls},
                {"actual_target_calls",target_calls},{"actual_target_positions",target_positions},
                {"performance_measured",true},{"eog",llama_vocab_is_eog(vocab,generated.back())},
                {"scope","raw greedy single-request decode; proposal, snapshot, restore and replay costs included; no RONIN usefulness qualification"}});
        }
        llama_free(ctx);ctx=nullptr;llama_model_free(model);model=nullptr;llama_backend_free();
        report["complete"]=true;save(root/"report.json",report);save(root/"complete.json",{{"completed",true}});return 0;
    } catch(const std::exception & error) {
        if(ctx)llama_free(ctx);if(model)llama_model_free(model);
        if(!root.empty() && std::filesystem::is_directory(root)) {report["error"]=error.what();save(root/"report.json",report);}
        std::cerr<<error.what()<<'\n';return 2;
    }
}
