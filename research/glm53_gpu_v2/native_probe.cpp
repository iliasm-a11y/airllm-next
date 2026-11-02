// Experimental placement diagnostic. Captures serialize operations; not a throughput benchmark.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <regex>
#include <cstdlib>
#include <cmath>
#if defined(__APPLE__)
#include <libproc.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <string>
#include <vector>

struct capture {
    std::filesystem::path directory;
    std::ofstream manifest;
    int step = 0;
    int serial = 0;
    bool profile = false;
    bool failed = false;
    std::ofstream operations;
    std::chrono::steady_clock::time_point operation_start;
#if defined(__APPLE__)
    rusage_info_v2 before{};
    bool before_valid = false;
#endif
};

static bool evaluate(ggml_tensor * tensor, bool ask, void * user) {
    const std::string name = ggml_get_name(tensor);
    const bool selected = name.find("ffn_moe_topk-") == 0 ||
                          name.find("ffn_moe_logits-") == 0 ||
                          name.find("kda_scan_out-") == 0 || name == "result_output";
    auto & out = *static_cast<capture *>(user);
    if (out.profile) {
        if (ask) {
#if defined(__APPLE__)
            out.before_valid = proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                reinterpret_cast<rusage_info_t *>(&out.before)) == 0;
#endif
            out.operation_start = std::chrono::steady_clock::now();
        } else {
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now-out.operation_start).count();
            out.operations << "{\"step\":" << out.step << ",\"name\":\"" << name
                << "\",\"op\":\"" << ggml_op_name(tensor->op) << "\",\"seconds\":" << seconds;
#if defined(__APPLE__)
            rusage_info_v2 after{};
            if (out.before_valid && proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                    reinterpret_cast<rusage_info_t *>(&after)) == 0) {
                mach_timebase_info_data_t scale{};
                mach_timebase_info(&scale);
                const double tick_seconds = double(scale.numer)/scale.denom/1e9;
                out.operations << ",\"cpu_seconds\":" << (after.ri_user_time+after.ri_system_time-
                    out.before.ri_user_time-out.before.ri_system_time)*tick_seconds
                    << ",\"os_read_bytes\":" << after.ri_diskio_bytesread-out.before.ri_diskio_bytesread
                    << ",\"pageins\":" << after.ri_pageins-out.before.ri_pageins;
            }
#endif
            out.operations << "}\n";
        }
    }
    if (ask) {
        return selected || out.profile;
    }
    if (!selected) {
        return true;
    }
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes > 64 * 1024 * 1024 || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32)) {
        std::fprintf(stderr, "Unsupported diagnostic tensor: %s\n", name.c_str());
        out.failed = true;
        return false;
    }
    std::vector<unsigned char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    const std::string file = std::to_string(out.step) + "-" + std::to_string(out.serial++) + ".bin";
    std::ofstream stream(out.directory / file, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(reinterpret_cast<const char *>(data.data()), data.size());
    out.manifest << "{\"step\":" << out.step << ",\"name\":\"" << name << "\",\"file\":\"" << file
                 << "\",\"type\":" << int(tensor->type) << ",\"bytes\":" << bytes << ",\"shape\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.manifest << (i ? "," : "") << tensor->ne[i];
    }
    out.manifest << "],\"strides\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.manifest << (i ? "," : "") << tensor->nb[i];
    }
    out.manifest << "]}\n";
    out.manifest.flush();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 9) {
        std::fprintf(stderr, "usage: native_fidelity model.gguf output_dir threads gpu_layers tokens [prompt [forced_tokens.jsonl [cpu|head-only|quant-tail|weights-only|legacy]]]\n");
        return 2;
    }
    const int threads = std::stoi(argv[3]);
    const int gpu_layers = std::stoi(argv[4]);
    const int count = std::stoi(argv[5]);
    if (threads < 1 || threads > 10 || gpu_layers < 0 || gpu_layers > 46 || count < 1 || count > 64) {
        return 2;
    }
    const std::string policy = argc == 9 ? argv[8] : (gpu_layers ? "weights-only" : "cpu");
    const bool head_only = policy == "head-only";
    const bool quant_tail = policy == "quant-tail";
    const bool legacy = policy == "legacy";
    if ((policy != "cpu" && !head_only && !quant_tail && policy != "weights-only" && !legacy) ||
        (policy == "cpu" && gpu_layers != 0) || ((head_only || quant_tail) && gpu_layers != 0) ||
        ((policy == "weights-only" || legacy) && gpu_layers == 0)) {
        throw std::runtime_error("Policy and GPU layers disagree");
    }
    int cache_mib = gpu_layers > 0 ? 256 : 0;
    if (const char * value = std::getenv("AIRLLM_GPU_CACHE_MIB")) {
        if (!std::regex_match(value, std::regex("0|128|256|384|512"))) {
            throw std::runtime_error("Invalid bounded GPU cache budget");
        }
        cache_mib = std::stoi(value);
        if (!gpu_layers && cache_mib) {
            throw std::runtime_error("GPU expert cache requires GPU weight layers");
        }
    }
    const std::string prompt = argc >= 7 ? argv[6] : "La capitale de la France est";
    std::vector<llama_token> forced;
    if (argc >= 8 && std::strcmp(argv[7], "-") != 0) {
        std::ifstream source(argv[7]);
        if (!source || std::filesystem::file_size(argv[7]) > 65536) {
            throw std::runtime_error("Invalid forced-token capture");
        }
        const std::regex pattern("\"token\"\\s*:\\s*([0-9]+)");
        std::string line;
        while (std::getline(source, line)) {
            std::smatch match;
            if (!std::regex_search(line, match, pattern)) {
                throw std::runtime_error("Invalid forced-token row");
            }
            forced.push_back(std::stoi(match[1]));
        }
        if (forced.size() != size_t(count)) {
            throw std::runtime_error("Forced-token count differs from requested count");
        }
    }
    const std::filesystem::path root(argv[2]);
    if (std::filesystem::exists(root)) {
        throw std::runtime_error("Refusing to overwrite an existing capture");
    }
    std::filesystem::create_directories(root);
    std::ofstream(root / "run.json") << "{\"threads\":" << threads << ",\"gpu_layers\":" << gpu_layers
        << ",\"count\":" << count << ",\"context\":512,\"batch\":8,\"moe_cache_mib\":"
        << cache_mib << ",\"policy\":\"" << policy << "\",\"op_offload\":" << (legacy ? "true" : "false")
        << ",\"offload_kqv\":" << (legacy ? "true" : "false")
        << ",\"forced\":" << (forced.empty() ? "false" : "true") << "}\n";
    std::ofstream knobs(root / "environment.json");
    knobs << "{";
    bool comma = false;
    for (const char * name : {"AIRLLM_MMAP_DEMAND", "AIRLLM_EXPERT_PREFETCH", "AIRLLM_DENSE_PREFETCH",
            "AIRLLM_DENSE_PREFETCH_MAX_COLS", "AIRLLM_GPU_CACHE_MIB", "AIRLLM_GLM_HC_POST", "AIRLLM_GLM_LAYER_LOCAL_HC",
            "AIRLLM_GLM_CACHE_SPLIT"}) {
        const char * value = std::getenv(name);
        if (value) {
            if (!std::regex_match(value, std::regex("[0-9]+"))) {
                throw std::runtime_error("Non-numeric experimental knob");
            }
            knobs << (comma ? "," : "") << "\"" << name << "\":\"" << value << "\"";
            comma = true;
        }
    }
    knobs << "}\n";
    knobs.close();
    capture captured{root, std::ofstream(root / "tensors.jsonl")};
    captured.manifest.exceptions(std::ios::failbit | std::ios::badbit);
    const char * profile = std::getenv("AIRLLM_PROFILE_OPS");
    captured.profile = profile && std::strcmp(profile, "1") == 0;
    if (captured.profile) {
        captured.operations.open(root / "operations.jsonl");
        captured.operations.exceptions(std::ios::failbit | std::ios::badbit);
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    ggml_backend_dev_t gpu = (head_only || quant_tail) ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    if ((head_only || quant_tail) && (!gpu || std::string(ggml_backend_dev_name(gpu)).find("MTL") != 0)) {
        throw std::runtime_error("head-only requires a Metal device");
    }
    ggml_backend_dev_t devices[] = { gpu, nullptr };
    if (gpu_layers == 0) {
        mp.devices = devices;
    }
    llama_model_tensor_buft_override overrides[] = {
        {".*ffn_(gate|up|down)_exps.*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
    };
    if (gpu_layers > 0) {
        mp.tensor_buft_overrides = overrides;
    }
    const char * tail_pattern = std::getenv("AIRLLM_Q8_TAIL_PATTERN");
    if (quant_tail && (!tail_pattern || std::strlen(tail_pattern) > 8192)) {
        throw std::runtime_error("Missing or oversized Q8 placement plan");
    }
    llama_model_tensor_buft_override tail_overrides[] = {
        {tail_pattern, quant_tail ? ggml_backend_dev_buffer_type(gpu) : nullptr}, {nullptr, nullptr}
    };
    if (quant_tail) mp.tensor_buft_overrides = tail_overrides;
    llama_model_tensor_buft_override head_overrides[] = {
        {"^output\\.weight$", head_only ? ggml_backend_dev_buffer_type(gpu) : nullptr}, {nullptr, nullptr}
    };
    if (head_only) {
        mp.tensor_buft_overrides = head_overrides;
    }
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 3;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.op_offload = legacy;
    cp.offload_kqv = legacy;
    cp.moe_cache_size = size_t(cache_mib) * 1024 * 1024;
    cp.cb_eval = evaluate;
    cp.cb_eval_user_data = &captured;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 4;
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (n < 1 || n > 400) {
        throw std::runtime_error("Invalid diagnostic prompt length");
    }
    std::vector<llama_token> input(n);
    n = llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true);
    if (n < 1 || n > 400) {
        throw std::runtime_error("Invalid diagnostic prompt length");
    }
    std::ofstream prompt_tokens(root / "prompt_tokens.json");
    prompt_tokens << "[";
    for (int i = 0; i < n; ++i) {
        prompt_tokens << (i ? "," : "") << input[i];
    }
    prompt_tokens << "]\n";
    prompt_tokens.close();
    for (const auto token : forced) {
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("Forced token outside vocabulary");
        }
    }
    std::ofstream tokens(root / "tokens.jsonl");
    tokens.exceptions(std::ios::failbit | std::ios::badbit);
    int captured_steps = 0;
    for (int offset = 0; offset < n;) {
        const int take = std::min(8, n-offset);
        auto batch = llama_batch_get_one(input.data()+offset, take);
        if (llama_decode(ctx, batch) != 0 || captured.failed) {
            return 5;
        }
        offset += take;
    }
    for (int step = 0; step < count; ++step) {
        captured.step = step;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            throw std::runtime_error("Missing native logits");
        }
        if (!std::all_of(logits, logits+n_vocab, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("Non-finite native logits");
        }
        const llama_token token = std::max_element(logits, logits+n_vocab)-logits;
        std::ofstream binary(root / ("logits-" + std::to_string(step) + ".f32"), std::ios::binary);
        binary.exceptions(std::ios::failbit | std::ios::badbit);
        binary.write(reinterpret_cast<const char *>(logits), n_vocab*sizeof(float));
        const auto next_input = forced.empty() ? token : forced[step];
        tokens << "{\"step\":" << step << ",\"token\":" << token << ",\"next_input_token\":"
               << next_input << ",\"vocab\":" << n_vocab << "}\n";
        tokens.flush();
        captured_steps++;
        if ((forced.empty() && llama_vocab_is_eog(vocab, token)) || step+1 == count) {
            break;
        }
        auto next = next_input;
        captured.step = step+1;
        auto batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0 || captured.failed) {
            return 6;
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::ofstream(root / "complete.json") << "{\"completed\":true,\"captured_steps\":" << captured_steps << "}\n";
    return 0;
}
