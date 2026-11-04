// Private, default-off snapshot for one pinned CPU context.
#pragma once
#include "llama.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

class server_prefix_cache {
    static constexpr size_t mib = size_t(1) << 20;
    static constexpr size_t cap = 256 * mib;
    using clock = std::chrono::steady_clock;
    using hash_type = std::array<unsigned char, 32>;
    bool initialized_ = false;
    bool enabled_ = false;
    bool pending_ = false;
    size_t scratch_ = 0;
    uint64_t swap_baseline_ = 0;
    std::string key_;
    std::vector<llama_token> prefix_;
    std::vector<uint8_t> data_;
    hash_type digest_{};
    llama_context * bound_ctx_ = nullptr;

    struct memory_sample {
        bool valid = false;
        int pressure = 0;
        uint64_t available = 0, rss = 0, swap = 0;
    };

    static memory_sample memory() {
        memory_sample out;
#if defined(__APPLE__)
        size_t length = sizeof(out.pressure);
        if (sysctlbyname("kern.memorystatus_vm_pressure_level", &out.pressure, &length, nullptr, 0)) return out;
        xsw_usage swap{};
        length = sizeof(swap);
        if (sysctlbyname("vm.swapusage", &swap, &length, nullptr, 0)) return out;
        vm_statistics64_data_t vm{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        const mach_port_t host = mach_host_self();
        const auto rc = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count);
        mach_port_deallocate(mach_task_self(), host);
        const long page = sysconf(_SC_PAGESIZE);
        if (rc != KERN_SUCCESS || page <= 0) return out;
        mach_task_basic_info_data_t task{};
        count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&task), &count) != KERN_SUCCESS) return out;
        out.available = (uint64_t(vm.free_count) + vm.inactive_count) * uint64_t(page);
        out.rss = task.resident_size;
        out.swap = swap.xsu_used;
        out.valid = true;
#endif
        return out;
    }

    bool admit(size_t extra) const {
        const auto m = memory();
        constexpr uint64_t rss_cap = uint64_t(17) << 29;
        return m.valid && m.pressure == 1 && extra <= rss_cap && m.rss <= rss_cap - extra &&
            m.available >= (uint64_t(3) << 30) + extra &&
            (m.swap <= swap_baseline_ || m.swap - swap_baseline_ <= 32 * mib);
    }

    static hash_type digest(const std::vector<uint8_t> & bytes) {
        hash_type out{};
#if defined(__APPLE__)
        CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), out.data());
#endif
        return out;
    }

    void initialize() {
        if (initialized_) return;
        const char * path = std::getenv("AIRLLM_SHARED_PREFIX_MANIFEST");
        if (!path || !*path) { initialized_ = true; return; }
#if defined(__APPLE__)
        const int fd = open(path, O_RDONLY | O_NOFOLLOW);
        if (fd < 0) throw std::runtime_error("cannot open private prefix manifest");
        std::string text;
        try {
            struct stat before{}, after{};
            if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_uid != getuid() ||
                (before.st_mode & 077) || before.st_size < 1 || before.st_size > 128 * 1024)
                throw std::runtime_error("invalid private prefix manifest");
            text.resize(static_cast<size_t>(before.st_size));
            size_t offset = 0;
            while (offset < text.size()) {
                const auto n = read(fd, &text[offset], text.size() - offset);
                if (n <= 0) throw std::runtime_error("short prefix manifest read");
                offset += static_cast<size_t>(n);
            }
            if (fstat(fd, &after) || before.st_size != after.st_size ||
                before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec ||
                before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec ||
                before.st_ctimespec.tv_sec != after.st_ctimespec.tv_sec ||
                before.st_ctimespec.tv_nsec != after.st_ctimespec.tv_nsec)
                throw std::runtime_error("prefix manifest changed during read");
        } catch (...) { close(fd); throw; }
        close(fd);
        const auto record = json::parse(text);
        if (record.at("schema") != 1 || !record.at("prefix_tokens").is_array() ||
            record.at("prefix_tokens").empty() || record.at("prefix_tokens").size() >= 8192)
            throw std::runtime_error("invalid prefix manifest schema");
        std::vector<llama_token> tokens;
        for (const auto & token : record.at("prefix_tokens")) {
            if (!token.is_number_integer() || token.get<int64_t>() < 0 || token.get<int64_t>() >= 154880)
                throw std::runtime_error("invalid native prefix token");
            tokens.push_back(token.get<llama_token>());
        }
        const auto key = record.at("key_sha256").get<std::string>();
        if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos || tokens.size() % 8)
            throw std::runtime_error("invalid aligned prefix identity");
        if (!record.at("scratch_bytes").is_number_integer()) throw std::runtime_error("missing snapshot scratch policy");
        const auto scratch = record.at("scratch_bytes").get<int64_t>();
        if (scratch < int64_t(mib) || scratch > int64_t(64 * mib)) throw std::runtime_error("invalid snapshot scratch policy");
        const auto m = memory();
        if (!m.valid) throw std::runtime_error("prefix memory accounting unavailable");
        prefix_ = std::move(tokens);
        key_ = key;
        scratch_ = static_cast<size_t>(scratch);
        swap_baseline_ = m.swap;
        enabled_ = true;
        initialized_ = true;
#else
        throw std::runtime_error("private prefix cache requires Darwin memory guards");
#endif
    }

public:
    bool enabled() { initialize(); return enabled_; }
    size_t boundary() const { return prefix_.size(); }
    bool pending() const { return pending_; }
    void cancel_pending() { pending_ = false; }

    template<class Tokens> bool matches(const Tokens & tokens) const {
        if (!enabled_ || tokens.size() <= prefix_.size()) return false;
        for (size_t i = 0; i < prefix_.size(); ++i) if (tokens[i] != prefix_[i]) return false;
        return true;
    }

    size_t begin(llama_context * ctx) {
        const auto start = clock::now();
        pending_ = false;
        if (llama_n_seq_max(ctx) != 1 || llama_n_batch(ctx) != 8 || llama_n_ubatch(ctx) != 8)
            throw std::runtime_error("prefix cache requires one batch8/ubatch8 sequence");
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (data_.empty()) { pending_ = true; return 0; }
        if (bound_ctx_ != ctx || !admit(scratch_) || digest(data_) != digest_)
            throw std::runtime_error("prefix restore identity, integrity or admission failed");
        if (llama_state_seq_set_data_ext(ctx, data_.data(), data_.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != data_.size())
            throw std::runtime_error("short complete prefix restore");
        llama_synchronize(ctx);
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != llama_pos(prefix_.size() - 1) || !admit(0))
            throw std::runtime_error("prefix restore position or memory guard failed");
        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
        std::fprintf(stderr, "AIRLLM_SHARED_PREFIX event=restore tokens=%zu bytes=%zu seconds=%.9f key=%s\n", prefix_.size(), data_.size(), seconds, key_.c_str());
        return prefix_.size();
    }

    void capture(llama_context * ctx) {
        if (!pending_) return;
        const auto start = clock::now();
        pending_ = false;
        llama_synchronize(ctx);
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != llama_pos(prefix_.size() - 1))
            throw std::runtime_error("prefix capture position mismatch");
        const size_t bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!bytes || bytes > cap || !admit(bytes + scratch_)) {
            std::fprintf(stderr, "AIRLLM_SHARED_PREFIX event=capture_declined tokens=%zu\n", prefix_.size());
            return;
        }
        data_.resize(bytes);
        if (llama_state_seq_get_data_ext(ctx, data_.data(), bytes, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != bytes)
            throw std::runtime_error("short complete prefix capture");
        digest_ = digest(data_);
        bound_ctx_ = ctx;
        if (!admit(0)) throw std::runtime_error("prefix capture exceeded memory guards");
        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
        std::fprintf(stderr, "AIRLLM_SHARED_PREFIX event=capture tokens=%zu bytes=%zu seconds=%.9f key=%s\n", prefix_.size(), bytes, seconds, key_.c_str());
    }

    void reset() {
        std::vector<uint8_t>().swap(data_);
        std::vector<llama_token>().swap(prefix_);
        key_.clear();
        initialized_ = enabled_ = pending_ = false;
        bound_ctx_ = nullptr;
    }
};
