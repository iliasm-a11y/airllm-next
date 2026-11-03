#pragma once

#include "llama.h"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace glm53_prefill {

// Research only. The caller owns the one-slot lock, watchdog and fresh sampler.
// This stores sequence memory, not a resumable generation or its output logits.
class prefix_snapshot {
public:
    static constexpr size_t cap_bytes = 256ULL * 1024 * 1024;
    using guard_fn = std::function<void(size_t, const char *)>;

    struct timing {
        size_t bytes = 0;
        size_t reserved_bytes = 0;
        double size_seconds = 0;
        double save_seconds = 0;
        double restore_seconds = 0;
        size_t successful_restores = 0;
    };

    prefix_snapshot() = default;
    prefix_snapshot(const prefix_snapshot &) = delete;
    prefix_snapshot & operator=(const prefix_snapshot &) = delete;
    prefix_snapshot(prefix_snapshot &&) = delete;
    prefix_snapshot & operator=(prefix_snapshot &&) = delete;

    bool ready() const { return bool(data_); }
    const timing & metrics() const { return timing_; }
    std::string checksum_sha256() const {
        static constexpr char hex[] = "0123456789abcdef";
        std::string result;
        if (!ready()) return result;
        result.reserve(digest_.size() * 2);
        for (unsigned char value : digest_) {
            result.push_back(hex[value >> 4]);
            result.push_back(hex[value & 15]);
        }
        return result;
    }

    // native_scratch_bound is a separately audited/measured simultaneous budget.
    // The guard must admit it together with active state, graphs and other buffers.
    void capture(llama_context * ctx, const std::string & key_sha256,
                 size_t prefix_tokens, size_t native_scratch_bound,
                 const guard_fn & guard) {
        if (ready()) throw std::runtime_error("immutable snapshot already exists");
        check_context(ctx);
        try {
            check_key(key_sha256);
            if (!guard || !native_scratch_bound || !prefix_tokens ||
                prefix_tokens > size_t(std::numeric_limits<llama_pos>::max())) {
                throw std::runtime_error("missing prefix, scratch bound or guard");
            }
            auto start = clock::now();
            llama_synchronize(ctx);
            if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != llama_pos(prefix_tokens - 1)) {
                throw std::runtime_error("active state is not at the prefix boundary");
            }
            const size_t bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            timing_ = {};
            timing_.size_seconds = elapsed(start);
            if (!bytes || bytes > cap_bytes || native_scratch_bound >
                std::numeric_limits<size_t>::max() - bytes) {
                throw std::runtime_error("snapshot size refused before allocation");
            }
            const size_t reserve = bytes + native_scratch_bound;
            guard(reserve, "before_snapshot_allocation");
            data_.reset(new uint8_t[bytes]);
            size_ = bytes;
            timing_.bytes = bytes;
            timing_.reserved_bytes = reserve;
            guard(reserve, "before_snapshot_write");
            start = clock::now();
            const size_t written = llama_state_seq_get_data_ext(
                ctx, data_.get(), size_, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            timing_.save_seconds = elapsed(start);
            if (written != size_) throw std::runtime_error("short snapshot write");
            digest_ = digest(data_.get(), size_);
            guard(reserve, "after_snapshot_write");
            key_ = key_sha256;
            prefix_tokens_ = prefix_tokens;
        } catch (...) {
            invalidate(ctx);
            throw;
        }
    }

    // A nonempty suffix must run next to replace old logits before any sampling.
    size_t restore(llama_context * ctx, const std::string & key_sha256,
                   size_t suffix_tokens, const guard_fn & guard) {
        try {
            check_context(ctx);
            check_key(key_sha256);
            if (!ready() || key_sha256 != key_ || !guard || !suffix_tokens) {
                throw std::runtime_error("snapshot key, suffix or guard refused");
            }
            guard(timing_.reserved_bytes, "before_snapshot_restore");
            if (digest(data_.get(), size_) != digest_) {
                throw std::runtime_error("snapshot integrity mismatch");
            }
            llama_synchronize(ctx);
            llama_memory_clear(llama_get_memory(ctx), true);
            const auto start = clock::now();
            const size_t read = llama_state_seq_set_data_ext(
                ctx, data_.get(), size_, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            llama_synchronize(ctx);
            timing_.restore_seconds += elapsed(start);
            if (read != size_ || llama_memory_seq_pos_max(llama_get_memory(ctx), 0) !=
                llama_pos(prefix_tokens_ - 1)) {
                throw std::runtime_error("short or misplaced snapshot restore");
            }
            guard(timing_.reserved_bytes, "after_snapshot_restore");
            ++timing_.successful_restores;
            return prefix_tokens_;
        } catch (...) {
            invalidate(ctx);
            throw;
        }
    }

    void invalidate(llama_context * ctx) {
        if (ctx) {
            llama_synchronize(ctx);
            if (llama_get_memory(ctx)) llama_memory_clear(llama_get_memory(ctx), true);
        }
        data_.reset();
        size_ = 0;
        prefix_tokens_ = 0;
        key_.clear();
        digest_ = {};
    }

private:
    using clock = std::chrono::steady_clock;
    using digest_type = std::array<unsigned char, CC_SHA256_DIGEST_LENGTH>;
    static double elapsed(clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    }
    static digest_type digest(const uint8_t * data, size_t n) {
        digest_type result{};
        CC_SHA256(data, CC_LONG(n), result.data());
        return result;
    }
    static void check_key(const std::string & key) {
        if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos) {
            throw std::runtime_error("a canonical SHA256 cache key is required");
        }
    }
    static void check_context(llama_context * ctx) {
        if (!ctx || !llama_get_memory(ctx) || llama_n_seq_max(ctx) != 1) {
            throw std::runtime_error("snapshot requires one physical sequence");
        }
    }

    std::unique_ptr<uint8_t[]> data_;
    size_t size_ = 0;
    size_t prefix_tokens_ = 0;
    std::string key_;
    digest_type digest_{};
    timing timing_{};
};

} // namespace glm53_prefill
