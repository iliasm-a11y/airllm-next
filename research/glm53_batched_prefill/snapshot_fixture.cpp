#include "snapshot.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

// A native API fixture, not a GLM state or model fidelity test.
struct llama_memory_i {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(32, 42);
    llama_pos pos = 207;
    int clears = 0;
};
struct llama_context {
    llama_memory_i memory;
    uint32_t slots = 1;
    size_t advertised = 32;
    bool short_write = false;
    bool short_read = false;
    int writes = 0;
    int reads = 0;
};
static uint8_t * written_buffer = nullptr;

extern "C" {
llama_memory_t llama_get_memory(const llama_context * ctx) {
    return const_cast<llama_memory_t>(&ctx->memory);
}
uint32_t llama_n_seq_max(const llama_context * ctx) { return ctx->slots; }
void llama_synchronize(llama_context *) {}
llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq) {
    assert(seq == 0);
    return mem->pos;
}
void llama_memory_clear(llama_memory_t mem, bool data) {
    assert(data);
    mem->pos = -1;
    std::fill(mem->bytes.begin(), mem->bytes.end(), 0);
    ++mem->clears;
}
size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq,
                                  llama_state_seq_flags flags) {
    assert(seq == 0 && flags == LLAMA_STATE_SEQ_FLAGS_NONE);
    return ctx->advertised;
}
size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t n,
                                  llama_seq_id seq, llama_state_seq_flags flags) {
    assert(seq == 0 && flags == LLAMA_STATE_SEQ_FLAGS_NONE);
    assert(n == ctx->memory.bytes.size());
    written_buffer = dst;
    std::memcpy(dst, ctx->memory.bytes.data(), n);
    ++ctx->writes;
    return ctx->short_write ? n - 1 : n;
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t n,
                                  llama_seq_id seq, llama_state_seq_flags flags) {
    assert(seq == 0 && flags == LLAMA_STATE_SEQ_FLAGS_NONE);
    assert(ctx->memory.pos == -1);
    assert(std::all_of(ctx->memory.bytes.begin(), ctx->memory.bytes.end(),
                       [](uint8_t x) { return x == 0; }));
    ctx->memory.bytes.assign(src, src + n);
    ctx->memory.pos = 207;
    ++ctx->reads;
    return ctx->short_read ? n - 1 : n;
}
}

template<class F> static void refuses(F fn) {
    bool refused = false;
    try { fn(); } catch (const std::runtime_error &) { refused = true; }
    assert(refused);
}

int main() {
    using glm53_prefill::prefix_snapshot;
    const std::string key(64, 'a');
    int guard_calls = 0;
    auto guard = [&](size_t bytes, const char *) {
        assert(bytes == 32 + 1024);
        ++guard_calls;
    };
    {
        llama_context ctx;
        prefix_snapshot snap;
        snap.capture(&ctx, key, 208, 1024, guard);
        assert(snap.ready() && snap.metrics().bytes == 32);
        assert(snap.checksum_sha256().size() == 64);
        for (int agent : {1, 600, 1}) {
            std::fill(ctx.memory.bytes.begin(), ctx.memory.bytes.end(), uint8_t(agent));
            ctx.memory.pos = 447;
            assert(snap.restore(&ctx, key, 239, guard) == 208);
            assert(ctx.memory.bytes == std::vector<uint8_t>(32, 42));
        }
        assert(ctx.writes == 1 && ctx.reads == 3 && ctx.memory.clears == 3);
        assert(snap.metrics().successful_restores == 3);
        refuses([&] { snap.capture(&ctx, key, 208, 1024, guard); });
        assert(snap.ready() && ctx.writes == 1);
        snap.invalidate(&ctx);
        assert(!snap.ready() && ctx.memory.pos == -1);
    }
    {
        llama_context ctx;
        ctx.advertised = prefix_snapshot::cap_bytes + 1;
        prefix_snapshot snap;
        refuses([&] { snap.capture(&ctx, key, 208, 1024, guard); });
        assert(!snap.ready() && ctx.writes == 0 && ctx.memory.pos == -1);
        assert(ctx.memory.bytes == std::vector<uint8_t>(32, 0));
    }
    {
        llama_context ctx;
        prefix_snapshot snap;
        auto deny = [](size_t, const char *) { throw std::runtime_error("guard"); };
        refuses([&] { snap.capture(&ctx, key, 208, 1024, deny); });
        assert(!snap.ready() && ctx.writes == 0 && ctx.memory.pos == -1);
        assert(ctx.memory.bytes == std::vector<uint8_t>(32, 0));
    }
    {
        llama_context ctx;
        ctx.short_write = true;
        prefix_snapshot snap;
        refuses([&] { snap.capture(&ctx, key, 208, 1024, guard); });
        assert(!snap.ready() && ctx.memory.pos == -1);
    }
    for (int fault = 0; fault < 4; ++fault) {
        llama_context ctx;
        prefix_snapshot snap;
        snap.capture(&ctx, key, 208, 1024, guard);
        if (fault == 0) ctx.short_read = true;
        if (fault == 1) written_buffer[0] ^= 1;
        const auto candidate = fault == 2 ? std::string(64, 'b') : key;
        refuses([&] { snap.restore(&ctx, candidate, fault == 3 ? 0 : 239, guard); });
        assert(!snap.ready() && ctx.memory.pos == -1);
        if (fault != 0) assert(ctx.reads == 0);
    }
    {
        llama_context ctx;
        prefix_snapshot snap;
        snap.capture(&ctx, key, 208, 1024, guard);
        auto cancel = [](size_t, const char *) { throw std::runtime_error("cancelled"); };
        refuses([&] { snap.restore(&ctx, key, 239, cancel); });
        assert(!snap.ready() && ctx.memory.pos == -1 && ctx.reads == 0);
    }
    {
        llama_context ctx;
        ctx.slots = 2;
        prefix_snapshot snap;
        refuses([&] { snap.capture(&ctx, key, 208, 1024, guard); });
        assert(ctx.memory.pos == 207 && ctx.memory.clears == 0);
        ctx.slots = 1;
        ctx.memory.pos = 208;
        refuses([&] { snap.capture(&ctx, key, 208, 1024, guard); });
        assert(ctx.writes == 0 && ctx.memory.pos == -1);
    }
    assert(guard_calls > 0);
    std::cout << "{\"fixture_passed\":true,\"native_model_loaded\":false,"
                 "\"scope\":\"snapshot ownership, flags, guard, reset and failure contracts\"}\n";
}
