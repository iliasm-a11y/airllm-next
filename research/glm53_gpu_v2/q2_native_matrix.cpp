// Small benchmark/oracle adapter, not part of the runtime. MIT license.
// One Python call dispatches the pinned library's existing ARM Q2_K dot.
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

extern "C" void ggml_vec_dot_q2_K_q8_K(int, float*, size_t, const void*, size_t,
                                       const void*, size_t, int);

class q2_group_pool {
    const int rows, width, nthreads, rows_per_expert, activations;
    std::vector<std::thread> workers;
    std::mutex mutex, invocation;
    std::condition_variable ready, done;
    bool stopping = false;
    unsigned generation = 0;
    int pending = 0;
    const uint8_t* packed = nullptr;
    const void* q8 = nullptr;
    float* output = nullptr;
    void calculate(int thread_id, const uint8_t* p, const void* q, float* y) {
        const size_t row_bytes = size_t(width / 256) * 84;
        for (int row = thread_id; row < rows; row += nthreads) {
            ggml_vec_dot_q2_K_q8_K(width, y + row, 0, p + row * row_bytes, 0, static_cast<const uint8_t*>(q) + (activations == 1 ? 0 : row / rows_per_expert) * size_t(width / 256) * 292, 0, 1);
        }
    }
public:
    q2_group_pool(int e, int r, int w, int a, int n): rows(e*r), width(w), nthreads(n), rows_per_expert(r), activations(a) {
        if (nthreads == 1) return;
        try {
        for (int id = 0; id < nthreads; ++id) {
            workers.emplace_back([this, id] {
                unsigned seen = 0;
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    ready.wait(lock, [&] { return stopping || generation != seen; });
                    if (stopping) return;
                    seen = generation;
                    auto p = packed; auto q = q8; auto y = output;
                    lock.unlock();
                    calculate(id, p, q, y);
                    lock.lock();
                    if (--pending == 0) done.notify_one();
                }
            });
        }
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            ready.notify_all();
            for (auto& worker : workers) worker.join();
            throw;
        }
    }
    ~q2_group_pool() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto& worker : workers) worker.join();
    }
    void run(const uint8_t* p, const void* q, float* y) {
        std::lock_guard<std::mutex> call_lock(invocation);
        if (nthreads == 1) { calculate(0, p, q, y); return; }
        std::unique_lock<std::mutex> lock(mutex);
        packed = p; q8 = q; output = y; pending = nthreads; ++generation;
        ready.notify_all();
        done.wait(lock, [&] { return pending == 0; });
    }
};

extern "C" void* airllm_q2_group_create(int experts, int rows, int width, int activations, int nthreads) {
    if (experts < 1 || experts > 8 || rows < 1 || rows > 8192 || width < 256 || width > 65536 || width % 256 ||
        (activations != 1 && activations != experts) || nthreads < 1 || nthreads > 10 ||
        size_t(experts)*rows*(width/256)*84 > (32<<20)) return nullptr;
    try { return new q2_group_pool(experts, rows, width, activations, nthreads); } catch (...) { return nullptr; }
}
extern "C" int airllm_q2_group_run(void* handle, const uint8_t* p, const void* q, float* y) {
    if (!handle || !p || !q || !y) return -1;
    static_cast<q2_group_pool*>(handle)->run(p, q, y);
    return 0;
}
extern "C" void airllm_q2_group_destroy(void* handle) {
    delete static_cast<q2_group_pool*>(handle);
}
