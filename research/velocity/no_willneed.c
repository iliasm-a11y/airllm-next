/* Process-local macOS research interposer: suppress only WILLNEED advice.
 *
 * This does not change tensor data, arithmetic, mmap, memory protection, or
 * ordinary page faults. All other advice is forwarded to the actual libc.
 * Use only in an explicitly owned benchmark process; never system-wide.
 */
#include <stddef.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static unsigned long long skipped_calls;
static unsigned long long skipped_bytes;

__attribute__((constructor)) static void mark_loaded(void) {
    static const char message[] = "AIRLLM_NO_WILLNEED_READY process_local_interposer_loaded\n";
    (void)write(STDERR_FILENO, message, sizeof(message)-1);
}

static void record_skip(const char *name, size_t bytes) {
    unsigned long long count = __atomic_add_fetch(&skipped_calls, 1, __ATOMIC_RELAXED);
    unsigned long long total = __atomic_add_fetch(&skipped_bytes, bytes, __ATOMIC_RELAXED);
    if (count <= 16) {
        dprintf(STDERR_FILENO,
            "AIRLLM_NO_WILLNEED function=%s calls=%llu bytes=%zu total_bytes=%llu\n",
            name, count, bytes, total);
    }
}

static int airllm_madvise(void *address, size_t length, int advice) {
    if (advice == MADV_WILLNEED) {
        record_skip("madvise", length);
        return 0;
    }
    /* dyld does not interpose bindings originating in the replacement image. */
    return madvise(address, length, advice);
}

#define AIRLLM_INTERPOSE(replacement, original) \
    __attribute__((used)) static struct { \
        const void *replacement_function; const void *original_function; \
    } airllm_interpose_##original __attribute__((section("__DATA,__interpose"))) = { \
        (const void *)(unsigned long)&replacement, (const void *)(unsigned long)&original \
    }

/* On this macOS libc, posix_madvise ultimately calls madvise: the owned probe
 * confirms both APIs are intercepted and retain their original error status. */
AIRLLM_INTERPOSE(airllm_madvise, madvise);
