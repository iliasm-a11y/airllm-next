/* Minimal owned-process smoke: no model or disk payload is loaded. */
#include <errno.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    size_t size = (size_t)sysconf(_SC_PAGESIZE);
    void *address = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (address == MAP_FAILED) return 2;
    int posix_willneed = posix_madvise(address, size, POSIX_MADV_WILLNEED);
    int willneed = madvise(address, size, MADV_WILLNEED);
    int random = posix_madvise(address, size, POSIX_MADV_RANDOM);
    errno = 0;
    int invalid = posix_madvise(address, size, 123456);
    int posix_errno = errno;
    errno = 0;
    int native_invalid = madvise(address, size, 123456);
    int native_errno = errno;
    munmap(address, size);
    printf("{\"posix_willneed\":%d,\"madvise_willneed\":%d,\"random\":%d,"
           "\"posix_invalid\":%d,\"posix_errno\":%d,\"madvise_invalid\":%d,\"madvise_errno\":%d}\n",
           posix_willneed, willneed, random, invalid, posix_errno, native_invalid, native_errno);
    /* Darwin aliases posix_madvise to madvise; invalid advice returns -1. */
    return posix_willneed || willneed || random || invalid != -1 || posix_errno != EINVAL ||
           native_invalid != -1 || native_errno != EINVAL;
}
