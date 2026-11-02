#if defined(__APPLE__)
static bool airllm_expert_drop_readonly_file(uintptr_t begin, uintptr_t end) {
    for (unsigned regions = 0; begin < end && regions < 64; ++regions) {
        mach_vm_address_t address = begin;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t basic;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t status = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
                                             (vm_region_info_t) &basic, &count, &object);
        if (object != MACH_PORT_NULL) { mach_port_deallocate(mach_task_self(), object); }
        if (status != KERN_SUCCESS || address > begin || size == 0 || size > UINTPTR_MAX-address ||
                !(basic.protection & VM_PROT_READ) || (basic.protection & VM_PROT_WRITE)) {
            return false;
        }
        const uintptr_t next = address + size;
        address = begin;
        size = 0;
        vm_region_extended_info_data_t extended;
        count = VM_REGION_EXTENDED_INFO_COUNT;
        object = MACH_PORT_NULL;
        status = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_EXTENDED_INFO,
                                (vm_region_info_t) &extended, &count, &object);
        if (object != MACH_PORT_NULL) { mach_port_deallocate(mach_task_self(), object); }
        if (status != KERN_SUCCESS || address > begin || !extended.external_pager) {
            return false;
        }
        begin = next < end ? next : end;
    }
    return begin == end;
}

static void airllm_expert_drop_after_use(const struct ggml_tensor * src0, const int64_t * counts, int n_experts) {
    if (src0->type != GGML_TYPE_Q2_K || !src0->buffer || !src0->data ||
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            strcmp(ggml_backend_buffer_name(src0->buffer), "CPU_Mapped") != 0 ||
            !ggml_is_contiguous(src0) || src0->ne[3] != 1 || n_experts <= 0 || n_experts > 4096 ||
            src0->ne[2] != n_experts) {
        return;
    }
    const long page_long = sysconf(_SC_PAGESIZE);
    const size_t stride = src0->nb[2];
    if (page_long <= 0 || stride == 0 || stride > (size_t)64*1024*1024 ||
            (size_t)n_experts > ggml_nbytes(src0)/stride) {
        return;
    }
    const uintptr_t page = (uintptr_t)page_long;
    const uintptr_t base = (uintptr_t)src0->data;
    size_t selected = 0;
    for (int expert = 0; expert < n_experts; ++expert) {
        if (counts[expert] < 0) { return; }
        selected += counts[expert] > 0;
    }
    if (selected > ((size_t)128*1024*1024)/stride) { return; }
    size_t advised = 0, bytes = 0, skipped = 0, errors = 0;
    int first_errno = 0;
    for (int expert = 0; expert < n_experts; ++expert) {
        if (counts[expert] == 0) { continue; }
        const size_t offset = (size_t)expert*stride;
        if (offset > UINTPTR_MAX-base || stride > UINTPTR_MAX-base-offset) { ++skipped; continue; }
        const uintptr_t raw_begin = base + offset;
        const uintptr_t raw_end = raw_begin + stride;
        const uintptr_t padding = (page - raw_begin%page)%page;
        if (padding > raw_end-raw_begin) { ++skipped; continue; }
        const uintptr_t begin = raw_begin + padding;
        const uintptr_t end = raw_end - raw_end%page;
        if (begin >= end || !airllm_expert_drop_readonly_file(begin, end)) { ++skipped; continue; }
        if (madvise((void *)begin, end-begin, MADV_DONTNEED) != 0) {
            if (errors++ == 0) { first_errno = errno; }
        } else {
            ++advised;
            bytes += end-begin;
        }
    }
    fprintf(stderr, "AIRLLM_EXPERT_DROP tensor=%s selected=%zu advised=%zu bytes=%zu skipped=%zu errors=%zu first_errno=%d\n",
            src0->name, selected, advised, bytes, skipped, errors, first_errno);
}
#endif
