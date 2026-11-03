#include <mach/mach.h>
#include <mach/task_info.h>
#include <cstddef>
static std::string self_vm_event(int step) {
    task_vm_info_data_t value{};
    mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    const kern_return_t status=task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&value,&count);
    const bool enough=status==KERN_SUCCESS && size_t(count)*sizeof(natural_t)>=offsetof(task_vm_info_data_t,phys_footprint)+sizeof(value.phys_footprint);
    std::ostringstream out;
    out << "{\"step\":" << step << ",\"available\":" << (enough?"true":"false") << ",\"kern_return\":" << status;
    out << ",\"returned_words\":" << count;
    if(enough) out << ",\"resident_bytes\":" << value.resident_size << ",\"physical_footprint_bytes\":" << value.phys_footprint
        << ",\"compressed_bytes\":" << value.compressed << ",\"internal_bytes\":" << value.internal << ",\"external_bytes\":" << value.external
        << ",\"device_category_bytes\":" << value.device << ",\"reusable_bytes\":" << value.reusable;
    out << ",\"scope\":\"native process TASK_VM_INFO; overlapping categories, device category is not exhaustive GPU pinning\"}";
    return out.str();
}
static std::string lifecycle_event(const std::string & label) {
    rusage_info_v2 usage{};
    const bool available=proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t *>(&usage))==0;
    std::ostringstream out;out << std::setprecision(17);
    out << "{\"phase\":\"" << label << "\",\"monotonic_seconds\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
        << ",\"process_counters_available\":" << (available?"true":"false");
    if(available) out << ",\"os_read_bytes\":" << usage.ri_diskio_bytesread << ",\"pageins\":" << usage.ri_pageins
        << ",\"resident_bytes\":" << usage.ri_resident_size << ",\"physical_footprint_bytes\":" << usage.ri_phys_footprint
        << ",\"user_ticks\":" << usage.ri_user_time << ",\"system_ticks\":" << usage.ri_system_time;
    out << ",\"self_vm\":" << self_vm_event(-1) << "}";return out.str();
}
