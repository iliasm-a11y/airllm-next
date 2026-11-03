// MIT. Diagnostic phase counters. OS disk accounting is never labelled NAND.
struct timing_sample {
    std::chrono::steady_clock::time_point stamp;
    rusage_info_v2 os{};
    struct rusage faults{};
    bool os_ok=false,faults_ok=false;
};
static timing_sample timing_snapshot(){
    timing_sample s;
    s.os_ok=proc_pid_rusage(getpid(),RUSAGE_INFO_V2,reinterpret_cast<rusage_info_t*>(&s.os))==0;
    s.faults_ok=getrusage(RUSAGE_SELF,&s.faults)==0;
    s.stamp=std::chrono::steady_clock::now();return s;
}
static std::string timing_delta(const char* phase,int step,const timing_sample& a,
                                 const timing_sample& b,double call_seconds=-1){
    std::ostringstream out;out << std::setprecision(17);
    out << "{\"phase\":\"" << phase << "\",\"step\":" << step
        << ",\"wall_seconds\":" << std::chrono::duration<double>(b.stamp-a.stamp).count()
        << ",\"decode_call_seconds\":";
    if(call_seconds>=0)out << call_seconds;else out << "null";
    out << ",\"counters_available\":" << (a.os_ok&&b.os_ok?"true":"false");
    if(a.os_ok&&b.os_ok){
        mach_timebase_info_data_t scale{};mach_timebase_info(&scale);
        const double seconds=double(scale.numer)/scale.denom/1e9;
        out << ",\"cpu_core_seconds\":" << (b.os.ri_user_time-a.os.ri_user_time+b.os.ri_system_time-a.os.ri_system_time)*seconds
            << ",\"os_read_bytes\":" << b.os.ri_diskio_bytesread-a.os.ri_diskio_bytesread
            << ",\"os_write_bytes\":" << b.os.ri_diskio_byteswritten-a.os.ri_diskio_byteswritten
            << ",\"pageins\":" << b.os.ri_pageins-a.os.ri_pageins
            << ",\"resident_after_bytes\":" << b.os.ri_resident_size
            << ",\"footprint_after_bytes\":" << b.os.ri_phys_footprint;
    }
    if(a.faults_ok&&b.faults_ok)out << ",\"minor_faults\":" << b.faults.ru_minflt-a.faults.ru_minflt
        << ",\"major_faults\":" << b.faults.ru_majflt-a.faults.ru_majflt;
    out << ",\"physical_nand_bytes\":null,\"allocation_bytes\":null,\"pure_io_wait_seconds\":null}";
    return out.str();
}
