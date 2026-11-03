// MIT. Time completed backend work, including an explicit synchronization.
#pragma once
#include <chrono>
template<class Work,class Barrier>
static double completed_seconds(Work&& work,Barrier&& barrier) {
    const auto begin=std::chrono::steady_clock::now();
    work();
    barrier();
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
}
