// MIT. Exercise genuinely asynchronous work without a model or GPU.
#include "completed_call.h"
#include <atomic>
#include <thread>
#include <cstdio>
int main() {
    std::atomic<int> result{0};std::thread pending;
    bool barrier_called=false;
    const double seconds=completed_seconds([&]{
        pending=std::thread([&]{std::this_thread::sleep_for(std::chrono::milliseconds(20));result=42;});
    },[&]{pending.join();barrier_called=true;});
    if(!barrier_called||result!=42||seconds<.015)return 2;
    // A returned error status does not exempt pending backend work from its
    // barrier. The caller must have a complete cost even on a failed call.
    int status=0;barrier_called=false;
    completed_seconds([&]{status=7;},[&]{barrier_called=true;});
    if(status!=7||!barrier_called)return 3;
    std::printf("{\"completed_result\":%d,\"seconds\":%.9f,\"error_barrier_called\":true}\n",int(result.load()),seconds);
    return 0;
}
