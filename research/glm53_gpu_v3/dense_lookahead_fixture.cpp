// Actual GGML CPU_Mapped buffers; bounded advice/lifecycle fixture, no model/GPU.
#include "dense_lookahead.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <atomic>
#include <cerrno>
#include <fcntl.h>

struct probe_state {
    std::mutex mutex;std::condition_variable cv;
    bool entered=false,release=false,block=false;
    int result=0;
};
static bool yes(size_t,void *) {return true;}
static bool no(size_t,void *) {return false;}
static int injected(void *,size_t,void * data) {
    auto & s=*static_cast<probe_state *>(data);std::unique_lock<std::mutex> lock(s.mutex);
    s.entered=true;s.cv.notify_all();
    if(s.block)s.cv.wait(lock,[&]{return s.release;});
    return s.result;
}
static bool entered(probe_state & s) {
    std::unique_lock<std::mutex> lock(s.mutex);
    return s.cv.wait_for(lock,std::chrono::seconds(2),[&]{return s.entered;});
}
int main() {
    const size_t page=size_t(sysconf(_SC_PAGESIZE)),length=page*8+32768;
    char name[]="/tmp/airllm-dense-lookahead-XXXXXX";const int fd=mkstemp(name);
    if(fd<0||ftruncate(fd,length))return 2;
    float original[64];for(int i=0;i<64;++i)original[i]=float(i%11-5)*0.03125f;
    if(pwrite(fd,original,sizeof(original),0)!=sizeof(original)||pwrite(fd,original,sizeof(original),page)!=sizeof(original))return 3;
    void * mapping=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);unlink(name);close(fd);
    if(mapping==MAP_FAILED)return 4;
    auto buffer=ggml_backend_cpu_buffer_from_ptr(mapping,length);
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_init_params params{64<<10,nullptr,true};auto ctx=ggml_init(params);
    auto w0=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2),w1=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2);
    w0->data=mapping;w0->buffer=buffer;w1->data=static_cast<char *>(mapping)+page;w1->buffer=buffer;
    alignas(64) float input[32];for(int i=0;i<32;++i)input[i]=float(i%5-2)*0.0625f;
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,1);
    auto xb=ggml_backend_cpu_buffer_from_ptr(input,sizeof(input));x->data=input;x->buffer=xb;ggml_set_input(x);
    auto y0=ggml_mul_mat(ctx,w0,x),y1=ggml_mul_mat(ctx,w1,x),out=ggml_add(ctx,y0,y1);
    auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);
    auto cpu=ggml_backend_cpu_init();auto allocation=ggml_backend_alloc_ctx_tensors(ctx,cpu);
    int checks=0;auto check=[&](bool ok){++checks;if(!ok){std::fprintf(stderr,"failed check %d\n",checks);std::exit(6);}};
    airllm_dense::entry window{};
    check(airllm_dense::window_for(w0,page,window)&&window.bytes<=page);
    check(!airllm_dense::window_for(w0,airllm_dense::max_window+1,window));
    check(!airllm_dense::window_for(x,page,window));
    ggml_tensor bad=*w0;bad.data=static_cast<char *>(mapping)+length;
    check(!airllm_dense::window_for(&bad,page,window));
    bad=*w0;bad.nb[1]+=4;check(!airllm_dense::window_for(&bad,page,window));
    auto wide=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,64);wide->data=static_cast<char *>(mapping)+2*page;wide->buffer=buffer;
    check(airllm_dense::window_for(wide,page,window)&&window.bytes==page);
    w0->flags=GGML_TENSOR_FLAG_INPUT;check(!airllm_dense::window_for(w0,page,window));w0->flags=0;
    ggml_tensor aliases[17];for(int i=0;i<17;++i){aliases[i]=*w0;aliases[i].op=GGML_OP_VIEW;aliases[i].view_src=i?&aliases[i-1]:w0;}
    w0->flags=GGML_TENSOR_FLAG_COMPUTE;check(!airllm_dense::window_for(&aliases[15],page,window));w0->flags=0;
    check(!airllm_dense::window_for(&aliases[16],page,window));
    void * writable=mmap(nullptr,page,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    auto wb=ggml_backend_cpu_buffer_from_ptr(writable,page);ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bad=*w0;bad.data=writable;bad.buffer=wb;check(!airllm_dense::window_for(&bad,page,window));
    unsetenv("AIRLLM_DENSE_LOOKAHEAD");
    {airllm_dense::scope disabled(graph);check(!disabled.enabled());}
    setenv("AIRLLM_DENSE_LOOKAHEAD","1",1);
    {
        probe_state state;
        airllm_dense::scope first(graph,{yes,injected,&state},page);check(first.enabled());
        airllm_dense::scope concurrent(graph,{yes,injected,&state},page);check(!concurrent.enabled());
        first.advance(0);check(entered(state));first.cancel_and_join();
        const auto s=first.snapshot();check(s.calls==1&&s.advised_bytes<=page&&s.failures==0);
    }
    {
        airllm_dense::scope denied(graph,{no,nullptr,nullptr},page);denied.advance(0);
        denied.cancel_and_join();const auto s=denied.snapshot();check(s.calls==0&&s.admission_skips==1);
    }
    {
        probe_state state;state.result=EIO;
        airllm_dense::scope failed(graph,{yes,injected,&state},page);failed.advance(0);
        check(entered(state));failed.cancel_and_join();const auto s=failed.snapshot();check(s.calls==1&&s.failures==1&&s.first_error==EIO);
    }
    {
        probe_state state;state.block=true;
        airllm_dense::scope slow(graph,{yes,injected,&state},page);slow.advance(0);check(entered(state));
        std::atomic<bool> stopped{false};std::thread stopper([&]{slow.cancel_and_join();stopped=true;state.cv.notify_all();});
        {std::unique_lock<std::mutex> lock(state.mutex);
         check(!state.cv.wait_for(lock,std::chrono::milliseconds(20),[&]{return stopped.load();}));}
        airllm_dense::scope concurrent(graph,{yes,injected,&state},page);check(!concurrent.enabled());
        {std::lock_guard<std::mutex> lock(state.mutex);state.release=true;state.cv.notify_all();}
        stopper.join();check(stopped&&slow.snapshot().calls==1);
    }
    int real_result=0;
    {
        airllm_dense::scope real(graph,{yes,nullptr,nullptr},page);real.advance(0);
        // Drain a genuine bounded advice call, if it started before cancellation.
        for(int i=0;i<100&&real.snapshot().calls==0;++i)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        real.cancel_and_join();const auto s=real.snapshot();check(s.calls==1&&s.advised_bytes<=page);real_result=s.first_error;
    }
    unsetenv("AIRLLM_DENSE_LOOKAHEAD");
    check(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS);
    float actual[2];ggml_backend_tensor_get(out,actual,0,sizeof(actual));
    // Compare the flagged native path to the unflagged dispatch on identical bytes.
    // Admission may deny native advice; the separate component call above proves
    // a real bounded syscall, not full-engine overlap or a token-rate gain.
    setenv("AIRLLM_DENSE_LOOKAHEAD","1",1);
    check(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS);
    unsetenv("AIRLLM_DENSE_LOOKAHEAD");
    float expected[2];ggml_backend_tensor_get(out,expected,0,sizeof(expected));check(!std::memcmp(actual,expected,sizeof(actual)));
    std::printf("{\"status\":\"passed\",\"checks\":%d,\"real_advice_result\":%d,\"max_fixture_advice_bytes\":%zu,\"cpu_outputs_bit_identical\":true,\"gpu_initialized\":false,\"model_loaded\":false}\n",checks,real_result,page);
    ggml_backend_buffer_free(allocation);ggml_backend_buffer_free(xb);ggml_backend_buffer_free(wb);ggml_backend_buffer_free(buffer);
    ggml_free(ctx);ggml_backend_free(cpu);munmap(writable,page);munmap(mapping,length);return 0;
}
