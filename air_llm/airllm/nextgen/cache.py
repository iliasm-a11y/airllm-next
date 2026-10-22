"""Byte-bounded caches with bounded protected segment and single-flight IO."""
from collections import OrderedDict, Counter, defaultdict,deque
from concurrent.futures import ThreadPoolExecutor, Future
from threading import RLock
from time import perf_counter


def size_of(value):
    if isinstance(value,dict):return sum(size_of(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(size_of(v) for v in value)
    return int(value.nbytes)


class WeightCache:
    def __init__(self,capacity,policy='slru'):
        if policy not in ('lru','slru','lfu','cost'):raise ValueError('Unknown cache policy')
        self.capacity=max(0,int(capacity));self.policy=policy
        self.data=OrderedDict();self.hot=OrderedDict();self.sizes={};self.freq=Counter();self.trace=deque(maxlen=4096)
        self.bytes=0;self.peak_bytes=0;self.hot_bytes=0;self.lock=RLock();self.stats=Counter()
        # GreedyDual-size-frequency keeps expensive transfers/conversions whose
        # measured reuse benefit per byte is largest. No synthetic SSD bandwidth
        # assumption enters the score. Existing policies remain the default.
        self.costs={};self.priorities={};self.inflation=0.
        self.cost_trace=deque(maxlen=4096);self.load_trace=deque(maxlen=4096)

    def record_load(self,key,seconds):
        """Observe a successful demand/prefetch load, including conversion time."""
        seconds=float(seconds)
        if not 0<=seconds<float('inf'):raise ValueError('Load time must be finite and nonnegative')
        with self.lock:
            previous=self.costs.get(key)
            self.costs[key]=seconds if previous is None else previous*.75+seconds*.25
            self.load_trace.append((key,seconds))

    def _priority(self,key,n):
        # Missing observations get zero priority until their first load succeeds.
        return self.inflation+self.costs.get(key,0.)*self.freq[key]/max(1,n)

    def get(self,key,size_hint=None):
        with self.lock:
            if size_hint is not None:
                self.trace.append((key,int(size_hint)))
                self.cost_trace.append((key,int(size_hint),self.costs.get(key)))
            self.freq[key]+=1
            if sum(self.stats.values())%8192==8191:
                self.freq=Counter({k:max(1,v//2) for k,v in self.freq.items() if k in self.sizes})
                self.costs={k:v for k,v in self.costs.items() if k in self.sizes}
            if key not in self.data:
                self.stats['misses']+=1;return None
            self.stats['hits']+=1;self.data.move_to_end(key)
            if self.policy=='cost':self.priorities[key]=self._priority(key,self.sizes[key])
            if self.policy=='slru':
                if key not in self.hot:self.hot_bytes+=self.sizes[key]
                self.hot[key]=None;self.hot.move_to_end(key);self._demote()
            return self.data[key]

    def _demote(self):
        while self.hot and self.hot_bytes > int(self.capacity*.75):
            k,_=self.hot.popitem(last=False);self.hot_bytes-=self.sizes[k]

    def contains(self,key):
        with self.lock:return key in self.data

    def _evict(self):
        if self.policy=='cost':
            k=min(self.data,key=lambda k:self.priorities.get(k,0.))
            self.inflation=self.priorities.get(k,self.inflation)
        elif self.policy=='lfu':k=min(self.data,key=lambda k:self.freq[k])
        elif self.policy=='slru':k=next((k for k in self.data if k not in self.hot),next(iter(self.data)))
        else:k=next(iter(self.data))
        self.data.pop(k)
        self.priorities.pop(k,None)
        if k in self.hot:self.hot_bytes-=self.sizes[k];self.hot.pop(k)
        self.bytes-=self.sizes.pop(k);self.stats['evictions']+=1

    def put(self,key,value):
        n=size_of(value)
        with self.lock:
            if key in self.data or n>self.capacity or self.capacity==0:return False
            # LFU admission can reject a one-off scan rather than evict a hot entry.
            if self.policy=='lfu' and self.bytes+n>self.capacity and self.data:
                if self.freq[key] < min(self.freq[k] for k in self.data):
                    self.stats['rejected']+=1;return False
            while self.bytes+n>self.capacity:self._evict()
            self.data[key]=value;self.sizes[key]=n;self.bytes+=n
            if self.policy=='cost':self.priorities[key]=self._priority(key,n)
            self.peak_bytes=max(self.peak_bytes,self.bytes);return True

    def resize(self,capacity):
        with self.lock:
            self.capacity=max(0,int(capacity))
            while self.bytes>self.capacity:self._evict()
            self._demote()

    def clear(self):
        with self.lock:
            self.data.clear();self.hot.clear();self.sizes.clear();self.priorities.clear()
            self.bytes=0;self.hot_bytes=0;self.inflation=0.


class SingleFlight:
    """Pending + completed-but-unconsumed futures hold their byte reservations.

    Demand requests deduplicate speculative requests. No nested worker submits
    are allowed. Inference consumes in route order; stale prefetch is explicitly
    drained on pressure, errors and shutdown.
    """
    def __init__(self,budget,workers=2):
        if not isinstance(budget,int) or isinstance(budget,bool) or budget<0:
            raise ValueError('IO budget must be a nonnegative integer')
        if not isinstance(workers,int) or isinstance(workers,bool) or workers<1:
            raise ValueError('IO workers must be a positive integer')
        self.budget=budget;self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='airllm-offset')
        self.pending={};self.bytes=0;self.peak_bytes=0;self.lock=RLock();self.closed=False;self.stats=Counter()

    def submit(self,key,size,loader):
        if not isinstance(size,int) or isinstance(size,bool) or size<0:
            raise ValueError('IO reservation must be a nonnegative integer')
        with self.lock:
            if self.closed:raise RuntimeError('IO scheduler is closed')
            if key in self.pending:self.stats['deduplicated']+=1;return True
            if size>self.budget or self.bytes+size>self.budget:return False
            self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
            def run():
                value=loader()
                if size_of(value)>size:raise MemoryError('IO result exceeded its byte reservation')
                return value
            try:future=self.pool.submit(run)
            except BaseException:
                self.bytes-=size;raise
            self.pending[key]=(future,size);self.stats['submitted']+=1;return True

    def take(self,key):
        with self.lock:item=self.pending.get(key)
        if item is None:return None
        start=perf_counter()
        try:return item[0].result()
        finally:
            with self.lock:
                # A completed Future may be consumed by concurrent waiters, but
                # exactly one waiter releases its shared reservation.
                if self.pending.get(key) is item:
                    self.pending.pop(key);self.bytes-=item[1];self.stats['consumed']+=1
                self.stats['wait_seconds']+=perf_counter()-start

    def drain(self):
        with self.lock:items=list(self.pending.items())
        error=None
        for key,expected in items:
            future,_=expected
            future.cancel()
            try:
                if not future.cancelled():future.result()
            except BaseException as e:
                if error is None:error=e
            finally:
                with self.lock:
                    # take() may consume this key while drain waits. A later
                    # producer can reuse the key; its reservation belongs to a
                    # different future and must not be released by this drain.
                    if self.pending.get(key) is expected:
                        self.pending.pop(key);self.bytes-=expected[1];self.stats['discarded']+=1
        if error is not None:raise error

    def close(self):
        # Stop admission before waiting for pending readers. Otherwise a producer
        # can enqueue after drain's snapshot and leak an unconsumed reservation.
        with self.lock:
            if self.closed:return
            self.closed=True
        try:self.drain()
        finally:self.pool.shutdown(wait=True,cancel_futures=True)
