# Exact selected-expert cache experiment

`expert_cache.py` is an opt-in research component. It is not installed into the
native RONIN backend, and enabling it is not evidence of an inference speedup.
The existing native `llama_moe_cache` already caches packed experts in bounded
GPU banks. This experiment supplies a separately testable **host-byte** staging
cache and asynchronous exact reads for the Metal/MLX kernel probe.

Pass byte ranges from validated GGUF metadata only after the model's own router
has selected its experts. The cache does not compute, reorder or predict routes.
`acquire_selected` returns leases in the caller's original selection order;
duplicate IDs reuse one fill. Each expert may include three spans for its
gate/up/down projections. Payloads contain the original packed bytes.

```python
from expert_cache import ByteSpan, ExpertRef, ExpertByteCache

# Offsets/lengths must come from the shard's validated tensor layout.
refs = [ExpertRef((layer, expert_id), tuple(validated_spans))]
with ExpertByteCache(shard, capacity_bytes=128 * 1024**2,
                     max_inflight_bytes=64 * 1024**2,
                     enabled=True, admission=current_memory_admission) as cache:
    cache.prefetch_selected(refs)
    # Independent work may proceed here while readinto fills the packed bytes.
    with cache.acquire_selected(refs) as leases:
        packed_gate, packed_up, packed_down = leases[0].parts()
        # Submit/evaluate a packed-byte GPU kernel here.
        # Synchronize its completion BEFORE releasing the lease.
```

These budgets are examples, not a measured optimum. The payload reservation
includes queued/running fills, completed residents and pinned leases together.
Allocation starts only after reservation and external admission. Another
in-flight limit bounds queued/running reads. Completed unpinned entries can be
evicted; a selected pinned set cannot. If the selected set, existing pins or
pending fills prevent admission, the caller receives `CacheBudgetExceeded`.
There is no hidden fallback buffer or relaxed memory threshold.

The budget excludes Python metadata, file-system page cache, CPU/GPU tensors and
device copies. The required external admission callback must apply the owning
application's RSS, swap and pressure guards to those additional costs. Keep the
model slot at one; this cache does not alter the 600 logical RONIN identities.
`read_seconds` and `logical_bytes_read` describe these explicit reads. They are
not physical SSD throughput or NAND traffic, and must not be used as such.

Cancellation waits for a running read to finish before releasing its byte
reservation. Closing rejects new requests, drains readers and preserves existing
pinned leases until release. Read-only memoryviews are valid only while their
lease remains active; retaining them after release violates the ownership
contract. An asynchronous GPU consumer must finish before lease release.
Read errors and source-file identity/size/modification changes fail closed.
The stat check is not a cryptographic re-verification of the entire model.

Portable tests exercise byte fidelity, exact range/order/duplicate handling,
combined reservations, in-flight limits, pinning, admission rejection, eviction,
source truncation, cancellation and shutdown:

```sh
python3 -m unittest discover -s research/glm53_gpu -p 'test_expert_cache.py' -v
```

These small fixtures do not qualify full-model fidelity, sustained memory use,
cache hit rates on GLM or decoding performance on the M5/Fanxiang combination.

## Captured route / Metal projection integration

`stream_selected.py` connects this cache to `Q2KMetalMatvec`. Execution requires
`--experimental`; invoke it through the independent process/swap watchdog. The
default selection is layer 3, decode step 1, its original eight ordered expert
IDs. Prefill captures may select one explicit `--column`. `--projection gate`
is the default; `up` and `down` can be tested separately. The GGUF header parser
reads one row to locate the bank and then the cache reads complete selected
expert projections, preserving their packed bytes.

```sh
python3 -m research.glm53_gpu.stream_selected --experimental \
  --gguf /absolute/path/to/model-shard-00002-of-00002.gguf \
  --capture /absolute/path/to/native-cpu-capture \
  --libraries /absolute/path/to/pinned-native-libraries \
  --output /absolute/path/to/projection-replay.json
```

Two passes exercise asynchronous exact fills followed by cached reuse. At most
64 MiB is reserved for packed host cache payloads and 16 MiB for each prepared
matrix's known payloads. Only one expert matrix is submitted to the GPU at a
time. The MLX allocation guideline is 128 MiB; an additional local pressure and
available-memory admission check fails closed. The external process/swap guard
is still required, because these bounds exclude runtime/compiler overhead.

The report records original route IDs, exact source ranges and hashes, native
and GPU result hashes, bitwise comparisons, both pass timings and cache counters.
The activation is seeded **synthetic** FP32 data. It does not replay the actual
model hidden states, execute its router, combine gate/up/down, or perform MoE
reduction. A passing report qualifies these projection arithmetic calls only;
neither full GLM fidelity nor a decoding speedup follows from it. CPU timing is
one row dot per ctypes call with one caller, not the CPU10 model reference.
