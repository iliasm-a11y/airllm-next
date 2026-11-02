# Exact selected-expert staging and adaptive cache

This research component keeps the original GGUF bytes and the model's original
ordered route IDs. It does not install a production cache or change RONIN's
600 logical agents and single physical model slot. No token-rate gain follows
from the cache experiments below.

## Implementation

`cache_adaptive.py` extends the PR18 byte cache. A selected request reserves and
pins all its exact expert payloads **before** asynchronous reads start. Its
consumer visits `iter_ready()` in original route order and can work on one
expert while later experts are still being read. Duplicate IDs share one fill.
A consumer must synchronize GPU use before releasing the request's leases.

Resident, pinned and in-flight packed bytes share one budget with a hard ceiling
of 128 MiB. Python, driver tensors, file-system cache and other applications are
outside this payload budget and still require the external RSS/swap/pressure
guard. The existing memory-admission callback remains mandatory when enabled.
The cache is disabled by default.

`resize()` can shrink a target while a consumer is active. It evicts completed
unpinned payloads, retains every active reader/lease reservation, reports any
deferred shrink and rejects a new selected request until the target is met.
`adapt_to_memory()` accepts measured available memory and pressure, leaving the
requested headroom and setting the target to zero for warning/critical pressure.
Unknown pressure fails closed. Cancellation drains running readers before
releasing reservations. The requesting consumer must close its active request.

LRU remains the default based on the captured trace result below. SLRU, LFU and
weighted eviction are explicit alternatives. Weighted scores use only observed
successful load duration × observed reuse / packed size. No assumed SSD speed
or future route trace enters admission or eviction. Reuse history is bounded.

Exactly adjacent spans can be combined into one `readinto` in payload order.
It reads no intervening gaps and creates no combined temporary payload. This
reduces three reads to two in the adjacent-range fixture. GLM's gate/up/down
expert banks are separated; this does not imply a coalescing gain for them.
The real one-span gate probe below needs one read per miss either way.

There is no extra spare-buffer pool or unproven double-buffer allocation.
Resident hits reuse their existing packed buffer. A spare pool would need to
remain charged to the same payload ceiling, and allocation cost has not been
shown to justify it. Asynchronous selected fills and ordered consumption are
implemented; overlap speedup has not been isolated or established.

## Captured policy replay: metadata only

`cache_replay policy` read all 42 MoE layers across four captured steps: the
six-token prefill and three decode tokens, 378 top-8 groups and 3,024 expert IDs.
It uses actual GGUF gate/up/down sizes, 8,257,536 bytes per complete expert.
Only metadata-sized row probes are read while identifying the banks. The replay
does not allocate the simulated expert payloads or measure IO/wait latency.

| Packed budget | Policy | Hits | Misses | Evictions | Logical miss bytes |
|---|---|---:|---:|---:|---:|
| 64 MiB | LRU / SLRU / LFU | 558 | 2,466 | 2,458 | 20,363,083,776 |
| 128 MiB | LRU | 704 | 2,320 | 2,304 | 19,157,483,520 |
| 128 MiB | SLRU | 657 | 2,367 | 2,351 | 19,545,587,712 |
| 128 MiB | LFU | 562 | 2,462 | 2,446 | 20,330,053,632 |

The original CPU capture and head-only diagnostic contain the same ordered route
trace, so these are two source captures of one trajectory, not two independent
workloads. LRU has more hits on this trajectory at 128 MiB; changing the default
to SLRU or LFU is not supported. Weighted ranking is deliberately omitted from
metadata-only replay because no per-expert load costs were measured for this
complete trace. These miss bytes are neither physical disk reads nor a decoding
latency model.

Evidence: `evidence/cache-policy-replay.json`.

The separate phase replay, `evidence/cache-policy-phase-replay.json`, preserves
those totals and separates the six-token prefill from three decode tokens.
**All 704 LRU hits at 128 MiB occur during prefill.** Its decode phase has zero
hits and 1,008 misses, totaling 8,323,596,288 logical miss bytes. The 64 MiB
policies and 128 MiB LFU also have zero decode hits. At 128 MiB, SLRU has 15
decode hits and 993 misses (8,199,733,248 logical miss bytes), alongside 642
prefill hits. This small metadata-only difference does not establish a decode
speedup. The LRU default was selected from the mixed prefill/decode trace; its
aggregate hit count must not be presented as evidence of decode reuse.

## Real bounded internal-file pipeline

`cache_replay io --experimental` actually read and hashed complete gate
projections for the first 12 ordered layer-3 decode demands, with a 16 MiB
cache, separately for LRU, SLRU, LFU and measured-cost weighted eviction.
The whole four-policy run read **132,120,576 logical bytes (126 MiB)**, below
its shared 128 MiB read ceiling. Each policy had 0 hits, 12 misses, 6 evictions,
12 read calls and a peak packed reservation of 16,515,072 bytes. All source
hashes matched across policies, and closing returned the reservations to zero.

This prefix provides no evidence of a hit-rate improvement. Waits and timings
are recorded in the raw report, but warm/uncontrolled macOS file cache, fixed
policy order, hash work and background load prevent interpreting their small
differences as an SSD or cache-policy speed gain. Weighted costs are actual
observations for this small prefix only.

Evidence: `evidence/cache-io-pipeline.json`.

The `reuse` mode additionally supports two *actual consecutive* layer-40 decode
route records. They share six experts; the mode verifies identical resident
buffer reuse and hashes, preserves both original top-8 orders, and requests a
zero-byte target while eight leases remain pinned. Its report distinguishes the
deferred shrink from actual reclamation after release. The 32 MiB payload and
64 MiB logical-read ceilings remain independent from process RSS. This mode
does not execute expert arithmetic or model generation.

The completed internal-file reuse run read **27,525,120 logical bytes
(26.25 MiB)** for ten distinct experts. The second recorded token produced
six hits and reused the same six resident byte buffers, with unchanged hashes;
its cumulative cache counters were 6 hits, 10 misses and 10 read calls. Peak
packed reservations were 27,525,120 bytes. Shrinking to zero while the second
token still held eight leases correctly reported a deferred 22,020,096-byte
reservation; closing those leases reclaimed it, and final reservations and
effective capacity were both zero. This establishes the tested byte ownership
and lifecycle behavior, not full-model overlap or a token-rate improvement.

Evidence: `evidence/cache-two-token-reuse.json`. It includes selected source
hashes, source-file stat scope, source-code hashes and both actual route orders.
This isolated layer-40 test omits the other 41 MoE layers between token visits;
its six resident hits do not establish that a 32/128 MiB cache can retain those
experts across a complete model step. The whole-trace phase results above show
that distinction directly.

## Reproduce

From the repository root, using a Python environment with NumPy:

```sh
python -m research.glm53_gpu_v2.cache_replay policy \
  --gguf /absolute/path/to/expert-containing-shard.gguf \
  --capture /absolute/path/to/complete-native-capture \
  --output /absolute/path/to/new-policy-report.json

python -m research.glm53_gpu_v2.cache_replay reuse --experimental \
  --gguf /absolute/path/to/expert-containing-shard.gguf \
  --capture /absolute/path/to/complete-native-capture \
  --output /absolute/path/to/new-reuse-report.json

python -m unittest research.glm53_gpu_v2.test_cache_adaptive \
  research.glm53_gpu_v2.test_cache_replay -v
```

Real-read modes reject existing output files and accept one capture per run so
their explicit read ceiling covers the entire invocation. No model or GPU
executes. No model or SSD contents are written. At inspection, `/Volumes/SSD`
was a directory on the internal root filesystem, not a mounted Fanxiang volume;
none of these measurements qualifies the external SSD's throughput.
