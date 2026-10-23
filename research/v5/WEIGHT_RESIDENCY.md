# Exact weight residency experiments

## Problem and hypothesis

Version 4.1 retains decoded matrices. FP8-to-FP32 expansion uses roughly four
times the source payload; BF16-to-FP32 uses twice the source payload. A sequence
of experts or dense layers can therefore exceed a decoded cache even when its
unchanged source bytes fit the same budget. Version 4.1 also deliberately drops
each evaluated output-head tile after use, so a large tiled head is reread on
each subsequent token even when the explicit weight budget has spare capacity.

Two opt-in mechanisms now share the existing budget:

1. Exact source-array residency retains FP8 plus its original scale selections,
   BF16, or floating source selections before decode. `compact_cache_bytes = S`
   partitions the total configured weight-cache budget `C`; decoded capacity is
   `C - S`. There is no extra hidden cache allowance.
2. `output_tile_cache` admits evaluated output tiles into that same decoded
   cache under a key containing the tensor name and exact row interval.

For source bytes `b_i`, decoded bytes `d_i`, and working set `W`, the useful
capacity window for all-source residency is `sum(b_i) <= C < sum(d_i)`. This is
a workload condition, not a universal property of a giant checkpoint. Partial
tiers can duplicate source and decoded data; they can waste capacity and lose.

Source-cache hits avoid file reads but still execute the original decode path.
They are favorable only when the avoided read/copy cost and improved residency
outweigh cache bookkeeping and repeated decoding. Output-tile hits avoid both
reading and decoding that tile, but can displace more useful expert matrices.

## Constraints and implementation

- FP8 payload and scale selections are admitted as one tuple. Scales are never
  inferred or replaced. Row offsets remain part of the exact decode equation.
- NumPy views retain their complete underlying allocations. Admission charges
  unique backing buffers, including bytes outside an individual view; it makes
  no large copy. Unknown external backing allocations fail explicitly.
- Source-cache admission follows a successful complete read. Interrupted reads
  never admit partial selections. A cached selection still requires validated
  integer expert and row selectors; Python equality such as `1 == 1.0` cannot
  bypass the storage validation contract.
- Both tiers are thread-safe and trim under the runtime's observed memory
  pressure. `SingleFlight` still reserves decoded demand/prefetch workspace.
- The source tier has a separate 4,096-entry metadata cap, so arbitrarily tiny
  selections cannot create an unbounded Python dictionary within a byte budget.
- The decoded tile path evaluates the output before releasing its temporary
  reference. Retained tiles remain charged to decoded cache capacity. Measured
  tile load time is recorded for the existing cost policy.
- No future route is assumed; only already requested tensor selections enter
  these caches. No on-disk writes, requantization, or added model downloads occur.

## Numerical verification and unfavorable controls

`tests_next/test_v5_weights.py` covers backing-buffer over-retention, shared
buffers, exact FP8/scales/BF16 values, row-selection separation, read failure and
retry, shrinking budgets, eviction, invalid cached selectors, and complete
reduced GLM logits/tokens compared with the decoded-only path. The low-reuse
control cycles three experts through a one-expert source cache: it produces
zero source hits and the same logical file traffic as ordinary uncached reads.
These reduced weights are numerical fixtures, not trained-model evidence.

## Trained-weight protocol

`benchmarks/benchmark_weight_residency_v5.py` runs a complete locally supplied
trained Llama checkpoint. It does not download weights. Each variant starts in
a fresh process; a second round reverses the variant order. Both paired controls
receive the same total cache budget, prompts, precision, token count, I/O, KV,
workspace, row tile size, and prefill chunk size. EOS is explicitly disabled to
keep the payload equal. All emitted token sequences are compared.

The 300 MiB pair compares decoded-only with all-source residency, targeting the
capacity window above for SmolLM2-135M-Instruct. The 600 MiB pair compares
decoded-only with optional decoded tile residency. These budgets are
experimental conditions, not recommended giant-model defaults.

The report separates the first empty-software-cache generation from repeated
warm generations. It records complete generation time, TTFT, decoded token rate,
logical file bytes, read calls, process physical read counters when available,
system swap growth, tier occupancy, and process-lifetime peak RSS. RSS includes
framework/tokenizer overhead and is not equivalent to the configured cache.

OS page caches are uncontrolled. A small trained checkpoint fits host RAM;
neither this experiment nor its logical byte savings demonstrate a model larger
than system RAM, an external SSD speed, Metal performance, GLM-5.3 throughput,
or long-session thermal stability. A negative wall-time result remains in the
JSON and must not be promoted to an acceleration claim because reads decreased.

## Reproduction

```sh
PYTHONPATH=air_llm python benchmarks/benchmark_weight_residency_v5.py \
  --model-dir /absolute/path/to/local/SmolLM2-135M-Instruct \
  --output evidence-v5/weight-residency-trained.json --tokens 16 --repeats 3
```

Run in an otherwise idle environment. Timing measured while other inference or
regression processes are active is exploratory and requires a clean measurement
window before claiming a speedup. This is an integration and capacity experiment;
it does not assert a new compression family or a universal I/O lower bound.

## Observed trained results, 23 October 2026

Evidence: `evidence-v5/weight-residency-trained.json`, its eight per-worker JSON
files and process logs. Checkpoint SHA-256:
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
All **48 complete generations / 768 emitted tokens** have identical sequences
across variants. The 32 warm timed generations provide eight observations per
variant; the first two generations of each process are excluded from warm
medians. Both prompts are retained, without changing lengths or output counts.

| Fixed budget and mechanism | Control median | Variant median | Logical file bytes per 16-token generation | Maximum process peak RSS |
|---|---:|---:|---:|---:|
| 300 MiB, exact source tier | 8.320 s | 8.631 s | 4,729,352,256 → 0 | 829.2 → 772.4 MB |
| 600 MiB, decoded output tiles | 6.889 s | 6.502 s | 906,027,840 → 58,176 | 939.3 → 1,047.9 MB |

Source residency eliminates the measured warm file requests but is **3.75%
slower** on the median complete generation in this CPU/warm-page-cache workload.
It remains opt-in; reducing logical reads is not counted as acceleration. The
capacity condition is demonstrated, not practical GLM throughput.

Output tiles reduce complete median time by about 5.6% (speedup 1.059×), while
retaining an additional approximately 109 MB at the process peak. This fits the
same configured cache budget, which the control was not fully using. However,
the two order-specific speedup ratios are approximately 1.104 and 0.977; timing
is variable. This supports a modest observed aggregate difference, not a
robust universal speed claim or the mission's 2×/5×/10× ambitions.

The measured byte reductions refer to logical file payload requested by this
process. Linux physical read counters and system swap growth are in each raw
row; they are different quantities. Native MLX allocator peaks were not sampled
by this particular script. The first worker's initial excluded generation may
overlap the last approximately five seconds of a small test run. Other agents
held CPU-heavy tests during warm timing; a separate large-checkpoint network
download proceeded concurrently. No externally attached SSD was available.

The metadata entry cap was added after these runs as a guard against tiny-entry
scans. None of these workers approached the cap; numerical/cache-policy paths
and the fixed-budget comparisons are unchanged. A dedicated regression covers
that cap separately.
