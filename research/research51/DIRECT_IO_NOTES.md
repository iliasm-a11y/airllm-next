# Direct packed transport: observed results and reproduction

All observations below are local synthetic experiments on the current Mac.
They use generated GGUF payloads, never model weights. They do not certify
external-SSD cold reads, physical storage bandwidth, generation speed, or
full-model numerical parity.

## Implementation

`air_llm/airllm/nextgen/gguf_packed_io.py` implements
`read_packed_rows_into(reader, name, start, count, *, expert=None, owner_budget=None)`.
It reads a validated selection into its final writable NumPy allocation with
`os.preadv` when available, otherwise macOS/Linux 64-bit `libc.pread`.
No intermediate payload bytes or host packing copy is allocated. Existing
descriptor cache policy, selected bytes and arithmetic remain unchanged.

The reader and optional owner budgets are checked before allocation. The
finalizer belongs to the allocation owner, is registered immediately after
allocation, and takes over accounting before any fallible reshape or helper.
The complete allocation therefore remains charged through ndarray views,
memoryviews, DLPack and retained exception tracebacks. The reader descriptor
lock covers the read and prevents a concurrent close. Partial reads are
continued within the selected region; EOF fails. EINTR/EAGAIN retries are
bounded to three per selection. There is no silent copying backend fallback.

The existing root integration keeps this as an explicit opt-in transport and
offers the writable result to the existing optional DLPack importer. Import
success remains platform-dependent and its fallback is reported separately.

## CPU transport measurement

Original raw result: `work/direct-packed-cpu-microbenchmark.json`.
Each operation reads 1,179,648 encoded bytes under normal, already-warm OS cache
policy. There are 4 warmups per method followed by 51 repetitions with method
order randomized in every repetition. Every method has identical SHA-256 output.

| Transport | Median | Single-read tracemalloc peak |
|---|---:|---:|
| Existing two-host-copy load | 55.708 microseconds | 2,360,825 B |
| Readonly copyfree load | 26.042 microseconds | 1,182,015 B |
| Direct writable load | 26.334 microseconds | 1,181,620 B |

Direct is approximately the same cost as the readonly path in this measurement;
it is not established to be faster than readonly. It preserves the writable
buffer needed by the existing DLPack import while eliminating the two explicit
host copies of the baseline. Tracemalloc is an allocation observation, not an
RSS or device-memory ceiling. Timing varies with platform and cache state.

## Metal exporter lifetime measurement

Original raw result: `work/direct-packed-metal-lifetime.json`.
Each of 100 independent packed matmul calls uses an 1,152-byte synthetic buffer.
All result arrays are retained during the loop, then discarded. Lifetime is
sampled separately after eval, deleting outputs, synchronization, Python GC and
clearing the MLX cache.

* Readonly plus `mx.array`: host budget is zero after every sampled evaluated call.
* Direct plus `mx.array`: host budget is zero after every sampled evaluated call.
* Direct plus DLPack: all 100 imports succeed; only the final 1,152-byte exporter
  remains after the last eval and deletion of outputs. The observed peak is
  2,304 bytes. `mx.synchronize()` releases that last exporter and returns the
  budget to zero. Subsequent GC and cache clearing add no further host release.

This does not reproduce systematic host-buffer retention by MLX. The separate
full-model copyfree failure was subsequently traced by the audit agent to a
self-referential recursive closure in adaptive-cache retained-size accounting;
that cause and its fix belong to the root integration, not to this microbench.
The earlier MLX-retention hypothesis is therefore superseded by the confirmed
Python closure diagnosis.

## Reproduction

From the repository root, using the existing validation environment:

```sh
PYTHON=/Users/macbook/Documents/Codex/2026-10-22/files-pasted-by-the-user-mission/work/validation-venv/bin/python
env PYTHONPATH=air_llm "$PYTHON" benchmarks/research51_microbench.py cpu --output /absolute/path/cpu.json
env PYTHONPATH=air_llm "$PYTHON" benchmarks/research51_microbench.py metal-lifetime --output /absolute/path/lifetime.json
env PYTHONPATH=air_llm "$PYTHON" -m pytest -q tests_next/test_gguf_packed_io.py
```

The script independently writes the synthetic GGUF fixtures in temporary
directories and saves individual timings, allocation peaks, hashes and budget
stages. The saved script has the same experiment parameters and operations as
the original here-document runs. The finalizer-safety patch changes accounting
transfer timing without changing normal-path payload content. No additional
timing run was made after that safety patch.

Latest targeted verification: **22 tests passed in 0.46 s**, covering exact bytes
for three stored types, writable owned storage, budget refusal before payload
allocation, view lifetime, short reads, EINTR, EOF/EIO, retained failure
tracebacks with and without separate owner budgets, and concurrent close.
