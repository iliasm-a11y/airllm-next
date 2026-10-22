# Installation and validation on Apple Silicon

The delivered Linux results do not validate Metal, an M5, or the external SSD.
This protocol runs the code locally without inference APIs or implicit model
downloads. A full-model claim requires real checkpoint generation and quality
validation, beyond successful metadata preparation or random miniature tests.

## Install

Use a native arm64 Python environment on macOS, rather than Rosetta. From the
repository root:

```bash
bash scripts/setup_next_macos.sh
source .venv-next/bin/activate
python -m airllm.nextgen.cli doctor
```

The setup script installs the project's pinned requirements. Check the reported
MLX version, chip, memory and `metal_available`; record the same environment for
before/after comparisons. Apple's [M5 and MLX research article](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
states that MLX uses Metal 4 TensorOps and Metal Performance Primitives for M5
GPU Neural Accelerators, and requires macOS 26.2 or later for their enhanced
performance. The script records whether the OS version meets that condition.
This backend uses MLX's supported operations; it does not add a custom TensorOps
kernel or claim that every operation dispatches to a Neural Accelerator.

Apple's [M5 MacBook Air specifications](https://support.apple.com/en-us/126320)
list 153 GB/s unified-memory bandwidth and Thunderbolt 4 / USB 4 ports rated up
to 40 Gb/s. Unified-memory bandwidth is not external-storage bandwidth: 40 Gb/s
is a 5 GB/s link ceiling before overhead, and the actual SSD/enclosure/cable path
can be slower. Use the measured file-service rate below for transfer estimates.

## Run the bounded, read-only I/O comparison

```bash
python benchmarks/benchmark_ooc.py --output evidence-ooc/io-warm.json
python benchmarks/benchmark_ooc.py --cache-state cold-hint --output evidence-ooc/io-cold-hint.json
```

The default input is an actual deterministic 16 MiB safetensors file. It is an
I/O microbenchmark, with 4 MiB maximum read spans, rather than an inference test.
It compares separate reads and coalesced reads with both `pread` and `mmap`,
alternates their order using a recorded seed, and verifies every returned payload
using SHA-256. `read_seconds` excludes checksum computation; complete wall time
is also retained. The result includes every repetition, requested and read
bytes, call counts, physical counters when available, source fingerprint and
process-lifetime RSS peak.

To compare an immutable baseline `storage.py`, use
`--baseline-storage /path/to/original/air_llm/airllm/nextgen/storage.py`.
The baseline module is loaded explicitly and its SHA-256 is recorded. This only
compares the storage path; it does not establish end-to-end model speedup.

Warm mode explicitly touches the same selected tensors before each candidate.
Uncontrolled mode does not flush the OS page cache. A cold hint uses
`POSIX_FADV_DONTNEED` on platforms that expose it, and always states that eviction
is advisory. None is labelled verified cold. In particular, a large logical
throughput on warm data can be RAM bandwidth. On macOS, the Linux process
physical-read counter is unavailable and is JSON `null`.

For a real checkpoint, add `--checkpoint /Volumes/SSD/model`. The script only
reads tensors fitting the bounded `--max-mib` budget and never downloads weights.
Avoid interpreting these selected tensors as the complete model's working set.

## Characterize the actual external SSD

Use an existing, large weight shard on the SSD. Nothing is written to this file:

```bash
python benchmarks/benchmark_ooc.py --probe-file /Volumes/SSD/model/model-00001-of-00096.safetensors --probe-seconds 60 --probe-max-mib 65536 --output evidence-ooc/ssd-sequential.json
python benchmarks/benchmark_ooc.py --probe-file /Volumes/SSD/model/model-00001-of-00096.safetensors --probe-seconds 60 --probe-max-mib 65536 --probe-random --output evidence-ooc/ssd-random.json
```

The read-only probe requests Darwin `F_NOCACHE` on the opened file descriptor.
This is a public macOS file control defined by Apple's [XNU fcntl header](https://github.com/apple/darwin-xnu/blob/main/bsd/sys/fcntl.h).
Success or failure is recorded. It reduces host page-cache influence; it does
not certify physical NAND bandwidth or bypass SSD-controller caches. Read size,
duration, byte cap, sequential/random pattern and memory/swap samples are saved.
The default probe cap is only 512 MiB; increase it explicitly for sustained
measurements. Large files provide better evidence than repeatedly reading a
small file that fits entirely in cache.

Use the measured sequential and expert-sized random file-service rates. A
USB-C connector alone says nothing about the negotiated transport rate. Record
the enclosure, cable, drive filesystem, available space, power mode and whether
the drive shares a bus with another workload alongside the JSON. Do not assume
Thunderbolt bandwidth from connector appearance.

## Run actual Metal and prolonged validation

```bash
bash scripts/validate_ooc_macos.sh
bash scripts/validate_ooc_macos.sh --qwen-miniature
bash scripts/validate_ooc_macos.sh --model /Volumes/SSD/GLM-5.3 --ssd-file /Volumes/SSD/GLM-5.3/model-00001-of-00096.safetensors --seconds 900
```

Replace the example shard path with an existing file. The script verifies that
the host is arm64 macOS with working Metal. It runs the reference regression
suite, which includes CPU tests, then separately invokes
`benchmark_next.py --device gpu` for actual miniature GLM Metal numerical checks
and timings. The regression-suite result must not be presented as entirely GPU
validated. It then optionally performs real-checkpoint generation and repeated
generation for the requested duration. The real run
saves actual output text and runtime telemetry separately. A single generation
may exceed the duration target: the existing sustained runner finishes the
current request before stopping. Choose a small token count for extremely slow
checkpoints. The script uses offline mode and does not download any checkpoint.

The optional `--qwen-miniature` flag separately executes
`validate_qwen_runtime.py --device gpu`. It builds a deterministic small hybrid
Qwen fixture and compares real Metal logits and greedy token IDs against an
independent CPU Transformers reference; a missing Metal device fails the GPU
request. The saved result certifies only those miniature numerical checks, not
full Qwen3.8 generation, quantized formats, quality or speed.

The extended protocol reads sequential and random SSD workloads, each for up to
the specified duration with a 1 TiB read cap, then runs sustained inference.
Thus its total duration may exceed three times `--seconds`. These are reads,
not terabyte writes or allocations. Stop with Ctrl-C if needed.

Inspect TTFT, prefill and decode rates separately. Compare the first and last
inference epochs on the fanless Mac, plus memory and swap growth. `pmset -g therm`,
`vm_stat` and swap status are retained before and after. They are system-level
observations, not per-kernel timings or calibrated temperature sensors. The
script never obtains root privileges or changes wired-memory limits.

For energy and detailed thermal/GPU analysis, run Apple's `powermetrics` in a
second terminal if you choose to authorize it. First inspect `powermetrics -h`
on the installed macOS version and select available CPU/GPU/thermal samplers;
sampler availability varies. Preserve the raw output and align timestamps with
the inference run. Missing energy or temperature readings remain unknown, not
zero. Optional Metal capture is available through
`MTL_CAPTURE_ENABLED=1 python benchmarks/profile_metal_next.py /local/checkpoint`.

## Read results conservatively

Compare the same machine, model revision, precision, prompt/token count, context,
power mode and cache state. Keep all raw runs, not only the fastest. An I/O
microbenchmark speedup is not a tokens/s gain. A miniature GLM argmax match does
not establish natural-language quality for the full GLM checkpoint. Qwen's
distinct adapter must reject unsupported operations until real generation is
verified; the open checkpoint is not the commercial service. The optional Qwen
hybrid miniature result is separate from the GLM GPU suite. Qwen validation on
an actual M5 remains pending until that command is run on real hardware and its
results are inspected.

`transfer_bounds()` in `benchmark_ooc.py` computes an optimistic weight-only
limit from explicitly supplied parameter count, bit width, active parameters,
resident bytes, measured rate and accepted tokens per verification pass. Scales,
attention/KV traffic, random I/O, GPU compute and conversion costs are excluded.
It is labelled theoretical and must remain separate from the measured JSON.
