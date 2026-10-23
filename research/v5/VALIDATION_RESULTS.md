# Executed V5 trained-model validation

All runs here used **Linux x86_64, MLX 0.32.4 CPU**, Torch 2.14.1+cpu and
Transformers 5.20.2. Metal and the M5 are absent in this environment. Existing
M5 results in `evidence-final/` remain historical evidence, separately attributed.

## Genuine checkpoint larger than physical RAM

The complete original **codellama/CodeLlama-7b-hf** checkpoint, revision
`6c284d1468fe6c413cf56183e69b194dcfa27fe6`, contains 291 BF16 tensors totaling
**13,477,093,376 bytes**. Its two safetensors files total 13,477,127,208 bytes.
Both official LFS SHA256 values were verified during a bounded streamed download.
No config or weight conversion was written. Host memory was 10,451,439,616 bytes;
the cgroup limit was 8,589,934,592 bytes. The checkpoint exceeds both limits.

With the prompt `def add(a, b):` followed by a newline and four spaces, the
engine's ordinary forward/greedy loop generated **`return a + b`**, four tokens.
The same original checkpoint and input were processed sequentially by immutable
V4, V5, and an independent reference using official Transformers Llama layers,
RMS normalization, rotary embeddings and dynamic caches. The reference loads
only one layer at a time; a separate miniature control verifies it against fully
resident ordinary Transformers prefill and cached decoding.

| Observation | V4 | V5 | Independent streamed HF |
|---|---:|---:|---:|
| Four-token wall time | 165.164 s | 164.401 s | 44.709 s |
| First token | 82.194 s | 82.161 s | 13.127 s |
| Subsequent decode | 0.03616 tok/s | 0.03648 tok/s | 0.09499 tok/s |
| Process peak RSS | 1,405,104,128 B | 1,404,461,056 B | 1,370,660,864 B |
| Process physical read counter | 50,575,691,776 B | 50,383,945,728 B | 41,399,136,256 B |
| Max absolute logit error versus independent HF | 5.53e-5 | 5.53e-5 | 0 |

All four output tokens match. All observed logits meet the **unchanged,
predeclared** `atol=1e-4, rtol=5e-4` criterion. V4 and V5 logits are bit identical.
System swap was zero before and after each worker. Cgroup event counters report
no OOM or OOM kill; shared filesystem cache can reach the cgroup cap and must
not be confused with process RSS. Explicit V4/V5 budgets were 256 MiB decoded
cache, 256 MiB I/O, 512 MiB workspace, 128 MiB KV, 512 MiB system reserve,
256 MiB maximum tensor and FP32 computation from the unchanged BF16 source.

This is a real, complete, trained checkpoint larger than the available RAM,
not an artificially inflated fixture or capped-cache proxy. It validates this
short workload's numerical fidelity. **It is not a large-model speedup**: the
V4/V5 ratio is about 1.005, with one short case and uncontrolled OS cache. The
independent Torch reference is faster in this observed CPU order; that negative
comparison is retained. This does not establish useful autonomous-agent
throughput, long-context stability, full GLM/Qwen generation, Metal performance
or an external SSD rate.

Raw reports, all four logit vectors and source-module hashes are retained in
`evidence-v5/trained-large*`. The independent reference's initial constructor-API
failure is retained as `streamed-reference-initial-failure.*`; it was corrected
to the installed official API before the numerical control and large runs.
No tolerance was widened.

## Small trained regression

**HuggingFaceTB/SmolLM2-135M-Instruct**, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, weight SHA256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Three prompts, three repeats per prompt and sixteen tokens per generation were
executed by ordinary in-memory Torch, immutable V4, and final V5 default.
Every observed sequence matches Torch; V4/V5 final logits are bit identical;
maximum independent error is 7.25e-5 under the same fixed tolerance.

| Same-budget control | Median 16-token wall time | Process peak RSS |
|---|---:|---:|
| Ordinary resident Torch | 1.007 s | 1,287,405,568 B |
| V4 streaming default | 5.919 s | 939,409,408 B |
| V5 streaming default | 6.182 s | 939,929,600 B |

V5 default is approximately 4.4% slower in this observation. The controls were
measured earlier, with V5 measured after integration, so this is a correctness
regression/timing observation rather than a randomized performance conclusion.
The resident Torch CPU implementation is faster. The entire small checkpoint
fits physical RAM; it does not validate the out-of-core criterion. Separate
feature ablations and their positive/negative findings are in
`weight-residency-trained.json` and `speculation-trained.json`.

The small raw logits, generation traces, memory and I/O counters are preserved
under `evidence-v5/trained-control*`; the joined comparison is
`evidence-v5/trained-regression.json`. `trained-source-consistency.json` compares
run-source hashes with the current modules and exposes any later changes.

## Integrity and reproducibility

`evidence-v5/inherited-audit.json` inventories 249 inherited research/evidence
files with hashes; structured JSON, compressed JSON and XML all parse. This is
an evidence integrity check, not a rerun of old M5 benchmarks. Historical
compression regressions, failed XOR/low-rank ideas and teacher-forcing limits
remain preserved. Run `scripts/validate_v5.sh` for the offline suites and
reference control. Explicit download and trained benchmark commands are in
`download_validation_model.py`, `benchmark_trained.py`, `benchmark_large.py`.
Normal engine construction and tests never download these models.
