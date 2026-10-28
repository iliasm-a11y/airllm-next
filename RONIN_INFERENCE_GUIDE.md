# RONIN local inference in the V5 research branch

GLM-5.3 Flash uses the separate `glm5-next` architecture. The experimental
native GGUF port, its exact validation scope, download checks and local
conversation adapter are documented in [RONIN_GLM53_FLASH_FR.md](RONIN_GLM53_FLASH_FR.md).
The older `--backend glm` command below selects `glm_moe_dsa`, not this model.
No real GLM-5.3 Flash conversation or M5 qualification is claimed by this port.

The engine now includes a bounded asynchronous local job API. One loaded Next
model and its weight caches serve several logical agents. **Only one native
KV/recurrent context is resident at a time.** Registered jobs are not concurrent
inferences, and the tests do not establish full GLM-5.3/Qwen3.8 generation or
completed autonomous RONIN missions.

## Run the service

Create the model and serve it on the same thread. The CLI does this:

```sh
airllm-next agent-serve /path/to/local/checkpoint --backend llama \
  --state-dir /path/to/private/ronin-state --host 127.0.0.1 --port 8080 \
  --max-new-tokens 128 --quantum-tokens 8 --max-requests 64
```

The model must already exist locally. `--backend glm` selects the GLM native
backend; `qwen-experimental` retains that backend's experimental status. The
existing synchronous `/v1/completions` and chat API remain available through the
separate `serve` command. The agent service exposes the job API below rather
than pretending to implement the whole OpenAI API.

```python
from airllm.nextgen.agent_engine import AgentEngine

# Construct `model` on this thread with explicit NextConfig memory limits.
with AgentEngine(model, state_dir="/path/to/private/ronin-state") as engine:
    first = engine.submit([1, 7, 8], max_tokens=32, agent_id="planner", priority=10)
    second = engine.submit([1, 9, 11], max_tokens=32, agent_id="reviewer")
    while engine.status()["queued"]:
        completed_quantum = engine.step()
    result = engine.snapshot(first)
```

The IDs above illustrate a token API and do not constitute a meaningful prompt
for every tokenizer. A text request uses the selected model's local tokenizer.

## HTTP contract

| Method and path | Purpose |
|---|---|
| `GET /v1/models` | Local model ID |
| `GET /health` | Queue and memory/disk/write reservations |
| `POST /v1/agent/jobs` | Admit a token or text job; HTTP 202 |
| `GET /v1/agent/jobs` | Bounded retained job inventory |
| `GET /v1/agent/jobs/{id}` | State, output IDs/text and per-job cost/latency |
| `POST /v1/agent/jobs/{id}/pause` | Pause at the next HTTP/quantum boundary |
| `POST /v1/agent/jobs/{id}/cancel` | Cancel and release the resident context |
| `POST /v1/agent/jobs/{id}/resume` | Resume a paused/cancelled/failed transcript |
| `DELETE /v1/agent/jobs/{id}` | Forget a paused/terminal job and remove its journal |

Submit JSON such as:

```json
{"model":"airllm-next","agent_id":"planner","prompt":"What is the capital of France?","max_tokens":16,"priority":10}
```

Use exactly one of `prompt` and `input_ids`. `agent_id` is a scheduling/metrics
label, not an implicit shared conversation: submit the complete intended token
context for each new job. Control actions accept `{}`. Higher
integer priorities run first; equal priorities rotate after each token quantum.
Priorities range from -1000 to 1000. EOS lists contain at most 256 valid IDs.
Sampling, multimodal requests, streaming and `tools` fields are rejected. Tool
execution remains with the RONIN orchestrator: it receives text, validates its
own tool protocol, executes an authorized tool, and submits a new bounded
prompt with the resulting context. This module does not execute cybersecurity
or other external actions.

The server binds to IPv4 loopback only. It does not implement authentication,
remote serving or a multi-user permission system. HTTP parsing, tokenization,
model forward, state changes and journal writes all execute on the model-owning
thread, preserving MLX stream affinity. Cancellation cannot interrupt a kernel,
blocking read, prefill or replay already in progress. A large quantum or long
reconstruction therefore increases cancellation and high-priority request
latency. HTTP parsing is synchronous; a slow local client can also delay
inference (the socket has a five-second inactivity timeout, not a whole-request
deadline). Persistent high-priority demand can starve lower-priority jobs.

## State isolation and exact replay

A job owns its prompt, committed output IDs, EOS policy and durable status. It
never receives another job's KV or recurrent banks. Switching agents releases
the sole native session and replays the selected agent's transcript. This is an
explicit time/memory tradeoff: frequent context switching can greatly increase
weight reads, compute and latency.

Reconstruction repeats the original prompt prefill, then feeds each recorded
generated token one at a time, checking every greedy prediction. It does not
batch old generated tokens into a different numerical schedule. A changed
prediction yields `replay_token_mismatch`; it does not silently continue from a
different history. This is same-runtime ordinary-greedy replay, not a claim of
bitwise agreement across frameworks, hardware, quantization or precision modes.
No cross-agent prefix-state reuse is implemented in this branch.

The identity includes checkpoint fingerprint, architecture configuration,
compute dtype/device, MLX version, TF32 environment and prefill/output tiling.
The existing checkpoint fingerprint uses paths/header metadata and file
size/mtime; it is not a full hash of hundreds of gigabytes. The replay checks
are an additional observed-token guard, not a cryptographic full-model proof.

## Recovery and bounded writes

An explicit `state_dir` is required for the HTTP service. An exclusive advisory
process lock prevents two engines from writing that directory. Files contain
prompts and output IDs, not weight copies or KV payloads; protect this local
directory as private data. Newly created directories/files use 0700/0600.

A full transcript journal is committed with an atomic replacement, file fsync
and directory fsync once per completed quantum and on control changes. A crash
can discard an unreturned quantum; the last committed transcript remains the
recovery point. Incomplete temporary files are removed under the exclusive
lock. A corrupt checksum, incompatible identity or invalid record fails closed.
After restart, queued/running jobs become **paused** and require explicit resume.

On a model/storage failure, the uncommitted quantum is discarded, the native
session is released, and the job becomes failed while other queued jobs remain
usable. Resume reconstructs the last committed token prefix. If journal writing
itself fails, the reported error is `journal_write_failed`; durability cannot be
promised beyond the last successful journal. Recovery under actual sudden power
loss or an unplugged physical external SSD is not validated here.

| Budget | Bound and accounting |
|---|---|
| Model payload | Existing `NextConfig`: weights, I/O, one KV/recurrent context, workspace and OS reserve |
| Retained jobs | `max_requests` counts queued **and retained completed** jobs; forget old results |
| Transcript memory | Conservative 96 bytes per reserved prompt/output/EOS token plus 4096 bytes/job; this is managed transcript accounting, not a whole-process RSS bound |
| Journal disk | `max_journal_bytes` reserves both old and replacement copies for every admitted job; journal payload allowance is 8192 + 16 bytes per reserved token |
| Journal traffic | `max_journal_write_bytes` defaults to 256 MiB per engine process; counts bytes offered to file writes, including failed writes |

Full-transcript rewrites cost more than append-only logging: for a long response
at a fixed small quantum, cumulative journal bytes can grow quadratically in the
response length. The cumulative write cap makes that cost explicit and stops
further commits when exhausted. It resets on engine restart; it is not a
lifetime SSD-wear measurement. Logical journal traffic differs from filesystem,
controller and physical NAND writes. HTTP bodies and Python/framework/OS
allocators also require headroom beyond these managed reservations.

Per-job metrics include queue-inclusive first-token/completion latency,
inference time including replay, replay time/tokens, logical model-weight bytes,
encoded-file bytes, context payload peak, failed quanta and discarded tokens.
The global service reports live/reserved limits and cumulative journal traffic.
These do not claim measured physical external-drive reads or thermal behavior.

## Validation

`tests_next/test_agent_engine.py` and `test_agent_server.py` exercise priority
ordering, round-robin isolation, explicit thread ownership, restart/resume,
corruption/model/replay mismatch, interrupted writes, memory/disk/write limits,
EOS bounds, cancellation, HTTP validation and real miniature native GLM, Llama
and recurrent Qwen sessions. Random miniature weights validate state behavior,
not trained quality or full giant-model support.

The executed trained CPU run used SmolLM2-135M-Instruct with two prompts,
eight output tokens each, a 16 MiB weight cache and warm OS pages. All queue and
restart output IDs matched ordinary greedy generation exactly.

| Path | Two-job wall time | Short-run inference jobs/hour | Logical weight bytes | Journal bytes offered |
|---|---:|---:|---:|---:|
| Ordinary serial | 12.341 s | 583.43 | Not recorded per job | None |
| Queue, quantum 8 | 12.899 s | 558.20 | 5,154,209,280 | 3,697 |
| Queue, quantum 1 | 88.671 s | 81.20 | 26,167,928,832 | 17,131 |

Quantum 8 added approximately 4.5% wall-time overhead in this one short run.
Quantum 1 was approximately 7.19 times slower than ordinary serial decoding and
replayed 602 token positions. This negative result quantifies the cost of
one-context interleaving; it is not an inference speed optimization. Durable
restart of the first job took 11.166 s including its initial two-token quantum,
reconstruction and completion. The measured whole-process peak RSS was
517,914,624 bytes. No external SSD, physical reads, memory-overflowing model or
thermal plateau was validated by this run.

The capital answer decoded to “The capital of France is Paris.” The arithmetic
answer stopped after “To calculate 2 + 2,” at the explicit eight-token budget.
Thus even successful inference-job completion does not certify completion of
the requested semantic task, and these numbers must not be presented as
finished autonomous RONIN missions/hour. Raw measurements and per-job costs are
in `evidence-v5/agent-queue-trained.json`; 27 final agent tests passed in 5.20 s
(`evidence-v5/agent-tests.log` and `.xml`).

The reproducible trained experiment is:

```sh
python research/v5/benchmark_agent_queue.py /path/to/SmolLM2-135M-Instruct \
  --output evidence-v5/agent-queue-trained.json
```

It compares two eight-token trained inference jobs against ordinary generation,
then interleaves and restarts them. It records completion jobs/hour for that
short CPU workload, replay overhead, journal traffic and RSS. That quantity is
**not autonomous RONIN missions/hour**, a sustained thermal result, or a giant
checkpoint benchmark. See the actual JSON evidence for executed measurements.
