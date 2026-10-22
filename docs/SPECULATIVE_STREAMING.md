# Greedy speculative expert streaming

The GLM runtime can verify several local draft tokens in one causal target
forward. Its existing MoE executor groups the actual router-selected token rows
by expert, so a weight loaded for an expert is reused across those rows. No
predicted expert replaces the router selection. This is a real executable path,
with regression tests and a miniature-checkpoint benchmark.

The supported algorithm is **greedy block verification**. Stochastic sampling,
MTP heads, verification trees, and Qwen recurrent-state speculation are not
implemented. Speculation is optional and disabled by omitting `draft_model`.

```python
from airllm.nextgen import AirLLMNextModel, NextConfig

# Both directories must already exist locally. Nothing is downloaded.
# Select budgets appropriate for BOTH models together; each has its own cache.
target = AirLLMNextModel("/Volumes/Models/target", next_config=NextConfig())
draft = AirLLMNextModel("/Volumes/Models/draft", next_config=NextConfig())
try:
    output = target.generate(
        [[1, 8, 9]], max_new_tokens=32,
        draft_model=draft, temperature=0,
        speculative_depth=4, speculative_adaptive=True,
    )
    print(output.tolist())
    print(target.metrics()["runtime"]["speculative"])
finally:
    draft.close()
    target.close()
```

The current built-in model adapter accepts GLM checkpoints. A compatible small
draft from another local runtime can be supplied through a callable
`propose(prefix_token_ids, requested_count) -> candidate_token_ids`. It must use
the target's token mapping and return at most the requested count. The runtime
does not create a remote connection or silently download a draft. The convenient
`generate()` adapter recomputes the draft prefix; an advanced proposal callable
can maintain its own bounded state. Target and draft resources must jointly fit
the machine: the scheduler does not automatically allocate a shared memory
budget between independently constructed models.

## Verification and rollback

After prefill, the target emits its own argmax token. One subsequent real target
decode step provides a same-generation timing baseline. A block then contains
the last emitted, not-yet-processed token followed by draft proposals. The target
evaluates every position under a causal mask. Proposals are compared with the
target argmax at the matching preceding position. On the first disagreement,
only the accepted prefix and the target's correction are emitted. If all draft
tokens agree, the target supplies one additional bonus token.

The GLM latent KV buffers retain only the committed prefix logically. Rejected
positions are discarded using `LatentKV.truncate(position)` and overwritten on
the next append. Every future attention operation reads through the logical
position, so stale bytes are never attended to. An accepted EOS can end with
the EOS itself already present in KV; otherwise the final emitted token remains
pending. No future request reuses this internal generation session implicitly.

The greedy distribution is verified by the target. Ordinary floating-point
rounding can differ between batched and one-token evaluation, particularly at
an exact argmax or router tie; this code does not claim bitwise equality across
all possible hardware kernels. Tests check actual output equality for the
miniature checkpoint and exercise rollback by comparing its logits with a full
recomputation. A recurrent state cannot be rolled back by moving a token
position; Qwen generation remains unsupported until its state snapshots or
recomputation are integrated and tested.

## Measurements and automatic fallback

Telemetry includes actual proposed, accepted and emitted token counts; target
forward calls; rejected KV positions; proposal and verification wall times; and
logical target bytes per finally accepted output token. A round's denominator
includes its accepted draft tokens and actual target correction or bonus token.
The adaptive policy increases depth after fully accepted blocks and reduces it
on poor acceptance. After warmup, two consecutive rounds whose proposal plus
verification cost per emitted token exceeds the measured autoregressive decode
baseline disable speculation for the rest of that generation. These are local
observations, not an assumed SSD bandwidth or a guarantee of optimal depth.
Four consecutive rounds accepting no proposed token also disable speculation,
independently of the timing baseline: such rounds cannot amortize a target
weight pass. This protects fallback from a noisy initial decode measurement.

Run the actual checkpoint benchmark with:

```bash
python benchmarks/benchmark_speculative_ooc.py \
  --checkpoint fixtures/glm-mini \
  --tokens 24 --repeats 3 \
  --output evidence-ooc/speculative.json
```

It compares depths 1, 2, 4 and 8 at identical float32 target precision with a
4 KiB weight cache, forcing matrix reloads. Every generation starts with empty
software weight caches. The OS page cache and compiled kernel caches are not
flushed. Logical reads are therefore not claims about physical external-SSD
bandwidth. On Linux, available process-level physical counters are reported
separately; unavailable counters remain null.

The accurate local draft in this benchmark is an independent copy of the same
random miniature checkpoint with an 8 MiB weight cache. It is deliberately a
correctness and expert-reuse fixture, not a representative small trained draft
for the full model. Draft reads and compute time are included in the raw results
and totals. The wrong zero-token draft exercises rejection and automatic
fallback. All generated token sequences must equal the target autoregressive
sequence or the benchmark fails. These results do not establish full GLM-5.3
quality, M5/Metal performance, or feasibility of an interactive trillion-scale
checkpoint on 16 GiB.
