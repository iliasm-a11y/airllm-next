# V5 measured greedy verification and exact prompt retrieval

Base: `9629c27b9d8bdf697ae866498ad468e11eb183d4`. The existing target verifier,
lagging pending token, accepted-prefix rollback and EOS rules are retained.
This is an integrated policy/retrieval improvement, not a new numerical model.

## Problem and hypothesis

V4 grows depth from acceptance alone and estimates ordinary decoding from one
initial pass. A cold/noisy pass can authorize poor depth choices. Target reads
are reported but draft traffic is missing, and a prompt search without matches
repeats nested suffix slices.

Hypothesis: bounded, generation-local depth trials can reduce wasted target
passes by selecting measured time per emitted token first, then measured byte
cost when time is within 3% of the fastest observed depth. Exact retrieval can
use a linear suffix comparison index without changing the proposed tokens.
Neither hypothesis predicts unknown future MoE experts or presumes useful
acceptance on a giant trained model.

## Integrated implementation

`AdaptiveSpeculation` now stores at most eight observations per requested depth.
Its cost is `sum(draft_and_verify_seconds) / sum(emitted_tokens)`, so rejected
proposals and all draft wall time remain in the denominator/numerator correctly.
Encoded bytes break time ties only when every compared depth has encoded
observations; otherwise one common logical domain is used, or bytes are omitted.
A lower byte count never excuses a measured slowdown beyond the existing guard.
Per-depth samples measure draft-plus-verification service cost: policy
bookkeeping and periodic calibration are outside those samples. The reported
whole-generation time includes both. The controller has no guarantee that
smaller service cost offsets its own overhead on every workload.
A second guard compares observed total decode time per emitted token with the
rolling singleton reference, including calibration and bookkeeping, and disables
proposals after repeated observed regressions. It is an online safeguard, not a
counterfactual proof of the unexecuted ordinary trajectory's time.

Two successful observations permit a neighboring depth trial. Low acceptance
shrinks depth; repeated time regressions, empty drafts or zero acceptance disable
further proposals for this generation. Every four speculative rounds, a real
single-token target pass refreshes a bounded lower-median baseline. Calibration
is part of end-to-end generation time and target I/O; it is not a free probe.
This conservative policy has no universal optimality or speed guarantee.

The runtime observes target and loaded native draft checkpoint counters
separately. A generic callable's unknown transport remains `null`. The in-memory
prompt retriever explicitly declares zero transport. New per-round fields
include encoded bytes, draft bytes, requested depth, actual verification input
length, rejected proposals, baseline and next depth. Full generation fields
include calibration calls, target positions and weighted depth costs. Unknown
physical SSD reads, draft allocations or hardware costs are not invented.

`PromptSpanDraft` now computes exact common suffix lengths with the Z algorithm
on the reversed bounded transcript: O(lookback) comparisons/storage. It uses no
hash collisions and preserves V4's longest-match, newest-occurrence and overlap
rules. Eight bounded full-length probes retain the old repeated-pattern fast
case before constructing the linear index. Its output still passes through the
actual target verifier.

## Numerical and state contract

Greedy equivalence still assumes block-invariant target argmax. Floating-point
batched reductions can differ from ordinary one-token execution; the change does
not claim universal bit-identical trajectories. All target outputs are checked
by the target; rejected state is truncated with the existing append-only GLM and
Llama semantics. `adaptive=False` retains fixed-depth operation. Qwen speculative
generation remains explicitly disabled until recurrent snapshots are integrated.

## Controls and failures retained

`test_v5_speculative_cost.py` covers cost validation, bounded weighted histories,
cold baseline outliers/drift, byte-domain consistency, bytes-versus-time
tradeoffs, real calibration, exact state prefixes, draft transport attribution,
unknown transport and empty-draft termination. Differential retrieval controls
cover random, repeated, overlapping, empty and truncated transcripts. An
additional development run checked 10,000 randomized cases against V4.

The first linear-search implementation improved misses but made immediate
repeated matches roughly 80 times slower (about 0.8 ms instead of 10 us).
`evidence-v5/speculation-lookup-initial-z.json` preserves this negative result.
The bounded full-match fast path was added in response.

The final isolated lookup microbenchmark (`speculation-lookup.json`, nine
randomized-order repetitions) measures roughly 26.5x faster no-match lookup and
33.4x faster random-token lookup for an 8,192-token window; repeated full matches
remain about 9–10 us in both versions. These are CPU retrieval microbenchmarks,
not full inference, model-quality or SSD speed results. Reproduce with:

```bash
python research/v5/benchmark_prompt_lookup.py --output evidence-v5/speculation-lookup.json
```

The trained-checkpoint policy ablation is executable with:

```bash
python research/v5/benchmark_speculation.py --model /path/to/SmolLM2-135M-Instruct --tokens 16 --repeats 2 --output evidence-v5/speculation-trained.json
```

It loads the original policy/retriever from the exact Git base and uses the same
current weight engine and budgets for every policy. This isolates policy and
retrieval; it is not a claim comparing every V4/V5 engine change. Construction
and tokenization are excluded from time, warm OS pages are not purged, the small
checkpoint fits host RAM, and no external SSD or M5 is measured. A deliberately
wrong zero-transport draft is a negative correctness/cost control. A real trained
small-model draft and complete GLM/Qwen route traces remain unvalidated.

## Executed trained experiment: positive and negative results

`evidence-v5/speculation-trained.json` and its log contain 13 completed
16-token generations: two prompts, two randomized-order repetitions, three
methods, plus a deliberately wrong draft. The checkpoint SHA-256 is
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Every speculative output matches the same ordinary target output in its group;
the wrong draft also matches ordinary decoding. This is observed trajectory
parity on these trained cases, not a universal batching equivalence theorem.
All runs use the current CPU weight engine, FP32 computation, an 8 MiB decoded
cache, 32 MiB I/O, no prefetch and 2,048-row output tiles. The ordinary/V4/V5
methods have identical model/budgets; only retrieval and speculative control
change. Tokenization and construction are excluded from the reported time.

| Prompt | Ordinary median | V4 lookup median | V5 lookup median | V5 versus V4 |
|---|---:|---:|---:|---|
| Repeated sentence | 7.107 s | 7.969 s | 7.852 s | 1.5% shorter, too small to generalize from two repetitions |
| Repeated `A B C` pattern | 6.927 s | 5.614 s | 5.835 s | **3.9% longer** |

On the sentence, V5 remains **10.5% slower than ordinary decoding**; the measured
total-cost guard disables further proposals. This profile does not justify
speculation. On the repeated pattern, both V4 and V5 accept ten draft tokens and
read 1,883,232,000 logical/encoded file bytes instead of ordinary decoding's
4,516,911,360: about 58.3% fewer bytes. V5 is about 1.19 times faster than ordinary
decoding here, but the existing V4 verifier is faster still. The traffic
amortization is inherited block verification, **not a new V5 byte reduction**.
The small checkpoint fits the host page cache; these bytes are not external SSD
physical traffic. Host physical read observations in these runs are zero.

The wrong-draft control emits the same tokens, accepts zero draft tokens and
disables proposals with `no_accepted_draft_tokens`. These short trained runs do
not trigger a periodic recalibration; its real-forward and prefix invariants
are exercised by the deterministic control test, not advertised as a giant
trained-model result. No complete GLM-5.3/Qwen3.8 generation is demonstrated.

The final focused regression (`speculation-tests.xml`) has **65 passing tests**,
including 46 inherited verifier tests and 19 V5 policy/retrieval cases. The
whole repository regression is the responsibility of the final validation.

Exact executed command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/path/to/dependencies:air_llm python research/v5/benchmark_speculation.py --model /path/to/SmolLM2-135M-Instruct --tokens 16 --repeats 2 --prompt-limit 2 --output evidence-v5/speculation-trained.json
```
