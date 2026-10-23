# Independent implementation review — V5 research branch

Reviewed against `9629c27b9d8bdf697ae866498ad468e11eb183d4` on 2026-10-23.
This is a source review, not an independent certificate of the giant checkpoints
or the unavailable M5/Metal hardware.

## Scope

Reviewed the weight/compact-cache integration, speculative cost controller and
runtime callbacks, agent queue and HTTP service, resource preflight, GLM/Llama
attention and caches, Qwen attention/recurrence, offset reader, reversible packs,
quantization and sidecar boundaries. Existing transport checks and the Qwen
speculation rejection remain in place.

## Findings and corrections

| Finding | Resolution |
|---|---|
| Agent EOS lists were retained outside transcript reservations | EOS lists are capped at 256 IDs before copying/sorting; reservations include them |
| Atomic journal replacement temporarily retains the previous and new files | Per-job disk reservation includes both copies; abandoned temporaries are checked and removed under the exclusive process lock |
| A retained-file limit does not bound repeated journal writes | Separate per-process cumulative offered-write allowance and counter; the guide discloses full-transcript rewrite costs |
| Output tiles had no observed load cost under the cost cache policy | Their measured load time is now recorded under the actual tile key |
| Compact-cache lookup could bypass storage selector validation through Python numeric key equality | Expert/row selectors are checked before a cache hit can be served |
| Resource preflight accepted a batch-size scenario the runtime cannot execute | Explicit batch-one blocker added; concurrency remains a labelled capacity scenario |
| Qwen shares the new weight cache but lacked its pressure/metrics visibility | Both tiers are trimmed together near the reserve and both are exposed in metrics |
| GLM latent-bank growth counted only the replacement allocation | Growth reserves the old/new overlap and evaluates copies before dropping prior banks |
| Qwen multi-chunk bank replacement could miss an old bank different from the entry snapshot | Reservations now conservatively count current banks, retained snapshots and the entire replacement before allocation; rollback preserves the entry banks on rejection |
| Compact NumPy residency must not also be offered to the MLX allocator ceiling | The ceiling subtracts the compact partition; combined target/draft admission still counts the whole managed payload |
| Late installed-package checking exposed inherited DSA boundary ties choosing different keys than Torch CPU | The unchanged seed/prompt/mask now use a bounded CPU-reference top-k fallback only for ambiguous finite boundaries; ordinary/agent eight-token parity and cached logits are checked |

The owner-thread agent service retains one architecture session, verifies
recorded greedy tokens during replay, and handles cancellation at quantum
boundaries. Switching agent contexts deliberately incurs replay work. It does
not certify simultaneous physical inference, prefix sharing, or tool execution.

## Additional guarded compatibility work

Llama now accepts full-head linear RoPE with a finite positive factor and theta.
Current and legacy aliases are normalized only when consistent. Unknown scaling
fields, dynamic schemes, attention rescaling and partial-head variants remain
rejected. GLM and Qwen retain their default-only RoPE validator. The implementation
uses MLX's documented position scale `1/factor`, matching Transformers' division
of inverse frequencies by that factor.

New independent miniature Torch parity tests cover scaled prefill, cached decode
and generation beyond the miniature configuration's declared original context.
The new cases and related existing Llama/config/adapter regressions passed:
**71 tests in 5.88 seconds** on Linux MLX CPU. Raw JUnit and log files are
`evidence-v5/linear-rope-tests.xml` and `evidence-v5/linear-rope-tests.log`.
This validates that narrow numerical implementation, not generation with the
complete DeepSeek Coder checkpoint or any extension on Metal.

## Limits retained

- Metadata fingerprints are not full checkpoint payload hashes. Replay checks
  committed greedy tokens; it does not replace a complete integrity audit.
- Configured tensor/cache/state reservations are not a complete Python/native
  allocator/RSS bound. Process and cgroup observations remain necessary.
- Compact source residency is opt-in and can lose when reuse is poor or
  conversion dominates. Output tile residency competes with other weights.
- Full GLM-5.3 and Qwen3.8 generation, external SSD behavior and new Metal timing
  require their own recorded validation. CPU tests cannot establish those facts.
- Greedy block verification retains its stated floating-point batch-invariance
  caveat. Recurrent Qwen speculative verification remains disabled.
- Per-depth controller samples cover draft and verification service, excluding
  policy bookkeeping and periodic singleton recalibration. The final controller
  additionally checks observed cumulative decode time including those costs.
  Whole-generation before/after timings remain necessary to establish benefit;
  the guard can react to an observed slowdown, not guarantee it never occurs.
