# Model compatibility in the V5 research branch

The software starts at `9629c27b9d8bdf697ae866498ad468e11eb183d4`
(4.1.0rc1). Metadata recognition, weight decoding, reduced numerical validation,
trained generation, and full giant-checkpoint generation remain separate claims.
Neither a passing preflight nor an architecture identifier certifies generation.

| Model/representation | Metadata inspected in this mission | Weight/runtime status | Full-checkpoint generation in this mission |
|---|---|---|---|
| `zai-org/GLM-5.3`, official FP8 | Pinned config and complete index fetched; 141 shards, 118,629 tensors, 755,617,140,416 declared payload bytes | Existing E4M3/DSA/MoE backend plus V5 real trained gate/down matrix subrange checks: decoded FP32 bits equal independent Torch; no whole expert/attention validation | **Not performed** |
| `Qwen/Qwen3.8-2.4T-A95B`, BF16 | Pinned config and complete index fetched; 213 shards, 1,609 tensors, 4,892,365,451,008 declared payload bytes | Existing explicit experimental floating hybrid backend; Gated DeltaNet, convolution, gated attention and MoE miniature validation | **Not performed**; original payload exceeds nominal 2 TB |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | Pinned config and complete index fetched; 213 shards, 287,119 tensors, 2,496,066,252,544 declared payload bytes | FP8 metadata recognized; experimental Qwen runtime still requires floating weights; no full FP8 Qwen numerical certification | **Not performed**; original FP8 payload also exceeds nominal 2 TB |
| `nvidia/Qwen3.8-2.4T-A95B-NVFP4` | Pinned config and complete index fetched; 200 shards, 569,081 tensors, 1,444,420,107,432 declared payload bytes | NVIDIA ModelOpt mixed representation is explicitly rejected by the Next runtime; it is not affine MLX INT4 | **Not performed**; nominal capacity fit does not provide a decoder or numerical equivalence |
| Trained SmolLM2-135M-Instruct, floating Llama | Original release has full trained CPU/Metal evidence; V5 measurements are identified in `FINAL_VALIDATION_V5.md` | Separate native bias-free SiLU Llama backend, selective embeddings and output tiling | Small trained checkpoint, **not a giant-memory validation** |
| `codellama/CodeLlama-7b-hf`, original BF16 checkpoint | Pinned revision `6c284d1468fe6c413cf56183e69b194dcfa27fe6`; 291 tensors and 13,477,093,376 payload bytes; shard LFS SHA256 verified | Four FP32-compute tokens compared with independent streamed Transformers; fixed logit tolerance passed, max error 5.5313e-5; V4/V5 logits bit-identical on this run | **Completed on Linux CPU**; checkpoint exceeds host-visible RAM and 8 GiB cgroup; 0.03648 decode tok/s, no practical-speed/M5/full-quality certification |
| Full-head Llama with default or linear RoPE, floating weights | Current `rope_parameters` and legacy `rope_scaling` aliases are checked; the official DeepSeek Coder 6.7B configuration's linear factor 4 and theta 100,000 normalize successfully | V5 adds constant position scaling `1/factor`; 71 new/related tests pass on MLX CPU, including independent miniature Torch prefill/cached/generation parity beyond the declared original context | Metadata admission and miniature numerical validation only; no complete DeepSeek Coder generation claim |
| Lossless standalone packs | Existing bounded manifest and transport checks | Byte-preserving transport of the original stored weights; numerical computation retains its separate contract | No measured giant-pack compression ratio or giant-generation claim |

Payload bytes are the official index's `metadata.total_size`, not complete
filesystem footprints. Decimal TB is used for the capacity comparison. Auxiliary
assets, filesystem reserve, conversions and temporary duplicate representations
need additional space. The real formatted free capacity of the user's SSD has
not been measured here. No complete GLM-5.3 or Qwen3.8 shard was downloaded.
GLM acquisition was limited to six verified HTTP 206 subranges totalling
1,154,264 response-body bytes, including a shard header, two FP8 matrix row
tiles and their official scales. The separate complete 7B CodeLlama validation
download and its memory measurements are recorded in `FINAL_VALIDATION_V5.md`.

The GLM samples are layer 3, expert 0: gate rows 128:256 (128x6144) and down
rows 256:384 (128x2048). Source bytes remain exact after aligned rebasing into
a partial fixture. NativeWeights FP32 decode is bitwise equal to independent
Torch float8 conversion and official scales. Three fixed-position products
have maximum errors 1.15484e-7 and 3.72529e-8 under predefined `atol=rtol=2e-5`.
This is real trained tensor-component evidence, not full matrix/expert/DSA,
full model generation, dynamic activation quantization or Metal validation.
See `evidence-v5/glm53-real-sample.json`.

DSA boundary ties are now resolved against the installed PyTorch CPU top-k
convention when equal finite scores cross the selected-key boundary. An unchanged
seed-521 miniature exposed the inherited mismatch, reproduced on 4.1: explicit
all-ones masks did not remove it. The corrected eight-token native trajectory
and cached logits match the independent reference. No stable cross-build/CUDA
tie ordering is claimed. The check adds a host synchronization after T exceeds k;
tied rows require bounded score copies/CPU work, and a late Torch import can add
library RSS outside the managed scratch. This is a fidelity fix, not a new Metal
speed result; see `evidence-v5/glm-dsa-tie-diagnostic.json`.

Llama linear RoPE is a narrowly implemented constant scaling rule, not general
RoPE-variant support. Conflicting current/legacy declarations, nonfinite or
nonpositive theta/factor, dynamic/YaRN/attention scaling and partial-head variants
are rejected. GLM and Qwen keep their default-only RoPE guard. The CPU test log
and JUnit evidence are `evidence-v5/linear-rope-tests.log` and `.xml`; no new
linear-RoPE Metal measurement is implied.

## Official metadata provenance

The source snapshots and their SHA256 hashes are in
`evidence-v5/sources/manifest.json`; index payload totals, shard counts, index
hashes and retrieval timestamps are in the four `*-index-summary.json` files.
Configs were fetched again from the exact index revisions and matched the
initial `main` snapshots byte for byte.

| Repository | Pinned revision | Config |
|---|---|---|
| GLM-5.3 | `ba32f8c19c8738e2c57c71798c601d477a7bf844` | [Official config](https://huggingface.co/zai-org/GLM-5.3/resolve/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json) |
| Qwen BF16 | `207bd685a7e3696cfaff12ded7c6a7ea0f88c996` | [Official config](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json) |
| Qwen FP8 | `d2dc35658bcf77e66643428cb52e774cc3b5bd29` | [Official config](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8/resolve/d2dc35658bcf77e66643428cb52e774cc3b5bd29/config.json) |
| Qwen NVFP4 | `74eb734297318ecf7c48016a4ac8ce1b47cc74a9` | [Official config](https://huggingface.co/nvidia/Qwen3.8-2.4T-A95B-NVFP4/resolve/74eb734297318ecf7c48016a4ac8ce1b47cc74a9/config.json) |

GLM declares 78 layers, 256 routed experts and top-8 routing, with E4M3 weight
blocks of 128 by 128. Qwen declares 92 layers, 512 experts and top-10 routing;
its explicit schedule contains 69 linear-attention layers and 23 full-attention
layers. The NVFP4 config declares ModelOpt `MIXED_PRECISION`: expert weights use
NVFP4, other matrices may use FP8, and sensitive weights remain floating. That
layout and its scales require independent Apple-compatible decoding and numerical
validation; neither BF16 equivalence nor an INT4 shortcut is certified.

## Resource preflight added in V5

`airllm.nextgen.preflight.preflight_checkpoint` checks local metadata, declared
file presence, file-path confinement, state reservations, configured memory
budgets and actual or explicitly supplied resource scenarios. The CLI exposes
this as `airllm-next preflight`. It never downloads or writes model files. File
presence is deliberately reported separately from header and payload integrity.

For GLM and Llama it counts 128-token KV allocation granularity and conservatively
reserves overlap with a growing old layer. For Qwen it counts its recurrent and
convolution state plus **all existing banks retained by transactional rollback**,
including full-attention KV. This is broader than recurrence alone.

At 4,096 retained tokens and BF16 compute, the official configurations imply:

| State component | GLM-5.3 | Qwen3.8 |
|---|---:|---:|
| Logical KV payload | 390,070,272 B | 385,875,968 B |
| FP32 recurrent payload | 0 B | 578,813,952 B |
| Convolution payload | 0 B | 11,304,960 B |
| Logical state total | 390,070,272 B | 975,994,880 B |
| Existing-state snapshot retained by Qwen forward transaction | 0 B | 975,994,880 B |
| Largest Qwen bank replacement, conservatively reserved in addition | 0 B | 16,777,216 B |

These are code/config-derived reservations, not measured RSS or complete peak
bounds. Qwen's state, one retained snapshot and a conservatively charged bank
replacement require 1,968,766,976 B at that context; its default 512 MiB KV
budget is insufficient. A bank shared with the snapshot can be charged twice
by this safe replacement reservation; it is not an observed physical peak.
Attention/logit workspaces, array/copy transients, weights and host objects need
their own memory. The preflight reports the remaining unknowns rather than
claiming a guaranteed 16 GB run. Batch sizes above one are blocked; logical agent
registration does not multiply the shared weight cache or enable simultaneous
inference.

## Remaining generation gates

GLM's next full-checkpoint gate is a real local official checkpoint plus the
tokenizer, numerical checks against independently executed reference components,
and measured generation under a real 16 GB memory budget. Weight-only FP8
decoding does not reproduce a server's dynamic activation quantization semantics.
MTP weights being present does not enable MTP execution.

For Qwen, full floating-checkpoint validation is still missing; FP8 requires a
separate validated hybrid runtime path, while NVFP4 requires its own layout and
scale handling. Recurrent speculative verification is not enabled. Any lossy
conversion needs checkpoint-specific quality and numerical measurements.

The current execution host is Linux x86_64. It cannot supply new M5 Metal,
thermal, external USB SSD or 16 GB Apple unified-memory measurements. Existing
release Metal results remain attributed to the original release evidence.
