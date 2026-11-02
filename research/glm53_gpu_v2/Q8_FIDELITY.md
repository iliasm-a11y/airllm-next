# Q8_0 CPU-compatible Metal arithmetic experiment

This experiment preserves the original GGUF Q8_0 bytes. It reproduces the temporary activation quantization and the pinned Apple ARM NEON `nrc=1` dot arithmetic rather than multiplying the packed weights by unquantized FP32 activations. It is default-off and does not change RONIN identities, slot count, model files, or production configuration.

## Arithmetic contract

The independent oracle calls `quantize_row_q8_0` and `ggml_vec_dot_q8_0_q8_0` exported by the selected pinned `libggml-cpu.dylib`. It does not compare the GPU to a second implementation of the same Python algorithm. Relevant native source is `ggml/src/ggml-cpu/arch/arm/quants.c` at llama.cpp commit `3f8af15a76ac8675812e856da45454bafc90f12c`, functions beginning at lines 41 and 1153.

For every 32 FP32 activation values, the CPU computes `d=amax/127` in FP32 and quantizes with the reciprocal of this **unrounded** scale, using nearest-even integer rounding. The scale stored in the temporary Q8_0 block is then rounded to FP16. Weight blocks remain the original 34 bytes: one FP16 scale and 32 signed integers.

The ARM `nrc=1` dot maintains four FP32 accumulation lanes for even blocks and four for odd blocks. Integer dot groups each combine four low-half and four high-half products. The implementation applies the FP16 weight and activation scales in FP32, accumulates each lane with explicit FMA, reduces each four-lane accumulator with the ARM pairwise order, and adds even and odd totals. An odd final block uses the CPU scalar tail order. A single scalar float sum across all blocks would not preserve this contract.

`kernels_q8/q8_activation.metal` and `q8_matvec.metal` use two GPU dispatches with a small temporary activation array. `q8_inline_helpers.metal` and `q8_matvec_inline.metal` fuse activation quantization into each row's dot and require no activation scratch allocation. The fused path intentionally repeats activation work across rows; it is a minimal integration experiment, not a measured optimum.

Both Python paths explicitly compile with MLX `math_mode=safe` and disable implicit contraction; explicit FMAs remain. The Python admission path rejects nonfinite inputs, subnormal FP32 activation quantization, and activation scales that overflow FP16, rather than silently claiming equivalence outside its tested contract. Payload limits include the original host matrix, a device copy, activation and result buffers; they do not replace an independent process RSS/swap guard.

## Actual component validation on Apple M5

The six GPU parameter cases cover widths 32, 64, 96, 256, 512, and 4096, with 13 synthetic packed rows each. Each case tests seeded normal input, all-zero input, and explicit half-integer ties. Both two-dispatch and fused kernels match the native ARM output **bit for bit**. Temporary activation integers and FP16 scale bits also match. Widths 32 and 96 test the odd-block scalar tail. Alongside the portable admission test, **7 pytest tests passed**, with no skips in the actual MLX environment.

The bounded GGUF reader then read only the first 32 rows of the actual GLM output head, 139 264 bytes at absolute shard offset 86 720. This is the original `[4096,154880]` Q8_0 head, not a converted or synthetic weight matrix. The activation was a seeded synthetic FP32 vector; it was **not** captured from a model execution. Both GPU variants and their separately compiled standalone Metal timestamp versions matched all 32 native ARM outputs bit for bit. Selected-byte SHA-256 and oracle library hashes are recorded in [q8_real_head32.json](q8_real_head32.json).

The small-matrix run charged 315 904 known payload bytes and recorded 189 184 peak MLX bytes. Its separate command-buffer measurements report 0.0950 ms median for activation-plus-dot and 0.1169 ms for the fused command. MLX synchronized host-wall medians were 0.6243 and 0.5599 ms respectively. Those are different measurements from different submission paths, and are not combined into a speed ratio. The native reference made one ctypes call per row, not a CPU10 graph dispatch. No GLM token-rate improvement, whole-head performance, or Fanxiang storage throughput follows from these component numbers.

## Native output-head patch, arithmetic mode 1

[patches/llama-q8-cpuarith.patch](patches/llama-q8-cpuarith.patch) introduces `AIRLLM_METAL_Q8_CPU_ARITH=1` for the output head. The later selective patch also admits explicit mode 2, described below. Without either enabled mode, the new arithmetic dispatch and safe-compilation options remain inactive. In mode 1 the custom kernel is selected only when all these conditions hold:

- `output.weight` is the contiguous Q8_0 `[4096,154880,1,1]` tensor.
- The operation is MUL_MAT with one contiguous FP32 activation vector, singleton batch dimensions, and FP32 output.
- The existing matvec path is actually selected; no CPU `nrc=2` prefill behavior is claimed.

The shader uses eight output rows per SIMD group, four lanes per row, one SIMD group per threadgroup and no threadgroup scratch. Its row address uses the native tensor byte stride. The unchanged native host dispatcher therefore uses `nr0=8`, `nr1=1`, `nsg=1`, `smem=0`. The original kernels and guards are retained.

When the flag is enabled, the patch also disables fast math for the MUL_MV shader library. This is required to reproduce the safe arithmetic validated above. Apple documents fast math as allowing potentially lossy floating-point assumptions; see [MTLMathMode](https://developer.apple.com/documentation/metal/mtlmathmode) and [fastMathEnabled](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled). Although custom dispatch is restricted to the output head, this compile option can affect other kernels from that library if an experiment also offloads other weights. The first qualification must therefore use the head-only placement policy. A broader hybrid run requires its own fidelity qualification.

The patch passed application checks and was then compiled into the isolated pinned runtime. The full-model evidence below is separate from the 32-row component proof; it does not infer model fidelity or speed from that small component alone.

## Reproduction

Use an environment with MLX, NumPy and pytest, and set the native library directory explicitly:

```sh
AIRLLM_GGML_LIBRARIES=/absolute/path/runtime/bin \
  python -m pytest -q research/glm53_gpu_v2/test_q8_compatible.py

python -m research.glm53_gpu_v2.q8_probe \
  --gguf /absolute/path/GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf \
  --libraries /absolute/path/runtime/bin --rows 32 \
  --output /absolute/path/q8-head32.json
```

`q8_probe` takes the shared single-experiment file lock. Its reader caps parsing and selected bytes; it does not map the 110 GB checkpoint. `--activation-f32` accepts exactly one saved contiguous FP32 vector of the selected tensor's width for a later real-activation ablation. The optional `--device-timer` uses the separately built bounded command-buffer adapter, records real `GPUStartTime`/`GPUEndTime`, and revalidates its output against the oracle. GPU timestamps are never substituted with CPU wall time.

## Selective Q8 matvec policy, separate default-off experiment

[patches/llama-q8-selective.patch](patches/llama-q8-selective.patch) is incremental on top of the head-only Q8 patch and the PR18 context changes. `AIRLLM_METAL_Q8_CPU_ARITH=2` allows the same compatible `nrc=1` kernel for a simple contiguous 2-D Q8_0/F32 matvec: width 64..65536 divisible by 64, output rows divisible by eight, one activation column, and singleton batch dimensions. Mode 1 retains the original head-only shape/name scope. Safe MUL_MV compilation applies to modes 1 and 2.

The independent flag `AIRLLM_GLM_Q8_ONLY=1` enforces a strict placement policy in GLM inference contexts with `gpu_layers=0` and mode 2. After building the **entire** graph and before scheduler allocation, it assigns every computational node to CPU, including unnamed nodes, views and KDA state-copy side effects. The sole exception is an eligible Q8 vector multiplication whose original weight is already in a Metal buffer; it is assigned to the owning Metal backend. Both reserve/fused-probe graphs and freshly built inference graphs receive the same complete walk. Reused graphs retain those assignments. An unsupported required operation, missing Metal backend, wrong architecture/mode, or training context aborts rather than choosing another backend silently.

A complete graph walk is necessary because `graph_get_cb` only sees named tensors. Pinning its current tensor alone would miss unnamed projections and state updates. The final walk is linear in the number of graph nodes and does not rebuild the numerical graph. It also keeps HC/RMSNorm, KDA recurrence/state operations, routing and multi-column prefill products on CPU, so the original CPU `nrc=2` behavior is not replaced by the vector kernel. Eligible single-column Q8 projections, including some decode Q/K/V products, can use GPU. The layer device remains CPU, preserving the HC fusion-resolution comparison with the CPU baseline. Weights are selected explicitly by the experiment harness; no model-wide offload default is enabled.

The native Metal shared and mapped buffer types deliberately report `is_host=false`. CPU prefill operations using those Metal-owned weights can therefore require scheduler-owned CPU copies, even on unified memory. The initial five-layer experiment was rejected by the unchanged swap guard and exposed a 677.5 MiB CPU workspace. A bound on selected GPU weight bytes did **not** bound that copy workspace.

The additional [mapped CPU-read patch](MAPPED_CPU_READ.md) permits direct CPU access only to constant Q8_0 weight leaves in an actual `MTL0_Mapped` buffer whose complete VM range is read-only. It requires both `AIRLLM_METAL_CPU_MAPPED_WEIGHTS=1` and `AIRLLM_GLM_Q8_ONLY=1`; mutable allocations, activations, other devices and unsupported ranges remain excluded. This is an immutable-weight exception, not a global declaration that Metal buffers are host buffers. Existing synchronization and lifetime rules remain in force for mutable data. The observed CPU workspace fell to 11.0011 MiB in the subsequent full-model runs. The original failed run remains negative evidence.

RSS, pressure and swap admission remain mandatory. The static placement planner caps selected original Q8_0 weights at 3 GiB; this is a weight budget, not a claim that the whole runtime fits inside 3 GiB. Every additional placement still requires its own captured fidelity and guarded memory result.

## Complete-model proof for the selective inline Q8 path

The isolated runtime was compiled with the complete patch stack. The `quant-tail` experiment selects original Q8_0 matrices from the final layers plus the output head, with `gpu_layers=0`, arithmetic mode 2, strict Q8-only graph placement and read-only mapped CPU access. It keeps the CPU multi-column prefill and all noneligible graph operations. The twelve-layer placement contains 82 matrices and 2,592,538,624 packed bytes; the five-layer placement contains 35 matrices and 1,482,883,072 packed bytes.

Full-checkpoint comparisons against the same forced CPU input trajectory passed with **zero bit mismatches** in all captured families:

| Placement | Output steps | Full logits | Ordered-route elements | Router elements | Captured KDA elements | Peak sampled RSS | Peak system swap growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Five-layer mapped tail | 4 | 619,520 | 3,024 | 108,864 | 5,013,504 | 5,883,641,856 B | 0 B |
| Twelve-layer mapped tail | 4 | 619,520 | 3,024 | 108,864 | 5,013,504 | 6,141,067,264 B | 0 B |
| Twelve-layer mapped tail, timed runtime before final alias hardening | 16 | 2,478,080 | 7,056 | 254,016 | 11,698,176 | 6,354,485,248 B | 0 B |

The sixteen-step result contains 672 ordered-route tensors, 672 router tensors and 1,088 KDA tensors. The result-output tensors also match bit for bit. The latest complete selective proof is `quant-tail12-mapped-fidelity16-timing/{experiment,comparison,guard}.json` in the raw evidence; it records SHA-256 for the probe and all nine actual native libraries, so a timing campaign can reject runtime drift. Its 89.385 s guarded duration includes diagnostic capture/serialization and is **not** a token-rate measurement. The earlier sixteen-step capture before the logging-only timing adjustment also passed, and remains a separate record.

The delivered patch later gained a stricter leaf-flag check at the sixteen-alias traversal boundary. That final runtime was compiled and passed the mapped-weight fixture's 20 checks, but its subsequent CPU64 diagnostic was stopped by the system swap guard after 18 outputs. No complete-model selective fidelity or speed qualification is transferred from the earlier library hashes to this final hardened build. The numerical evidence above remains valid for the explicitly recorded earlier binaries.

These results qualify the captured trajectory and original CPU arithmetic on this pinned ARM runtime. They do not qualify every prompt, long context, uncaptured recurrent state, EOS behavior or a decoding gain. An equality of generated IDs alone is not used as a substitute for these tensor comparisons. Model-integrity checks record unchanged file sizes and modification times, plus hashes of selected component bytes; no new full 110 GB model hash was computed.


## New full-model proof for the inline output-head path

The principal campaign subsequently completed two full-checkpoint diagnostics with `AIRLLM_METAL_Q8_CPU_ARITH=1`, head-only placement, and the shared-activation flag absent. The saved native comparisons pass **bit for bit**, including ordered routes, routers, captured KDA outputs, generated IDs and every full logit vector:

- Four output steps against the historical CPU4 capture: 619 520 identical logits. Peak sampled RSS 5 881 528 320 bytes; peak system swap growth 985 661 bytes, within the unchanged guard.
- Sixteen output steps against the new CPU10 profile capture: 2 478 080 identical logits. Peak sampled RSS 6 141 394 944 bytes; no sampled system swap growth.

The separately saved [q8_full_head_fidelity.json](q8_full_head_fidelity.json) records criteria, coverage, environments, guard outcomes and SHA-256 of the original comparison/guard/experiment reports. This is direct evidence from full-model execution on those short forced trajectories, beyond the 32-row synthetic-activation component proof. It does not qualify every prompt or context length, uncaptured recurrent state, sustained stability, EOS, or an end-to-end speed gain. Diagnostic serialization is excluded from throughput claims.

## Shared activation within an eight-row tile

The additive [llama-q8-shared-activation.patch](patches/llama-q8-shared-activation.patch) supplies a second native head kernel, selected only by `AIRLLM_METAL_Q8_SHARED=1` together with arithmetic mode 1 and the original output-head shape. It does not activate for mode 2 or for KDA projection weights. The existing inline kernel remains available without the new flag.

All 32 lanes first quantize the same 4 096 FP32 activations into static threadgroup arrays: 4 096 signed bytes and 128 decoded FP16 scales stored as FP32, exactly 4 608 bytes of threadgroup SRAM. After a threadgroup barrier, eight four-lane row groups reuse those activations while preserving the same even/odd ARM FMA and pairwise reduction order. No full floating-point weight matrix or global activation scratch is allocated. Static threadgroup storage requires no dynamic host allocation (`smem=0`). Native threadgroups stay at one SIMD group of 32 threads and eight output rows.

Partial row tiles must still dispatch all 32 lanes and must not return before the quantization barrier. The MLX variant therefore rounds its grid to complete eight-row tiles. The 13-row fixture tests this case, along with normal, zero and nearest-even tie activations.

The complete seven-test suite was rerun on the actual M5 and passed, including the shared 4 096-wide path. The new real-head 32-row probe also passes bit for bit against native ARM in all three variants, and the independently compiled shared Metal timestamp job produces identical results. Its [raw report](q8_shared_real_head32.json) distinguishes host synchronization wall times, command wall times and GPU timestamps. On this small case, shared activation did **not** establish a speed advantage: median GPU time was 0.2170 ms, versus 0.1225 ms inline, under their separately submitted phases. This does not predict the complete 154 880-row head.

The subsequent full-model head-only shared run (`head-q8-shared-fidelity4` in the raw evidence) passes all 619,520 logits, 3,024 ordered-route elements, 108,864 router elements and 5,013,504 captured KDA elements bit for bit over four output steps. Peak sampled RSS was 6,031,179,776 bytes, with zero sampled system swap growth. This is a short tensor-fidelity proof for the shared head, not a paired performance result, a sixteen-step shared-head proof or qualification of shared activation for mode 2.
