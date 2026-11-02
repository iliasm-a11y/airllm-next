# Q2_K packed Metal experiment

The main native runtime already has a fused Metal Q2_K matvec and MUL_MAT_ID.
This experiment is justified as an arithmetic isolation path, not as a claim
that fused quantized multiplication is missing from llama.cpp.

The pinned GGML CPU Q2_K dot consumes temporary Q8_K activation blocks. The
pinned Metal Q2_K matvec consumes float32 activations. Consequently, the same
GGUF weights and same input float32 vector need not produce the same outputs.
Changing only floating point reduction order cannot remove that difference.
This static observation does not attribute the whole-model historical error
0.5348455906 to one operator: other quantized types, KDA, mHC and graph placement
must be isolated too.

`quantized.py` reads original Q2_K bytes inside a Metal kernel and never
materializes a floating point weight matrix. `cpu_q8k` first produces the
CPU reference's Q8_K temporary activations on GPU, then accumulates exact
32-bit integer group dots and minimum terms and reproduces the ARM NEON block
order with explicit float32 fused multiply-add operations. This does **not**
requantize the weights. The diagnostic `float` mode retains the float32 input
and is compared against native dequantization separately. The two modes have
different arithmetic contracts.

The first actual M5 probe read only 32 rows of one real 2048×4096 gate expert
(43,008 Q2_K bytes), with a seeded synthetic activation. Q8_K activation bytes
and scales and all 32 output values of `cpu_q8k` matched the pinned native ARM
CPU dot bit for bit. The float mode differed from the same CPU dot by at most
0.031523942947387695 on this sample. This is local operator evidence, not
whole-model logits/routes parity, a real captured model activation, an expert
cache speed measurement or a GLM token/s gain.

Admission checks run before the device import. They charge host/device packed
weights, activation buffers and output buffers. Matrix dimensions and file
reads are bounded. Calls synchronize before returning; asynchronous I/O and
GPU dispatch lifetime accounting belong to the separately bounded cache
caller. Payload accounting is not a process RSS ceiling: the probe also records
MLX peak allocations and process peak RSS. MLX's memory-limit setting is a
guideline, not a hard process memory cap.

The microbenchmark records each wall time, including activation import and
synchronization. The native reference is called once per row through ctypes
on one caller thread. The packed matrix is prepared and warm. These numbers
must never be presented as CPU10 whole-model gains or a Fanxiang SSD result.

Reproduce from the repository root with an installed MLX/NumPy environment:

```sh
python -m research.glm53_gpu.quantized_probe \
  --gguf /path/to/GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf \
  --libraries /path/to/pinned-native-runtime/bin \
  --tensor blk.3.ffn_gate_exps.weight --expert 0 --rows 32 \
  --output /path/to/quantized-q2k-real-32.json
```

The library directory must contain the pinned `libggml-base.dylib` and
`libggml-cpu.dylib`; the report hashes both oracle libraries. No model server is
started and no production configuration is changed. Independent native/Metal
tests are enabled by `AIRLLM_GGML_LIBRARIES` in addition to installing pytest.

The first full-expert microprobe subsequently checked all 2,048 gate outputs:
CPU-compatible mode again matched the native ARM dot bit for bit, with a
5,558,336-byte charged payload and 68,747,264-byte process peak RSS. Raw evidence
is in `evidence/quantized-q2k-real-expert2048.json`. The three optional GPU test
widths 256/512/4096, plus four portable/native tests, passed (seven tests total).

For a CPU reference without a Python call per row, compile the optional adapter:

```sh
clang++ -std=c++17 -O3 -dynamiclib \
  research/glm53_gpu/quantized_native_oracle.cpp \
  -L/path/to/pinned-native-runtime/bin -lggml-cpu \
  -Wl,-rpath,/path/to/pinned-native-runtime/bin \
  -o /path/to/libq2k-oracle.dylib
```

Pass `--cpu-helper /path/to/libq2k-oracle.dylib` to the probe. It reports both
CPU1 and CPU10 whole-matrix timings, using the same pinned native dot. CPU10
uses a persistent ten-worker pool, so creation is excluded and dispatch and
activation quantization are included. This still differs from the production
GGML graph scheduler and is a component measurement.

The final native whole-matrix pilot, with five recorded repetitions, gave:

| Prepared 2048×4096 real gate matrix | Median time |
|---|---:|
| Native CPU1, one whole-matrix adapter call | 0.454 ms |
| Native CPU10, persistent worker pool | 0.553 ms |
| GPU CPU-compatible Q2_K×Q8_K | 1.309 ms |
| GPU float activation diagnostic | 1.039 ms |

Thus this experimental GPU implementation was slower than the native CPU
component reference. Earlier per-row Python CPU measurements cannot support a
speedup claim. The CPU10 dispatch samples vary considerably (1.999 ms first,
then 0.881/0.553/0.544/0.533 ms); worker scheduling, run order and tiny sample
size limit this component pilot. It is not the production GGML CPU10 scheduler.
All 2,048 outputs of GPU CPU-compatible mode and both native matrix adapters
matched the pinned ARM reference bit for bit. The final report is
`evidence/quantized-q2k-real-expert2048-native-matrix.json`; eight tests passed,
including a regression that rejects nonfinite or incorrect compatible output
in the probe's exit status. None of these measurements establishes a GLM
whole-model speed improvement.
