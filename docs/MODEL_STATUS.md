# Architecture status and checkpoint provenance

Checked on 2026-10-22. Recognition, numerical primitive tests, miniature
generation and full-checkpoint generation are separate validation levels.
This delivery does not certify full GLM-5.3 or Qwen3.8 inference on an M5.

| Architecture | Executable Next backend | Full published weights | Metal/M5 | MTP |
|---|---|---|---|---|
| `glm_moe_dsa` | Existing native MLX GLM path, miniature validation | Not validated | Not validated here | Not executed |
| `qwen3_5_moe_text` | Explicit experimental floating, sigmoid-gate hybrid backend; real miniature generation | Full Qwen3.8 checkpoint unsupported/unvalidated | Not validated | Declared weights detected as metadata; not executed |
| Other architectures | No adapter in this registry | Existing upstream/Ultra paths remain separate | Depends on backend | No claim |

`get_adapter(config).require_runtime(config)` rejects unsupported architectures
before creating the Next model. Qwen's hybrid blocks cannot be run as GLM DSA
or as ordinary Llama attention. A local metadata inspection is executable with:

```sh
PYTHONPATH=air_llm python -m airllm.nextgen.adapters /path/to/checkpoint --tokens 4096
```

This reads JSON only. Its `generation_validated` flag is always false; an
index cannot establish a generation result, shard integrity or physical I/O.

The distinct experimental Qwen constructor is available explicitly:

```python
from airllm import AutoModel
from airllm.nextgen.config import NextConfig
with AutoModel.from_pretrained('/local/floating-qwen-checkpoint',
        backend='next-qwen-experimental',
        next_config=NextConfig(load_tokenizer=False)) as model:
    print(model.generate([[1, 8, 12]], max_new_tokens=8).tolist())
```

This path implements Gated DeltaNet projections, causal convolution, normalized
FP32 recurrence, zero-centered normalization, full gated attention/RoPE and exact
router-selected expert streaming through the generic reader/cache. It accepts
F32/F16/BF16 tensors, unpadded batch one, default RoPE and greedy generation.
The independent Transformers tests use a four-layer random miniature model,
including three linear and one full-attention block. Prefill/cached logits,
eight-token greedy sequences, chunked prefill, rollback and injected-failure
recovery are verified. BF16 is separately tested with explicit finite numerical
tolerances. The raw numerical/actual-token results are reproducible with
`benchmarks/validate_qwen_runtime.py`.

The official Qwen3.8 config declares `output_gate_type='swish'`, while the
inspected Transformers reference uses sigmoid without reading that field.
This backend explicitly rejects a swish declaration until its semantics and
checkpoint parity are independently validated. ModelOpt/NVFP4 and requests for Qwen MTP execution
are unsupported. Auxiliary MTP weights may exist unused during ordinary decoding. `experimental_runtime` in the registry advertises the
constructor; it does not enable the full-checkpoint `runtime_available` flag.

## Official checkpoint inventory

GLM source: [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
revision `ba32f8c19c8738e2c57c71798c601d477a7bf844` (API lookup).
Its config identifies `glm_moe_dsa`, 78 layers, 256 experts with 8 routed experts
per token and block E4M3 FP8 weights. The archived index declares
755,617,140,416 tensor bytes across 141 shards; this index was preserved from
the input archive, and its full current index could not be retrieved during
this audit. The live revision/config were checked, not all weights.
The FP8 path decodes stored weights for arithmetic; this does not reproduce
the checkpoint's dynamic FP8 activation-quantization kernels. Tests against a
dequantized reference do not establish parity with a production FP8 serving
kernel. Quality and numerical error still require a matched full-model test.

Qwen source: [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B),
revision `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`.
The [official index](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/model.safetensors.index.json)
declares **4,892,365,451,008 bytes** of tensor payload (213 shards). This BF16
checkpoint exceeds a 2 TB disk. No tensor shards were downloaded.
Its text config has 92 layers, following 23 groups of three Gated DeltaNet
blocks and one gated full-attention block, plus 512 experts (top 10 and a
shared expert). One MTP layer is declared. The hosted Qwen3.8-Max service has
additional features; this open text checkpoint is not that complete service.

The published [NVIDIA NVFP4 variant](https://huggingface.co/nvidia/Qwen3.8-2.4T-A95B-NVFP4)
was inspected without downloading weights. Its config uses ModelOpt, mixed
NVFP4 experts (groups of 16), FP8 projections and unquantized components.
It is **not MLX affine INT4**. The model card targets NVIDIA Blackwell and
Linux; no Metal compatibility follows from that. The card reports ModelOpt
0.46.0 while the fetched config's producer field reports 0.42.0; preserve this
discrepancy rather than silently choosing one. The complete NVFP4 index was
unavailable to this audit, so exact disk fit on 2 TB is not certified. A label
such as “1.3T params” on packed tensors is not an exact byte-size report.

## Recurrent state and speculative rollback

`nextgen/recurrent.py` implements the FP32 Gated DeltaNet recurrence as a
portable numerical reference. It is tested against the pinned official
Transformers recurrence at revision
`a672eb5f474b8b30173e9f96eab66adf292da635`, with normalized/un-normalized Q/K,
initial state, grouped heads, and split-versus-whole sequence execution. The
upstream function and Apache-2.0 attribution are retained under
`evidence-ooc/sources/qwen_recurrent_reference.py`. These primitive results
do not establish Qwen logits or generation parity.

With the official Qwen dimensions, FP32 recurrent payload alone is
`69 * 128 * 128 * 128 * 4 = 578,813,952` bytes. At a two-byte compute dtype,
the full-width convolution state adds 11,304,960 bytes; 4,096 tokens of full
attention K/V add 385,875,968 bytes. These are structural payload estimates,
excluding workspace, allocation slack and any rollback copies. Recurrent
state is overwritten at each token; changing a position counter cannot roll
it back. The reference explicitly snapshots both recurrent and convolution
banks and enforces a caller-supplied snapshot budget. A production Qwen
speculative backend must use bounded snapshots or recomputation and validate
its state transactions before enabling verification.

Remaining Qwen3.8 implementation: verify swish-gate semantics against a suitable
independent reference, mixed ModelOpt decoding and Metal kernels, native fusion
and measured performance, MTP loading/semantics, snapshot-aware speculative
integration, full-model numerical and quality comparison, then real Apple
hardware profiling. The runtime currently realizes recurrence token steps to
bound lazy graph lifetime; no high-performance native-kernel claim is made.
Its state snapshots are immutable array-bank versions, with retained states
charged conservatively to the context budget. Offset-only recurrent rollback
and unsupported top-k boundary ties are rejected instead of changing outputs.
