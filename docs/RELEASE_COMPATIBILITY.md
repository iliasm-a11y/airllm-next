# Compatibility of 4.1.0rc1

Recognition, reduced numerical tests and generation with trained weights are
separate statuses. Metadata inspection cannot promote one to another.

| Configuration | Candidate evidence | Limit |
|---|---|---|
| Full trained SmolLM2-135M-Instruct, Next Llama FP32 CPU/Metal | Three prompts, nine 16-token generations per backend; independent reference; source-free pack parity | Small trained checkpoint; no broad task-quality benchmark |
| Bias-free SiLU Llama, default full-head RoPE, floating weights | Tied/untied logits, suffixes, cached decode, interruption, dispatch and budgets | Other trained checkpoints require their own validation |
| GLM `glm_moe_dsa`, floating/block E4M3 FP8 | Reduced CPU/Metal logits/routes/KV/generation, affine sidecars and rollback | Full GLM-5.3 755+ GB payload not generated |
| Floating Qwen `qwen3_5_moe_text` | Experimental DeltaNet/convolution/gated attention/MoE; CPU/Metal SiLU/swish vs Transformers; reduced source-free generation | Full Qwen3.8 unvalidated; explicit experimental backend |
| Full Qwen3.8 BF16 | Official metadata/header inventory retained | About 4.89 TB including MTP; no certified full-pack compression ratio |
| NVIDIA ModelOpt/NVFP4 | Explicit early rejection | Different scales/layout/kernels from affine INT4 |
| Dynamic FP8 activation-server semantics | Not reproduced | Weight-decoding parity does not certify server-kernel quality |
| MTP, stochastic speculation, trees | Not implemented | Mathematical experiments are not executable capability |
| Other architectures/RoPE/bias variants | Early rejection unless a separate upstream/Ultra backend supports them | Upstream support does not imply Next support |

Giant sizes are from the preserved official-header inventory and independently
recomputed metadata in `research/exact_io/`. No GLM/Qwen giant shards were
downloaded during this finalization.
