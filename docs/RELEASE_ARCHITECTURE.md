# Candidate architecture

GLM DSA, Llama and experimental Qwen have separate computation and state
semantics. Local architecture JSON is checked before model execution. Next
dispatches Llama locally; Qwen remains explicit.

`open_checkpoint` selects the offset reader or a complete standalone pack.
Partial overlays wrap only once. Offset/shape/dtype validation precedes bounded
row/expert reads; grouped reads account for gaps and backing allocations.
Concurrent close cannot reopen an offset descriptor or close a borrowed pack
descriptor. Conversion commits publish only completed files/manifests.

`NativeWeights` shares selective embedding rows, expert slices, output tiles,
FP8 decoding and optional affine sidecars. `SingleFlight` reserves asynchronous
work and deduplicates requests; `WeightCache` supplies four explicit policies.
Historical routes may inform prefetch but never change the target's expert set.
Evaluated MLX intermediates release prior lazy graphs and weight references.

GLM holds latent KV and DSA selections. Llama uses causal GQA, full-head
non-interleaved RoPE, ordinary RMS weights and SiLU MLPs. Qwen holds convolution,
DeltaNet recurrent state and gated full-attention KV. Its `swish` DeltaNet RMS
gate is SiLU; its full-attention output gate remains sigmoid.

Strict inputs precede int32 narrowing. Requested logits spanning several prefill
chunks are collected without dropping positions. Attention/output scratch
estimates reject oversized work. Sessions have owners and context/budget guards.
GLM/Llama append-only caches truncate discarded tails; Qwen recurrent state
requires real snapshots. These rollback mechanisms are not interchangeable.

Lossless packs contain `weights.bin`, constant-sized `blocks.idx` records and a
committed manifest. Raw, zlib, shuffled-zlib and palette16 blocks reconstruct
original bytes. Raw fallback avoids expansion. Decoded sizes and checksums are
bounded before allocation. Standalone manifests include every tensor layout
and bounded hashed local config/tokenizer assets. Escaping payload symlinks,
corrupt assets and untrusted library paths are rejected. Native palette libraries
are supplied explicitly, never selected by a checkpoint manifest.

The verifier commits target-accepted tokens and correction/bonus tokens, restores
the committed KV prefix and separates logical, encoded and draft traffic.
Depth changes depend on observed cost. Known-prefix prompt lookup has bounded
search. Algorithmic exactness is distinct from floating batch invariance.

The HTTP server owns no remote inference client. It validates model IDs, JSON,
context, tokenizer results and output sequences, binds to loopback, and serializes
inference on its owning MLX stream/thread. Streaming/tools/vision are absent.

Experiments, archive and historical evidence remain outside production modules.
No custom Metal/C++ inference kernel is added without an established full-path
benefit; native MLX kernels remain the maintained calculation path.
