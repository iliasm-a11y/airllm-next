`chat_template.jinja` is the unmodified official GLM-5.3-Flash template from
`zai-org/GLM-5.3-Flash-BF16`, revision
`f9fed3bbc82291073373d67452e9f25317ef7577`:

https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/f9fed3bbc82291073373d67452e9f25317ef7577/chat_template.jinja

Its 10,830 bytes are identical to `tokenizer.chat_template` in the metadata
shard of `ggml-org/GLM-5.3-Flash-GGUF` Q2_K_S, revision
`119034bc3d9d76eb41f91508e6f9fb19fde42672`.

SHA256: `2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4`.

The runtime renders the template embedded in the actual checkpoint; this copy
is solely a reproducible test fixture. No tokenizer vocabulary, model weights,
vision projector or MTP checkpoint is included here.
