# Local text API

`airllm.nextgen.server` exposes a deliberately small OpenAI-compatible subset.
The service calls the already loaded local `AirLLMNextModel` and its local
tokenizer. It contains no inference client, remote provider or implicit model
download. It accepts only loopback addresses, serializes inference requests,
limits JSON bodies to 1 MiB by default, and checks the requested completion
against the configured context and token limits.

```python
from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.server import serve

# Construct the MLX model and run the serving loop on the same thread.
# MLX stream handles must not be transferred to another Python thread.
with AirLLMNextModel('/Volumes/Models/local-checkpoint') as model:
    serve(model, model_name='local-glm', host='127.0.0.1', port=8080,
          max_new_tokens=512)
```

`make_server(model, **options)` returns the underlying `HTTPServer` for callers
that manage shutdown themselves. The caller owns the model's lifetime. This
is a local development service without authentication or TLS; the binding
restriction is enforced even if `host='0.0.0.0'` is requested. It does not
enable browser cross-origin access.

Supported endpoints:

| Endpoint | Supported request |
| --- | --- |
| `GET /v1/models` | Lists the one locally loaded model ID. |
| `POST /v1/completions` | `model`, one text `prompt`, `max_tokens`, `temperature`, `stream=false`, `n=1`. |
| `POST /v1/chat/completions` | Same options, with text `messages` instead of `prompt`; roles `system`, `user`, `assistant`. |

Chat requests use the checkpoint tokenizer's actual chat template. A missing
template is an explicit error; the service does not invent a prompt format.
Streaming, tool calls, structured output, batching, logprobs, stop strings,
multimodal inputs and extra sampling controls are not implemented. Requests
using unsupported fields fail with an OpenAI-shaped JSON error rather than
having those fields silently ignored. This is sufficient for text clients;
agents requiring OpenAI tool calling need additional implementation.

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-glm","prompt":"Bonjour","max_tokens":16,"temperature":0}'
```

The HTTP integration tests include a real miniature GLM forward/generation
comparison against a direct call to the same engine. The tokenizer used by that
test is a fixed token mapping for transport verification. This validates the
local API path, not pretrained language quality, complete model compatibility,
Metal performance, or throughput of large checkpoints.
