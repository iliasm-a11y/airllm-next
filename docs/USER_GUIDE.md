# Candidate user guide

## Installation and CPU validation

Tested: macOS 27.2 arm64, Apple M5 16 GB, Python 3.12.13, MLX 0.32.4,
Torch 2.14.1 and Transformers 5.20.2. See the observed requirements and fresh
installation reports under `evidence-final/`.

```bash
AIRLLM_PYTHON=python3.12 bash scripts/setup_next_macos.sh
source .venv-next/bin/activate
export MLX_ENABLE_TF32=0
airllm-next doctor
```

Alternatively install the built candidate with dependencies:

```bash
python3.12 -m venv .venv-next
.venv-next/bin/python -m pip install -c requirements-release-macos.lock.txt 'dist-final/airllm-4.1.0rc1-py3-none-any.whl[next]'
.venv-next/bin/python -m pip check
```

For Linux CPU, create a Python 3.12 environment, install
`requirements-next-linux.txt`, then `python -m pip install --no-deps -e ./air_llm`.
Linux uses MLX CPU rather than the macOS Metal distribution. Current clean-install
evidence is macOS; earlier Linux evidence is preserved separately.

```bash
MLX_ENABLE_TF32=0 AIRLLM_TEST_DEVICE=cpu python -m pytest tests_next -q
MLX_ENABLE_TF32=0 python -m pytest tests_ultra -q
python -m pytest research/exact_io/tests -q
python -m pytest air_llm/tests -q
```

On Mac repeat Next tests with `AIRLLM_TEST_DEVICE=gpu`. Some dedicated tests
explicitly select Metal even when general fixtures choose CPU. Suites build
random reduced checkpoints without downloads.

## Local generation and budgets

Set `MODEL` to an absolute local checkpoint directory. Inspection reads metadata;
preparation validates its local tensor layout. Neither command certifies quality.

```bash
airllm-next inspect "$MODEL" --tokens 4096
airllm-next prepare "$MODEL"
airllm-next run "$MODEL" --backend llama --device gpu --float32 \
  --prompt 'The capital of France is' --max-new-tokens 16 \
  --cache-mib 128 --io-mib 32 --kv-mib 32 --workspace-mib 128 \
  --reserve-mib 512 --max-tensor-mib 32 --row-chunk 2048 \
  --json-report generation.json
```

Use `--backend glm` for the GLM adapter and `--backend qwen-experimental` for
the floating hybrid research backend (greedy only). Unsupported NVFP4 is rejected.
`--token-ids 1,8,9` permits numerical tests without a tokenizer.

Cache, I/O, KV, workspace and reserve use MiB. Managed data is bounded; metadata,
interpreter, native allocator caching and OS page cache add memory beyond those
settings. Context, allocation and pressure guards can refuse a request. Default
automatic configuration preserves up to 4 GiB for the system; explicit smaller
reserves must suit the actual available memory.

Choose `--cache-policy lru/slru/lfu/cost`, `--no-prefetch`, or optional
`--coalesce-reads` explicitly. `tune MODEL` measures local read alternatives;
warm OS cache can dominate. `calibrate MODEL --backend llama ...` performs real
generations, so budget its time before using it on a giant checkpoint.

## Speculation

Append `--prompt-lookup --speculative-depth 4` to GLM/Llama `run` to propose
bounded spans from already known context. Alternatively use
`--draft-model /local/compatible/draft`. Architecture/token mappings must match.
The target checks every proposal. Observed acceptance/cost drives adaptation;
`--no-adaptive-speculation` fixes depth. Supply the same memory controls as for
ordinary generation.

This is greedy draft-and-verify, not stochastic correction or MTP. Floating
block arithmetic can change a near-tied argmax; tested sequence equivalence is
not a universal bitwise trajectory proof. Keep speculation off when the ordinary
token-by-token numerical trajectory is the reference contract.

## Lossless storage and quantization

Use a new destination and an explicit source-payload allowance:

```bash
airllm-next pack "$MODEL" --output /local/new-pack --standalone \
  --max-total-bytes 536870912 --block-bytes 1048576 --codec auto
airllm-next inspect /local/new-pack
airllm-next run /local/new-pack --backend llama --device gpu \
  --prompt 'The capital of France is' --max-new-tokens 8
```

A complete pack includes every tensor plus bounded hashed config/tokenizer
assets. It needs no original shards and copies no checkpoint Python code.
Per-block hashes remain enabled. Incompressible blocks stay raw. The manifest
commits only after data/index/assets; interrupted creation without a manifest
is incomplete. Retry into a new directory. The base checkpoint is never changed.

Without `--standalone`, the result is an overlay requiring its base checkpoint;
use `--lossless-dir`. Compression is opt-in because the measured warm path is
slower despite smaller files. Measure actual complete inference on your storage.

`convert MODEL --output NEW --bits 4 --group-size 64 --max-matrices 10`
creates an initially bounded lossy affine sidecar. Conversion has resumable
commits. `verify MODEL --quantized-dir NEW` explicitly hashes its contents.
Use `--quantized-dir NEW` at generation. Original weights remain required;
quantization quality needs full trained-model evaluation. INT4 is not NVFP4.

## Local OpenAI-shaped API

```bash
airllm-next serve "$MODEL" --backend llama --device gpu --port 8080 \
  --max-new-tokens 128 --model-name local-model \
  --cache-mib 128 --io-mib 32 --kv-mib 32 --workspace-mib 128 --reserve-mib 512
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":16,"temperature":0,"stream":false}'
```

`POST /v1/completions` accepts one text prompt. Chat uses the local tokenizer's
template. Supported subset: non-streaming, `n=1`, text only. Responses API,
tools, vision, authentication and remote binding are absent. Requests are
serialized and body/context/generation bounded; invalid requests and memory
exhaustion return explicit errors. This loopback server is not an Internet service.

## Reproduce experiments

```bash
python benchmarks/validate_release_model.py --model-dir "$MODEL" \
  --baseline-source /local/preserved/research-tree --output trained.json
python benchmarks/benchmark_lossless_checkpoint.py "$MODEL" \
  --max-total-bytes 536870912 --output lossless.json
python benchmarks/verify_standalone_generation.py "$MODEL" /local/new-pack \
  --backend next-llama --device gpu --output exact-pack.json
python benchmarks/benchmark_ooc.py --cache-state warm --output io.json
```

Pinned trained reference: HuggingFaceTB/SmolLM2-135M-Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The explicit bounded download example
`benchmarks/validate_pretrained.py --download-dir NEW_DIRECTORY` validates the
older Torch Ultra path; use the release-model script for Next CPU/Metal.
The pinned [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md)
declares Apache-2.0 weights. No command implicitly downloads giant models.

Null physical bytes, energy, utilization and temperature fields remain unknown.
Logical file bytes are not SSD/NAND traffic.
