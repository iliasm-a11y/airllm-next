# Velocity evidence

`MANIFEST.json` identifies every raw numerical report and its lossless gzip.
Run `python benchmarks/archive_velocity_evidence.py --restore` from the repository
root to restore and verify the local JSON. A differing existing file is never
overwritten. No model weights, corpus text files or compiled binary is included.

Primary measured generation: `generation-metal.json`, 90 shuffled runs; separate
follow-ups: `long-metal.json` (12), `draft-metal.json` (24), `profile-cpu.json` (6,
instrumented). Source fingerprints and argument/budget/timing scope are inside
each report. `generation-analysis.json` contains the paired statistical analysis.
The source at the first campaign is preserved in GitHub commit `d66ce9f` (local
`574feec`, identical tree). Later Qwen adapter hardening does not alter the Granite
or draft code used by these experiments. Reports record all source-file hashes;
the initial HEAD alone is not the identity of uncommitted measured code.

`quality-expert8-metal.json` evaluates 512 fixed human labels. It measures
quality, not generation throughput. Q8 is lossy and changes some argmaxes.

`ollama-installed-qwen3-8b.json` is a successful **Ollama** system test.
`ollama-installed-smoke.json` emitted one token but failed safety requirements:
its original runner-name filter detected pressure without terminating the actual
llama-server soon enough. This defect and its swap growth are disclosed.
The three `llamacpp-installed-cpu*.json` use owned-PID monitors and all stop
before safe generation. The WILLNEED interposer's self-test is not model proof;
its real trial stopped before any hint was intercepted. No failed trial is
counted as a giant-model success, speedup, or sustained generation.

`installed-qwen36-blob-sha256.json` verifies all 23,938,321,664 bytes of the existing
GGUF. `qwen3-download.json` instead records a stopped, incomplete alternative
checkpoint download. A partial/preallocated shard is never a complete model.

`tests-final.log` is the final offline suite: 994 passed, 1 skipped, 11 subtests.
`tests-cpu.log` is an earlier checkpoint, retained separately. Guard JSON report
sampled RSS/reserve for native runs; reported cache bounds are not RSS measures.
See [the French report](../VELOCITY_REPORT_FR.md) for numerical scope and limits.
