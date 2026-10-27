# HyperStream 3 — plans only

Each numbered JSON is a separate `campaign_research51.py --plan` input: a
nonempty list of jobs with `id`, `engine_root`, `args` and optional `tasks`. Pair IDs are
`baseline-p00` / `candidate-p00`; order alternates AB/BA between pairs.
`index.json` is metadata, never a campaign input. No process is launched here.

Use the identical full checkpoint, frozen paired task file, original precision,
resource limits and source trees within each comparison. Historical/prefill and
isolated matrix/transport trials retain the 364-token historical task supplied
with `--tasks`. Speculative and combined plans set `tasks` explicitly on BOTH
baseline and candidate jobs to the candidate's frozen synthetic copying file,
with max_new_tokens=128. This lookup-friendly workload is not a general quality
or workload speed benchmark. Do not pool results across these distinct tasks.
The worker still checks actual token/context limits; the planner only bounds
prompt bytes and does not initialize a tokenizer. The campaign runner
recaptures source/protocol hashes and revisions. Independently record complete
checkpoint hashes before and after admitted runs. Planner hashes are not a
checkpoint integrity check. Three pairs are a pilot; ten pairs provide more
paired observations, without guaranteeing statistical or numerical acceptance.

The H2 independent full-checkpoint verifier handles non-speculative pairs.
Speculative plans require the H3 speculative verifier, including every checked
token, full logits, routes, positions, EOS, rollback/replay evidence, guards and
total costs. H2's one-forward-per-emitted-token schema is insufficient for a
grouped speculative verification. A passing plan-generation test is neither
full-checkpoint equivalence nor a speed measurement.

Run isolated plans first. The final combination is UNVALIDATED and requires
the `requires_prior_pass` experiments in the index to pass independent fidelity,
resource and measured-benefit checks. The current campaign runner does not
enforce this prerequisite; do not run that plan before checking the evidence.
Calibration, compilation, draft proposal, rollback, accepted-prefix replay,
expert preparation and synchronization belong to request latency. Preserve
raw outputs; corroborate timing separately without numerical instrumentation.
