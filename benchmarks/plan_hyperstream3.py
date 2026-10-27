#!/usr/bin/env python3
"""Write separate frozen AB/BA ablation plans; never launch inference."""
import argparse
import hashlib
import json
from pathlib import Path


COMMON = ('--expert-batch', '--concat-tiles', '--packed-direct-io',
          '--bulk-metadata', '--zero-copy-weights', '--expert-pipeline',
          '--string-buffer-bytes', '65536', '--numerical-evidence')
ADAPTIVE = ('--hyper-prefill', '--hyper-prefill-max-tokens', '128')
CORE_FILES = ('air_llm/airllm/nextgen/gguf_qwen.py',
              'air_llm/airllm/nextgen/gguf_metal.py')
LOOKUP_TASK = 'research/hyperstream3/speculation-context.json'
GENERAL_LONG_TASK = 'research/m5_validation/general-long.json'
PROFILES = ('retained', 'router_reference16')


def apply_profile_jobs(experiments, candidate, profile):
    """Declare the correction only on the current candidate, including controls.

    Historical H1/H2 roots remain immutable. Their original workers do not
    receive a new option they never implemented. This changes planned execution
    inputs; it never relabels recordings from a previous campaign.
    """
    if profile not in PROFILES:
        raise ValueError('Unknown campaign profile: '+str(profile))
    if profile == 'retained':
        return experiments
    candidate = str(Path(candidate).expanduser().resolve())
    for experiment in experiments:
        for job in experiment['plan']:
            if job['engine_root'] != candidate:
                continue
            flags = job['args']
            if '--router-reference-chunk' in flags:
                if flags[flags.index('--router-reference-chunk')+1] != '16':
                    raise ValueError('router_reference16 requires its declared router chunk16')
            else:
                flags.extend(['--router-reference-chunk','16'])
    return experiments


def paired_jobs(baseline, candidate, baseline_flags=(), candidate_flags=(), *, pairs=3, tasks=None):
    if type(pairs) is not int or pairs < 1:
        raise ValueError('pairs must be a positive integer')
    roots = dict(baseline=str(Path(baseline).expanduser().resolve()),
                 candidate=str(Path(candidate).expanduser().resolve()))
    flags = dict(baseline=tuple(baseline_flags), candidate=tuple(candidate_flags))
    jobs = []
    for pair in range(pairs):
        for variant in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')):
            job = dict(id=f'{variant}-p{pair:02d}', engine_root=roots[variant],
                       args=[*COMMON, *flags[variant]])
            if tasks is not None:
                job['tasks'] = str(Path(tasks).expanduser().resolve())
            jobs.append(job)
    return jobs


def build_experiments(h1_root, h2_root, candidate, *, pairs=3, profile='retained'):
    """Return ordered plans and separate metadata accepted by no worker itself.

    Plan job objects contain only campaign_research51-supported fields. The
    combination prerequisite is metadata for the operator, not an enforcement
    mechanism in the existing campaign runner.
    """
    if profile not in PROFILES:
        raise ValueError('Unknown campaign profile: '+str(profile))
    experiments = []
    lookup_tasks = Path(candidate).expanduser().resolve() / LOOKUP_TASK
    def add(name, baseline_root, candidate_root, baseline_flags=(), candidate_flags=(),
            *, notes, speculative=False, requires=()):
        experiments.append(dict(name=name,
            status='unvalidated_requires_prior_pass' if requires else 'unvalidated',
            requires_prior_pass=list(requires), speculative=speculative, notes=notes,
            tasks_kind='lookup_synthetic_copy' if speculative else 'historical_364',
            plan=paired_jobs(baseline_root, candidate_root, baseline_flags,
                             candidate_flags, pairs=pairs,
                             tasks=lookup_tasks if speculative else None)))
    add('h1_vs_h2', h1_root, h2_root, (), ADAPTIVE,
        notes='H1 fixed default chunk16 versus H2 adaptive cap128; H2 real-checkpoint fidelity remains unvalidated.')
    for chunk in (16, 32, 64, 128):
        add(f'prefill{chunk}_vs16', h2_root, h2_root,
            ('--prefill-chunk', '16'), ('--prefill-chunk', str(chunk)),
            notes='Same H2 engine, fixed chunks; hyper-prefill is absent. Chunk16 is the repeatability control.')
    add('tensorops_vs_h2', h2_root, candidate, ADAPTIVE,
        (*ADAPTIVE, '--metal-batch-backend', 'tensorops'),
        notes='Isolated TensorOps dispatcher with runtime numerical/timing fallback; all calibration costs count.')
    for tile in (2, 4, 8, 16):
        add(f'reuse_tile{tile}_vs_h2', h2_root, candidate, ADAPTIVE,
            (*ADAPTIVE, '--metal-batch-backend', 'reuse', '--metal-token-tile', str(tile)),
            notes='Isolated packed weight register reuse at the declared token tile; same adaptive prefill cap.')
    add('grouped_transport_vs_h2', h2_root, candidate, ADAPTIVE,
        (*ADAPTIVE, '--grouped-direct-io'),
        notes='Isolated final grouped allocation/direct reads; individual expert tile cache is bypassed.')
    for depth in (2, 4, 8, 16):
        add(f'speculation_depth{depth}_vs_h2', h2_root, candidate, ADAPTIVE,
            (*ADAPTIVE, '--speculative-depth', str(depth)), speculative=True,
            notes='Lookup-friendly synthetic copying workload shared by both sides; not a general quality or workload speed claim. Grouped verification, snapshot/replay and adaptive fallback costs all count.')
    add('combined_tensorops_grouped_spec8', h2_root, candidate, ADAPTIVE,
        (*ADAPTIVE, '--metal-batch-backend', 'tensorops', '--grouped-direct-io',
         '--speculative-depth', '8'), speculative=True,
        requires=('h1_vs_h2', 'tensorops_vs_h2', 'grouped_transport_vs_h2',
                  'speculation_depth8_vs_h2'),
        notes='UNVALIDATED COMBINATION on the paired synthetic copying workload. Do not execute until every listed isolated experiment passes independent numerical and resource validation with measured benefit; gains do not generalize automatically.')
    if profile == 'router_reference16':
        names = {trial['name']:trial['name']+'_reference16' if trial['name'].endswith('_vs_h2')
                 else trial['name'] for trial in experiments}
        h2 = str(Path(h2_root).expanduser().resolve())
        current = str(Path(candidate).expanduser().resolve())
        for trial in experiments:
            trial['name'] = names[trial['name']]
            trial['requires_prior_pass'] = [names[name] for name in trial['requires_prior_pass']]
            trial['notes'] += ' Current-candidate routing uses contiguous reference16 score evaluation; expert order/weight pairing comes from each original16-row router call, without expert-ID sorting.'
            # Every H3 ablation compares two current-candidate executions with
            # the same correction. The first control alone retains historical
            # H1 as its baseline; original H2 remains recorded provenance.
            for job in trial['plan']:
                if job['engine_root'] == h2:
                    job['engine_root'] = current
        experiments.insert(1,dict(name='general_long_h1_vs_h2_reference16',status='unvalidated',
            requires_prior_pass=[],speculative=False,tasks_kind='general_long_noncopy',
            notes='Historical H1 versus corrected adaptive H2 on the same non-copy task, max_new_tokens=384. Report EOS/truncation and decode length; under64 emitted tokens is insufficient sustained-decode evidence.',
            plan=paired_jobs(h1_root,candidate,('--prefill-chunk','16'),
                             (*ADAPTIVE,'--prefill-chunk','16'),pairs=pairs,
                             tasks=Path(candidate)/GENERAL_LONG_TASK)))
    return apply_profile_jobs(experiments,candidate,profile)


def validate_root(root, extra_files=()):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Engine root must be an existing directory')
    for relative in (*CORE_FILES, *extra_files):
        if not (root / relative).is_file():
            raise ValueError(f'Missing required engine/protocol file: {root / relative}')
    # The campaign runner captures Git revision and full protocol again before
    # execution. These content hashes pin the planner's inspected Python tree.
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((root / 'air_llm/airllm/nextgen').rglob('*.py'))}
    return root, sources


def write_plans(h1_root, h2_root, candidate, output_dir, *, pairs=3, profile='retained'):
    if type(pairs) is not int or pairs < 1:
        raise ValueError('pairs must be a positive integer')
    h1_root, h1_sources = validate_root(h1_root)
    h2_root, h2_sources = validate_root(h2_root,
        ('air_llm/airllm/nextgen/hyperprefill.py', 'research/hyperstream2/context.json'))
    if profile not in PROFILES:
        raise ValueError('Unknown campaign profile: '+str(profile))
    candidate, candidate_sources = validate_root(candidate,
        ('air_llm/airllm/nextgen/hyperprefill.py', 'air_llm/airllm/nextgen/gguf_m5.py',
         'air_llm/airllm/nextgen/gguf_speculative.py', 'air_llm/airllm/nextgen/gguf_packed_io.py',
         LOOKUP_TASK, *([GENERAL_LONG_TASK] if profile == 'router_reference16' else [])))
    lookup_tasks = candidate / LOOKUP_TASK
    lookup_raw = lookup_tasks.read_bytes()
    lookup = json.loads(lookup_raw)
    if (not isinstance(lookup, list) or len(lookup) != 1 or not isinstance(lookup[0], dict)
            or not isinstance(lookup[0].get('id'), str) or not lookup[0]['id'].strip()
            or not isinstance(lookup[0].get('prompt'), str) or not lookup[0]['prompt'].strip()
            or type(lookup[0].get('max_new_tokens')) is not int or lookup[0]['max_new_tokens'] != 128
            or len(lookup[0]['prompt'].encode('utf-8')) > 900):
        raise ValueError('Lookup workload must be one task, a short nonempty prompt and max_new_tokens=128')
    experiments = build_experiments(h1_root, h2_root, candidate, pairs=pairs,profile=profile)
    # Do not resolve the final path component: an existing broken symlink must
    # fail exclusive creation, never redirect output into a new target.
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    index = dict(schema_version=1, status='plans_only_no_inference_launched', pairs=pairs,profile=profile,
        numerical_evidence=True,
        tasks_recommendation={'historical_364': str(h2_root / 'research/hyperstream2/context.json'),
                              'lookup_synthetic_copy': str(lookup_tasks)},
        task_sha256={'historical_364': hashlib.sha256((h2_root / 'research/hyperstream2/context.json').read_bytes()).hexdigest(),
                     'lookup_synthetic_copy': hashlib.sha256(lookup_raw).hexdigest()},
        inspected_source_sha256={'h1': h1_sources, 'h2': h2_sources, 'candidate': candidate_sources},
        engine_roots={'h1': str(h1_root), 'h2': str(h2_root), 'candidate': str(candidate)},
        experiments=[])
    if profile == 'router_reference16':
        index['tasks_recommendation']['general_long_noncopy'] = str(candidate/GENERAL_LONG_TASK)
        index['task_sha256']['general_long_noncopy'] = hashlib.sha256((candidate/GENERAL_LONG_TASK).read_bytes()).hexdigest()
    for ordinal, experiment in enumerate(experiments):
        name = f'{ordinal:02d}_{experiment["name"]}.json'
        with (output_dir / name).open('x') as file:
            json.dump(experiment['plan'], file, indent=2)
            file.write('\n')
        index['experiments'].append({**{key: value for key, value in experiment.items() if key != 'plan'},
                                     'plan_file': name, 'jobs': len(experiment['plan'])})
    with (output_dir / 'index.json').open('x') as file:
        json.dump(index, file, indent=2)
        file.write('\n')
    with (output_dir / 'SCHEMA.md').open('x') as file:
        file.write('''# HyperStream 3 — plans only

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
''')
        if profile == 'router_reference16':
            file.write('''
The explicit router_reference16 profile uses historical H1 only in the first
short comparison and the second general-long comparison (max_new_tokens=384).
Both sides of later ablations use the current candidate and declare router
chunk16. Matrix controls retain the frozen parser's default backend off and
omit the acceleration flag, so they do not require an accelerator kernel. The original H2 root
remains provenance; no historical checkout is changed or relabeled. Ordered
expert routes and full logits must pass the same independent checks. The
profile declares an experimental calculation change, not an accepted gain.
''')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--h1-root', type=Path, required=True)
    parser.add_argument('--h2-root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--profile',choices=PROFILES,default='retained')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        index = write_plans(args.h1_root, args.h2_root, args.candidate, args.output_dir, pairs=args.pairs,profile=args.profile)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=index['status'], experiments=len(index['experiments']),
                          pairs=index['pairs'], output_dir=str(args.output_dir.absolute()))))


if __name__ == '__main__':
    main()
