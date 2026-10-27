"""CPU-only checks for research evidence; never import or launch a model."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks'))
    benchmark = importlib.import_module('benchmark_research51')
    campaign = importlib.import_module('campaign_research51')
    return benchmark, campaign


def task(task_id='one', prompt='Calcule 2 + 2.'):
    return dict(id=task_id, prompt=prompt, max_new_tokens=8)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def evidence_pair():
    sources = {'air_llm/airllm/nextgen/gguf.py': 'engine-digest'}
    protocol = {'benchmark_research51.py': 'harness-digest', 'tasks.json': 'task-digest'}
    tasks = [task()]
    evidence = dict(
        status='completed', source_unchanged=True, model_stat_unchanged=True,
        source_sha256=deepcopy(sources), protocol_sha256=deepcopy(protocol),
        protocol_unchanged=True,
        turns=[dict(task(), status='completed', stop_reason='eos',
                    generated_ids=[7, 99], eos_token_ids=[99])],
    )
    guard = dict(
        exit_code=0, violation=None, sample_count=2,
        limits=dict(rss_bytes=2 << 30, reserve_bytes=3 << 30,
                    swap_growth_bytes=128 << 20, admitted_payload_bytes=768 << 20,
                    timeout_seconds=600., interval_seconds=.05),
    )
    return evidence, guard, sources, protocol, tasks


def validate(campaign, evidence, guard, sources, protocol, tasks):
    return campaign.validate_run_evidence(evidence, guard, sources, protocol, tasks)


def test_load_tasks_preserves_the_actual_prompt(tmp_path, harness):
    benchmark, _ = harness
    tasks = [task(), task('two', 'Réponds en français.\nDeuxième ligne.')]
    path = write_json(tmp_path / 'tasks.json', tasks)
    assert benchmark.load_tasks(path) == tasks


def test_task_snapshot_hashes_the_same_bytes_that_were_validated(tmp_path, monkeypatch, harness):
    benchmark, _ = harness
    initial = json.dumps([task()]).encode()
    changed = json.dumps([task('changed')]).encode()
    reads = []

    def read_bytes(path):
        reads.append(path)
        return initial if len(reads) == 1 else changed

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    tasks, digest = benchmark.load_task_snapshot(tmp_path / 'tasks.json')
    assert tasks == [task()]
    assert digest == hashlib.sha256(initial).hexdigest()
    assert len(reads) == 1


def test_protocol_drift_or_deleted_input_is_rejected(tmp_path, harness):
    benchmark, _ = harness
    script = tmp_path / 'harness.py'
    tasks = write_json(tmp_path / 'tasks.json', [task()])
    script.write_text('original harness', encoding='utf-8')
    protocol = benchmark.protocol_manifest([script, tasks])
    assert benchmark.protocol_unchanged(protocol)
    script.write_text('changed harness', encoding='utf-8')
    assert not benchmark.protocol_unchanged(protocol)
    script.write_text('original harness', encoding='utf-8')
    tasks.unlink()
    assert not benchmark.protocol_unchanged(protocol)


@pytest.mark.parametrize('value', [
    [], {}, None, 'not a task list', [None], ['prompt'],
    [dict(prompt='Question', max_new_tokens=8)],
    [dict(task(), id='')], [dict(task(), id=1)],
    [dict(task(), prompt='')], [dict(task(), prompt=None)],
    [dict(task(), max_new_tokens=True)], [dict(task(), max_new_tokens=False)],
    [dict(task(), max_new_tokens=0)], [dict(task(), max_new_tokens=-1)],
    [dict(task(), max_new_tokens=1.)], [dict(task(), max_new_tokens='8')],
    [dict(id='one', prompt='Question')], [task(), task()],
])
def test_load_tasks_rejects_invalid_or_ambiguous_work(tmp_path, harness, value):
    benchmark, _ = harness
    path = write_json(tmp_path / 'tasks.json', value)
    with pytest.raises((ValueError, TypeError)):
        benchmark.load_tasks(path)


def test_job_arguments_allow_the_supported_comparison_controls(harness):
    _, campaign = harness
    arguments = [
        '--rows', '8192', '--cache-policy', 'lru', '--io-mode', 'pread_nocache',
        '--timeout', '600', '--expert-batch', '--concat-tiles', '--packed-copy-free',
        '--packed-direct-io', '--bulk-metadata', '--zero-copy-weights',
        '--expert-pipeline', '--string-buffer-bytes', '65536',
    ]
    assert campaign.validate_job_args(arguments) == arguments
    assert campaign.validate_job_args([]) == []


@pytest.mark.parametrize('arguments', [
    ['--worker'], ['--work'], ['--wor'],
    ['--model', 'other.gguf'], ['--tasks', 'other.json'],
    ['--output', 'other.json'], ['--engine-root', '/other/engine'],
    ['--label', 'forged'], ['--engine', '/other/engine'],
    ['--row', '8192'], ['--cache-p', 'lru'], ['--io-m', 'pread'],
    ['--time', '600'], ['--expert-b'], ['--packed-direct'],
    ['--unknown'], ['--profile'], ['--rows'], ['--rows', '--worker'],
    ['--string-buffer-bytes', '-1'], ['--string-buffer-bytes', '65537'],
    ['--string-buffer', '65536'], ['--expert-p'],
    '--worker', None, [None],
])
def test_job_arguments_cannot_override_ownership_or_abbreviate_flags(harness, arguments):
    _, campaign = harness
    with pytest.raises((ValueError, TypeError)):
        campaign.validate_job_args(arguments)


def test_complete_output_requires_successful_independent_guard(harness):
    _, campaign = harness
    assert validate(campaign, *evidence_pair()) is None


def frozen_identity_pair():
    evidence, guard, sources, protocol, tasks = evidence_pair()
    command = ['python', 'benchmark_research51.py', '--label', 'candidate']
    evidence.update(git_head='frozen-head', git_head_after='frozen-head', engine_head_unchanged=True)
    guard['command'] = [*command, '--worker']
    expected = dict(expected_head='frozen-head', expected_command=command,
                    expected_limits=deepcopy(guard['limits']))
    return evidence, guard, sources, protocol, tasks, expected


def test_campaign_identity_command_and_limits_are_bound_to_the_evidence(harness):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks, expected = frozen_identity_pair()
    assert campaign.validate_run_evidence(evidence, guard, sources, protocol, tasks, **expected) is None


@pytest.mark.parametrize('record,field,value', [
    ('evidence', 'git_head', 'other-head'),
    ('evidence', 'git_head_after', 'other-head'),
    ('evidence', 'engine_head_unchanged', False),
    ('guard', 'command', ['python', 'unrelated.py', '--worker']),
    ('guard', 'limits', {'rss_bytes': 64 << 30}),
    ('guard', 'exit_code', False), ('guard', 'sample_count', True),
])
def test_git_drift_unrelated_guard_or_relaxed_limits_cannot_certify_a_run(harness, record, field, value):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks, expected = frozen_identity_pair()
    target = evidence if record == 'evidence' else guard
    target[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        campaign.validate_run_evidence(evidence, guard, sources, protocol, tasks, **expected)


@pytest.mark.parametrize('field,value', [
    ('status', 'started'), ('source_unchanged', False),
    ('model_stat_unchanged', False), ('source_sha256', {'other.py': 'digest'}),
    ('protocol_sha256', {'other.py': 'digest'}), ('protocol_unchanged', False),
    ('turns', []),
])
def test_changed_or_incomplete_evidence_is_rejected(harness, field, value):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    evidence[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


@pytest.mark.parametrize('field', [
    'status', 'source_unchanged', 'model_stat_unchanged', 'source_sha256',
    'protocol_sha256', 'protocol_unchanged', 'turns',
])
def test_missing_evidence_fields_fail_closed(harness, field):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    del evidence[field]
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


@pytest.mark.parametrize('field,value', [
    ('id', 'other'), ('prompt', 'A different prompt'), ('max_new_tokens', 9),
    ('status', 'started'), ('stop_reason', 'max_new_tokens'),
    ('generated_ids', []), ('generated_ids', [7, 8]), ('eos_token_ids', []),
])
def test_turns_must_match_the_requested_tasks_and_actual_eos(harness, field, value):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    evidence['turns'][0][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


def test_extra_duplicate_turn_does_not_pass_a_completed_campaign(harness):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    evidence['turns'].append(deepcopy(evidence['turns'][0]))
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


@pytest.mark.parametrize('field,value', [
    ('exit_code', 1), ('exit_code', None), ('violation', 'timeout'),
    ('exception', 'MemoryError: admission rejected'),
    ('cleanup_error', 'PermissionError: process cleanup failed'),
    ('sample_count', 0),
])
def test_guard_failures_cannot_be_hidden_by_completed_worker(harness, field, value):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    guard[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


@pytest.mark.parametrize('field', ['exit_code', 'violation', 'sample_count', 'limits'])
def test_missing_guard_fields_fail_closed(harness, field):
    _, campaign = harness
    evidence, guard, sources, protocol, tasks = evidence_pair()
    del guard[field]
    with pytest.raises((ValueError, RuntimeError)):
        validate(campaign, evidence, guard, sources, protocol, tasks)


def test_prepare_resolves_paths_in_the_callers_directory(tmp_path, monkeypatch, harness):
    _, campaign = harness
    caller = tmp_path / 'caller'
    caller.mkdir()
    (caller / 'model.gguf').write_bytes(b'not loaded by this CPU-only test')
    default_tasks = [task()]
    other_tasks = [task('other', 'Autre question')]
    write_json(caller / 'tasks.json', default_tasks)
    write_json(caller / 'other-tasks.json', other_tasks)
    write_json(caller / 'plan.json', [
        dict(id='default', args=[]),
        dict(id='override', tasks='other-tasks.json', args=['--bulk-metadata']),
    ])
    monkeypatch.chdir(caller)
    arguments = Namespace(model=Path('model.gguf'), tasks=Path('tasks.json'),
                          plan=Path('plan.json'), output=Path('output'))
    plan, sources, protocol, tasks_by_job = campaign.prepare_campaign(arguments)
    for field in ('model', 'tasks', 'plan', 'output'):
        assert getattr(arguments, field).is_absolute()
        assert getattr(arguments, field).parent == caller
    assert Path(plan[1]['tasks']) == caller / 'other-tasks.json'
    assert tasks_by_job == {'default': default_tasks, 'override': other_tasks}
    assert sources and protocol


def test_prepare_records_explicit_historical_engine_without_argument_override(tmp_path, monkeypatch, harness):
    _, campaign = harness
    (tmp_path / 'model.gguf').write_bytes(b'never loaded')
    write_json(tmp_path / 'tasks.json', [task()])
    historical = tmp_path / 'historical-engine'
    historical.mkdir()
    write_json(tmp_path / 'plan.json', [dict(id='historical', engine_root='historical-engine', args=[])])
    seen_roots = []

    def manifest(root):
        root = Path(root).resolve()
        seen_roots.append(root)
        return dict(git_head='historical-head' if root == historical else 'supervisor-head',
                    source_sha256={'engine.py': str(root)})

    monkeypatch.setattr(campaign, 'engine_manifest', manifest)
    monkeypatch.chdir(tmp_path)
    arguments = Namespace(model=Path('model.gguf'), tasks=Path('tasks.json'),
                          plan=Path('plan.json'), output=Path('output'))
    plan, sources, _, tasks_by_job = campaign.prepare_campaign(arguments)
    assert Path(plan[0]['engine_root']) == historical
    assert historical in seen_roots
    assert campaign.ROOT.resolve() in seen_roots
    assert sources[str(historical)]['git_head'] == 'historical-head'
    assert tasks_by_job == {'historical': [task()]}


def test_campaign_rejects_worker_bypass_before_any_process_can_start(tmp_path, monkeypatch, harness):
    _, campaign = harness
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'never loaded')
    tasks = write_json(tmp_path / 'tasks.json', [task()])
    plan = write_json(tmp_path / 'plan.json', [dict(id='unsafe', args=['--worker'])])
    monkeypatch.setattr(sys, 'argv', [
        str(ROOT / 'benchmarks/campaign_research51.py'), '--model', str(model),
        '--tasks', str(tasks), '--plan', str(plan), '--output', str(tmp_path / 'output'),
    ])
    monkeypatch.setattr(campaign, 'engine_manifest',
                        lambda root: dict(git_head='head', source_sha256={'engine.py': 'digest'}))
    monkeypatch.setattr(campaign.subprocess, 'run',
                        lambda *a, **kw: pytest.fail('Rejected plan must not start any process'))
    with pytest.raises((ValueError, RuntimeError)):
        campaign.main()


def test_final_protocol_drift_persists_stopped_status_and_fails_shell_exit(tmp_path, monkeypatch, harness):
    _, campaign = harness
    from dataclasses import asdict
    from types import SimpleNamespace
    import sys

    monkeypatch.syspath_prepend(str(ROOT / 'air_llm'))
    from airllm.nextgen.experiment_guard import ExperimentLimits

    evidence, guard, source_files, protocol, tasks = evidence_pair()
    engine = dict(git_head='frozen-head', source_sha256=source_files)
    evidence.update(git_head=engine['git_head'], git_head_after=engine['git_head'],
                    engine_head_unchanged=True)
    plan = [dict(id='one', tasks=str(tmp_path / 'tasks.json'), engine_root=str(campaign.ROOT), args=[])]
    states = {str(campaign.ROOT): engine}
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [
        'campaign_research51.py', '--model', str(tmp_path / 'model.gguf'),
        '--tasks', str(tmp_path / 'tasks.json'), '--plan', str(tmp_path / 'plan.json'),
        '--output', str(output),
    ])
    monkeypatch.setattr(campaign, 'prepare_campaign', lambda args: (plan, states, protocol, {'one': tasks}))
    monkeypatch.setattr(campaign, 'engine_manifest', lambda root: deepcopy(engine))
    checks = iter([True, True, False])
    monkeypatch.setattr(campaign, 'protocol_unchanged', lambda expected: next(checks))

    def completed_worker(command, **kwargs):
        destination = Path(command[command.index('--output') + 1])
        write_json(destination, evidence)
        worker_guard = dict(guard, command=[*command, '--worker'],
                            limits=asdict(ExperimentLimits(timeout_seconds=600.)))
        write_json(destination.with_suffix('.guard.json'), worker_guard)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, 'run', completed_worker)
    with pytest.raises(RuntimeError, match='finalization'):
        campaign.main()
    manifest = json.loads((output / 'campaign.json').read_text())
    assert manifest['status'] == 'stopped'
    assert manifest['protocol_unchanged'] is False
    assert manifest['runs'][0]['status'] == 'completed'
