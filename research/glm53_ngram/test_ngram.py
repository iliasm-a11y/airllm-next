"""Native proposer fixtures and fail-closed qualification; no model is loaded."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from research.glm53_ngram.run import qualification
from research.glm53_ultra.structural import simple_draft


@pytest.fixture
def probe():
    value = os.environ.get('GLM53_NGRAM_PROBE')
    if not value:
        pytest.skip('set GLM53_NGRAM_PROBE for the compiled, model-free native fixtures')
    return Path(value)


def native_draft(probe, tmp_path, history, sampled, n, m):
    path = tmp_path/'fixture.json'
    path.write_text(json.dumps({'history': history, 'sampled': sampled, 'n': n, 'm': m}))
    result = subprocess.run([str(probe), '--draft-fixture', str(path)], capture_output=True,
                            text=True, check=True, timeout=5)
    return json.loads(result.stdout)


def test_native_draft_matches_pinned_replay_across_search_boundaries(probe, tmp_path):
    rng = random.Random(53041104)
    histories = [[], [1, 2, 3], [9, 1, 2, 3, 4, 1, 2, 7],
                 [9, 1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 7, 8]]
    histories += [[rng.randrange(5) for _ in range(length)] for length in (5, 8, 16, 32, 64)]
    for history in histories:
        for n, m in ((1, 1), (1, 2), (2, 2), (2, 4), (3, 4), (4, 4)):
            for sampled in (0, 2):
                assert native_draft(probe, tmp_path, history, sampled, n, m) == simple_draft(
                    history, sampled, size_n=n, size_m=m)


def test_native_parameters_do_not_load_a_model_or_create_extra_slots(probe):
    r = subprocess.run([str(probe), '--parameters'], capture_output=True, text=True, check=True, timeout=5)
    assert json.loads(r.stdout) == {'n_ctx': 512, 'n_batch': 8, 'n_ubatch': 8,
                                  'physical_slots': 1, 'threads': 10, 'gpu_layers': 0,
                                  'n_rs_seq': 0, 'second_model': False}


@pytest.fixture
def receipt(tmp_path):
    path = tmp_path/'report.json'
    (tmp_path/'capture').mkdir()
    identity = {'probe': 'a'*64, 'libraries': {'libllama.dylib': 'b'*64}}
    config = {'mode': 'ngram', 'draft_tokens': 2, 'prompt': 'public fixture'}
    outer = {'status': 'completed', 'artifacts': identity,
             'weight_identities_unchanged': True, 'native_artifacts_unchanged': True}
    cases = [{'draft_tokens': d, 'label': f'draft{d}/{label}', 'passed': True}
             for d in (1, 2) for label in ('all-accepted', 'reject-first', 'reject-middle')
             if d > 1 or label != 'reject-middle']
    native = {'complete': True, 'qualification_passed': True,
              'config': {**config, 'mode': 'audit'}, 'cases': cases}
    path.write_text(json.dumps(outer))
    (tmp_path/'capture/report.json').write_text(json.dumps(native))
    return path, config, identity, outer, native


def test_qualified_artifacts_admit_timing(receipt):
    path, config, identity, _, _ = receipt
    assert len(qualification(path, config, identity)['sha256']) == 64


@pytest.mark.parametrize('failure', ['missing', 'failed', 'guard_abort', 'other_binary', 'too_large',
                                     'missing_rejection', 'other_prompt', 'state_failure'])
def test_timing_refuses_incomplete_or_incompatible_qualification(receipt, failure):
    path, config, identity, outer, native = receipt
    if failure == 'missing':
        path = None
    elif failure == 'failed':
        native['qualification_passed'] = False
    elif failure == 'guard_abort':
        outer['status'] = 'failed'
    elif failure == 'other_binary':
        identity = {**identity, 'probe': 'c'*64}
    elif failure == 'too_large':
        config['draft_tokens'] = 4
    elif failure == 'missing_rejection':
        native['cases'] = [c for c in native['cases'] if not c['label'].endswith('reject-first')]
    elif failure == 'other_prompt':
        config['prompt'] = 'a different prompt'
    elif failure == 'state_failure':
        native['cases'][0]['passed'] = False
    if path:
        path.write_text(json.dumps(outer))
        (path.parent/'capture/report.json').write_text(json.dumps(native))
    with pytest.raises(ValueError):
        qualification(path, config, identity)
