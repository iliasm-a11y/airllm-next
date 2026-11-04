import json
from pathlib import Path

import pytest

from research.glm53_ngram.compare import compare


@pytest.fixture
def arms(tmp_path):
    roots = [tmp_path/'a', tmp_path/'b']
    for root, mode, seconds in zip(roots, ('reference', 'ngram'), (100., 80.)):
        (root/'capture').mkdir(parents=True)
        outer = {'status': 'completed', 'native_process_launched': True, 'performance_measured': True,
                 'weight_identities_unchanged': True, 'native_artifacts_unchanged': True,
                 'artifacts': {'probe': 'a'*64}, 'checkpoint': {'files': [
                     {'name': 'fixture.gguf', 'sha256': 'b'*64, 'identity': [1, 2, 3]}]},
                 'guard': {'exit_code': 0, 'violation': None}}
        native = {'complete': True, 'performance_measured': True,
                  'config': {'mode': mode, 'tokens': 32, 'prompt': 'public fixture', 'draft_tokens': 2, 'ngram_size': 2},
                  'parameters': {'physical_slots': 1}, 'generated_tokens': 32, 'eog': False,
                  'generation_seconds_excluding_prefill_first_token': seconds,
                  'draft_proposals': 2, 'accepted_draft_tokens': 2, 'replay_calls': 0}
        for path, value in ((root/'report.json', outer), (root/'capture/report.json', native),
                            (root/'capture/generated.json', list(range(32))),
                            (root/'capture/prompt_tokens.json', [100, 200]),
                            (root/'capture/blocks.json', [{'draft': [1, 2]}])):
            path.write_text(json.dumps(value))
    return roots


def test_one_faithful_pair_is_not_a_reproducibility_claim(arms):
    result = compare(*arms)
    assert result['single_pair_speedup'] == 1.25
    assert result['reproducible_gain_qualified'] is False
    assert result['decode_intervals'] == 31


@pytest.mark.parametrize('failure', ['partial', 'guard', 'tokens', 'eos', 'nan', 'other_artifact', 'checkpoint'])
def test_invalid_or_unfaithful_pair_cannot_claim_speedup(arms, failure):
    root = arms[1]
    if failure in ('guard', 'other_artifact', 'checkpoint'):
        path = root/'report.json'
        d = json.loads(path.read_text())
        if failure == 'guard':
            d['guard']['violation'] = 'memory_pressure'
        elif failure == 'other_artifact':
            d['artifacts']['probe'] = 'c'*64
        else:
            d['checkpoint']['files'][0]['identity'] = [1, 2, 9]
    elif failure == 'tokens':
        path = root/'capture/generated.json'
        d = list(range(32)); d[-1] = 99
    else:
        path = root/'capture/report.json'
        d = json.loads(path.read_text())
        if failure == 'partial':
            d['config']['tokens'] = 64
        elif failure == 'eos':
            d['eog'] = True
        else:
            d['generation_seconds_excluding_prefill_first_token'] = float('nan')
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        compare(*arms)
