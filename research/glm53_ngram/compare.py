"""Compare completed guarded A/B runs; no partial or diagnostic timings."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def read(root):
    root = Path(root)
    outer = json.loads((root/'report.json').read_text())
    native = json.loads((root/'capture/report.json').read_text())
    guard = outer.get('guard', {})
    if (outer.get('status') != 'completed' or not outer.get('native_process_launched') or
            not outer.get('weight_identities_unchanged') or not outer.get('native_artifacts_unchanged') or
            not outer.get('performance_measured') or not native.get('complete') or
            not native.get('performance_measured') or native['config']['mode'] == 'audit' or
            guard.get('exit_code') != 0 or 'violation' not in guard or guard['violation'] is not None):
        raise ValueError('comparison requires two complete, guarded, nondiagnostic timed runs')
    tokens = json.loads((root/'capture/generated.json').read_text())
    prompt = json.loads((root/'capture/prompt_tokens.json').read_text())
    if (len(tokens) != native['generated_tokens'] or len(tokens) < 2 or
            any(type(t) is not int or t < 0 for t in tokens) or
            type(native.get('eog')) is not bool or
            (len(tokens) != native['config']['tokens'] and not native['eog'])):
        raise ValueError('timed trajectory is incomplete or has no decode interval')
    return outer, native, tokens, prompt


def compare(reference, candidate):
    a, an, at, ap = read(reference)
    b, bn, bt, bp = read(candidate)
    if an['config']['mode'] != 'reference' or bn['config']['mode'] != 'ngram':
        raise ValueError('reference/ngram arms are required')
    for key in ('artifacts',):
        if a[key] != b[key]:
            raise ValueError('native artifacts differ')
    for key in ('parameters',):
        if an[key] != bn[key]:
            raise ValueError('effective native parameters differ')
    for key in ('tokens', 'prompt', 'draft_tokens', 'ngram_size'):
        if an['config'][key] != bn['config'][key]:
            raise ValueError('experiment configuration differs beyond speculation mode')
    ai = [(f['name'], f['sha256'], f['identity']) for f in a['checkpoint']['files']]
    bi = [(f['name'], f['sha256'], f['identity']) for f in b['checkpoint']['files']]
    if ai != bi:
        raise ValueError('checkpoint content or identities differ')
    equal = ap == bp and at == bt and an['eog'] == bn['eog']
    if not equal:
        raise ValueError('target trajectory or EOS differs; no speedup is accepted')
    a_seconds = an['generation_seconds_excluding_prefill_first_token']
    b_seconds = bn['generation_seconds_excluding_prefill_first_token']
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (a_seconds, b_seconds)):
        raise ValueError('positive complete generation times are required')
    return {'tokens_equal': equal, 'decode_intervals': len(at)-1,
            'reference_generation_seconds': a_seconds, 'ngram_generation_seconds': b_seconds,
            'single_pair_speedup': a_seconds/b_seconds,
            'ngram_offers': sum(bool(row['draft']) for row in json.loads((Path(candidate)/'capture/blocks.json').read_text())),
            'proposed_tokens': bn['draft_proposals'], 'accepted_tokens': bn['accepted_draft_tokens'],
            'replay_calls': bn['replay_calls'], 'reproducible_gain_qualified': False,
            'scope': 'single guarded pair; complete decode costs included; alternating repeats and useful RONIN outputs remain required'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(compare(a.reference, a.candidate), f, indent=2)
        f.write('\n')
