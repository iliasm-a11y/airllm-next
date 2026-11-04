"""Replay historical sequential IDs; never infer speculative speed or fidelity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from research.glm53_ultra.structural import simple_draft


def replay(prompt, generated, n, maximum):
    """Non-overlapping offers, with the native experiment's exact-shape policy."""
    step = 0
    offers = []
    while step < len(generated)-1:
        draft = simple_draft(prompt+generated[:step], generated[step], size_n=n, size_m=maximum)
        if len(draft) != maximum or len(draft) >= len(generated)-1-step:
            step += 1
            continue
        matching = 0
        while matching < len(draft) and draft[matching] == generated[step+1+matching]:
            matching += 1
        offers.append({'position': step, 'proposal': draft, 'matching_prefix': matching})
        step += matching+1
    return {'ngram_size': n, 'draft_tokens': maximum, 'offers': offers,
            'offer_count': len(offers), 'proposed_tokens': sum(len(o['proposal']) for o in offers),
            'matching_tokens': sum(o['matching_prefix'] for o in offers),
            'scope': 'historical sequential-trajectory agreement, not actual acceptance, fidelity or performance',
            'throughput_measured': False, 'decode_gain': None}


def analyze(capture):
    capture = Path(capture)
    prompt = json.loads((capture/'prompt_tokens.json').read_text())
    tokens = [json.loads(line)['token'] for line in (capture/'tokens.jsonl').read_text().splitlines() if line.strip()]
    if not json.loads((capture/'complete.json').read_text()).get('completed'):
        raise ValueError('only a completed sequential trajectory is admissible')
    return {'source_capture': str(capture), 'generated_tokens': len(tokens),
            'metadata_sha256': {name: hashlib.sha256((capture/name).read_bytes()).hexdigest()
                                for name in ('prompt_tokens.json', 'tokens.jsonl', 'complete.json')},
            'rows': [replay(prompt, tokens, n, m) for n in (1, 2, 3, 4) for m in (2, 4)],
            'new_model_loaded': False, 'performance_measured': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(analyze(a.capture), f, indent=2)
        f.write('\n')
