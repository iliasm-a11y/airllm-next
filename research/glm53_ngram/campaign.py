"""One bounded A/B/B/A campaign; no automatic retries or threshold changes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from airllm.nextgen.experiment_guard import system_snapshot
from research.glm53_ngram.compare import compare
from research.glm53_ngram.prepare import prepare
from research.glm53_ngram.run import fingerprints, qualification
from research.glm53_ultra_decode.build import sha


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def campaign(args):
    root = args.output.resolve()
    if root.exists() or root.is_relative_to(args.model.resolve()) or root.is_relative_to(args.runtime.resolve()):
        raise ValueError('a fresh campaign directory outside model/runtime is required')
    identity = fingerprints(args.probe, args.runtime)
    config = {'mode': 'ngram', 'tokens': args.tokens, 'ngram_size': args.ngram_size,
              'draft_tokens': args.draft_tokens, 'prompt': args.prompt}
    qualification(args.qualification, config, identity)
    sources = {str(p): sha(p) for p in Path(__file__).resolve().parent.glob('*')
               if p.is_file() and p.suffix in ('.py', '.cpp')}
    root.mkdir(parents=True)
    baseline = system_snapshot()['swap_used_bytes']
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'config': config,
              'order': ['A1', 'B1', 'B2', 'A2'], 'arms': [], 'artifacts': identity,
              'sources_sha256': sources, 'campaign_swap_used_bytes': baseline,
              'qualification': str(args.qualification.resolve()), 'production_changed': False,
              'reproducible_gain_qualified': False}
    write(root/'campaign.json', report)
    try:
        for label, mode in (('A1', 'reference'), ('B1', 'ngram'), ('B2', 'ngram'), ('A2', 'reference')):
            if any(sha(p) != digest for p, digest in sources.items()) or identity != fingerprints(args.probe, args.runtime):
                raise RuntimeError('campaign source or binary pin changed; no next model launched')
            row = {'label': label, 'mode': mode, 'status': 'preparing'}
            report['arms'].append(row)
            write(root/'campaign.json', report)
            preparation = prepare(args.model.resolve(), args.receipt.resolve())
            write(root/(label+'-preparation.json'), preparation)
            arm = root/label
            command = [sys.executable, '-m', 'research.glm53_ngram.run']
            for name in ('model', 'receipt', 'runtime', 'probe'):
                command.extend(('--'+name, str(getattr(args, name).resolve())))
            command.extend(('--output', str(arm), '--mode', mode, '--tokens', str(args.tokens),
                            '--draft-tokens', str(args.draft_tokens), '--ngram-size', str(args.ngram_size),
                            '--prompt', args.prompt, '--campaign-swap-used-bytes', str(baseline)))
            if mode == 'ngram':
                command.extend(('--qualification', str(args.qualification.resolve())))
            row['status'] = 'running'
            write(root/'campaign.json', report)
            started = time.monotonic()
            with (root/(label+'-runner.log')).open('wb') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=result.returncode, total_process_seconds=time.monotonic()-started)
            outer = json.loads((arm/'report.json').read_text())
            row['status'] = outer['status']
            if result.returncode or outer['status'] != 'completed':
                write(root/'campaign.json', report)
                raise RuntimeError('campaign stopped at '+label+'; no retry or following arm')
            native = json.loads((arm/'capture/report.json').read_text())
            for name in ('generated_tokens', 'decode_intervals', 'generation_seconds_excluding_prefill_first_token',
                         'decode_intervals_per_second', 'draft_proposals', 'accepted_draft_tokens',
                         'replay_calls', 'actual_target_calls', 'actual_target_positions', 'eog'):
                row[name] = native[name]
            write(root/'campaign.json', report)
        report['pairs'] = [compare(root/'A1', root/'B1'), compare(root/'A2', root/'B2')]
        report['status'] = 'completed'
        report['both_pairs_faster'] = all(p['single_pair_speedup'] > 1 for p in report['pairs'])
        report['scope'] = '32/64/128 raw greedy output cap, same native executable/runtime/checkpoint; no useful RONIN or EOS qualification'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(root/'campaign.json', report)
    print(json.dumps({'status': report['status'], 'arms': [r['label'] for r in report['arms']]}))
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'receipt', 'runtime', 'probe', 'qualification', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--tokens', type=int, choices=(32, 64, 128), default=32)
    p.add_argument('--draft-tokens', type=int, choices=(2, 4), default=2)
    p.add_argument('--ngram-size', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--prompt', default='La capitale de la France est')
    raise SystemExit(campaign(p.parse_args()))
