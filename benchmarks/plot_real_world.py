#!/usr/bin/env python3
"""Prepare or render publication figures from measured real-world JSON only.

Default/dry-run uses only the standard library and writes no files. Matplotlib
is imported only with --render, after checking for unfinished known runs. No
package installation, model import or inference is performed by this script.
"""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from analyze_real_world import PRIMARY_FILES, PRIMARY_IDS, completed, load_snapshot, primary_records, protect_inputs

ROOT = Path(__file__).resolve().parents[1]
CAPS = (128, 256, 512, 1024)
PHASES = {'campaign.json': 'Avant correctif du routeur',
          'campaign-resumed.json': 'Après correctif du routeur'}
COLORS = ('#2864A4', '#C76A17', '#228168', '#9254A0')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def trailing_means(seconds, window=32):
    """Full trailing windows only, indexed by one-based generated-token index.

    Input starts with the second generated token: the first token/prefill has
    already been removed. Thus the first full 32-step window ends at token 33.
    """
    if type(window) is not int or window < 1:
        raise ValueError('window must be a positive integer')
    recent, rolling, result = deque(), 0., []
    for index, value in enumerate(seconds, start=2):
        if not finite(value) or value < 0:
            raise ValueError('Token step durations must be finite and nonnegative')
        recent.append(float(value))
        rolling += value
        if len(recent) > window:
            rolling -= recent.popleft()
        if len(recent) == window:
            result.append((index, rolling / window))
    return result


def collect(evidence, window=32):
    filenames = (*PRIMARY_FILES, *(f'long-{cap}.json' for cap in CAPS))
    snapshots = {name: load_snapshot(evidence / name) for name in filenames}
    records, duplicates = primary_records(snapshots)
    points, curves, omitted = [], [], []
    for name, turn in records:
        full, appended = turn.get('full_input_ids'), turn.get('appended_input_ids')
        ttft = turn.get('ttft_seconds')
        if not isinstance(full, list) or not isinstance(appended, list) or not finite(ttft) or ttft < 0:
            omitted.append({'file': name, 'id': turn.get('id'), 'reason': 'Missing measured prompt IDs or valid TTFT'})
            continue
        points.append({'file': name, 'phase': PHASES[name], 'id': turn['id'],
                       'full_prompt_tokens': len(full), 'appended_prompt_tokens': len(appended),
                       'reused_prefix_tokens': turn.get('reused_prefix_tokens'),
                       'ttft_seconds': ttft})
    for cap in CAPS:
        name = f'long-{cap}.json'
        for turn in completed(snapshots[name]):
            times = turn.get('token_seconds')
            generated = turn.get('generated_ids')
            if not isinstance(times, list) or not isinstance(generated, list) or len(times) != len(generated):
                omitted.append({'file': name, 'id': turn.get('id'), 'reason': 'Missing complete per-token measured durations'})
                continue
            try:
                smoothed = trailing_means(times[1:], window)
            except ValueError as error:
                omitted.append({'file': name, 'id': turn.get('id'), 'reason': str(error)})
                continue
            if not smoothed:
                omitted.append({'file': name, 'id': turn.get('id'),
                                'reason': f'Fewer than {window} measured decode steps after removing prefill'})
                continue
            stats = turn.get('metrics', {}).get('weights', {})
            curves.append({'file': name, 'id': turn.get('id'), 'cap': cap,
                           'output_tokens': len(generated), 'stop_reason': turn.get('stop_reason'),
                           'window': window, 'samples': smoothed,
                           'router_fallback_calls_cumulative': stats.get('qwen_router_tie_fallback_calls')})
    active = [name for name, snapshot in snapshots.items()
              if isinstance(snapshot.get('data'), dict) and snapshot['data'].get('status') == 'started']
    return {
        'kind': 'measured_native_real_world_figure_inputs',
        'inputs': {name: {'status': snapshot['status'], 'sha256': snapshot['sha256']}
                   for name, snapshot in snapshots.items()},
        'primary_completed_ids': [turn['id'] for _, turn in records],
        'primary_expected_count': len(PRIMARY_IDS), 'primary_duplicate_observations_not_counted': duplicates,
        'points': points, 'curves': curves, 'omitted_observations': omitted,
        'unfinished_runs': active,
        'missing_runs': [name for name, snapshot in snapshots.items() if snapshot['status'] == 'absent'],
        'measurement_scope': {
            'ttft': 'Native inference TTFT; rendering/tokenization, model startup and human delays excluded.',
            'prompt': 'Both full template token count and actually appended token count are retained; cache reuse is not hidden.',
            'decode': 'Measured token_seconds after excluding first token/prefill; each point averages one full trailing window. Callback persistence is excluded from token_seconds.',
            'fallback': 'Cumulative process counters only; no exact token position is inferred for router fallback events.',
        },
    }


def plot_primary(plt, manifest, token_scope):
    points = manifest['points']
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    key = 'appended_prompt_tokens' if token_scope == 'appended' else 'full_prompt_tokens'
    for index, (name, phase) in enumerate(PHASES.items()):
        group = [point for point in points if point['file'] == name]
        if not group:
            continue
        ax.scatter([point[key] for point in group], [point['ttft_seconds'] for point in group],
                   label=phase, color=COLORS[index], marker=('o', '^')[index],
                   s=58, edgecolors='white', linewidths=.75, zorder=3)
    key_ids = {'A1', 'A2', 'D2', 'E1', 'E2', 'E5'}
    key_ids.add(max(points, key=lambda point: point['ttft_seconds'])['id'])
    for index, point in enumerate(points):
        if point['id'] in key_ids:
            ax.annotate(point['id'], (point[key], point['ttft_seconds']),
                        xytext=(5, 6 if index % 2 else -12), textcoords='offset points',
                        fontsize=8.5, color='#30363B')
    ax.set_xlabel('Tokens réellement ajoutés au cache' if token_scope == 'appended' else 'Tokens du prompt complet, historique compris')
    ax.set_ylabel('Délai du premier token du moteur (secondes)')
    ax.set_title('Prompt et délai de préremplissage — tâches principales', loc='left', pad=16, weight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.grid(axis='both', color='#DFE4E8', linewidth=.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    caption = ('TTFT moteur : tokenisation et lancement exclus. Observations uniques des phases initiale et reprise.\n'
               'La réutilisation KV en conversation modifie le nombre de tokens effectivement traités ; aucune causalité isolée n’est estimée.')
    fig.text(.11, .035, caption, fontsize=8.4, color='#505A63', ha='left', va='bottom')
    fig.subplots_adjust(left=.11, right=.97, top=.88, bottom=.20)
    return fig


def plot_decode(plt, manifest):
    curves = manifest['curves']
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    styles = ('-', '--', '-.', ':')
    for index, curve in enumerate(curves):
        fallback = curve['router_fallback_calls_cumulative']
        label = f"Plafond {curve['cap']} · {curve['output_tokens']} tokens"
        if fallback is not None:
            label += f' · replis cumulés {fallback}'
        ax.plot([sample[0] for sample in curve['samples']],
                [sample[1] for sample in curve['samples']], color=COLORS[index % len(COLORS)],
                linestyle=styles[index % len(styles)], linewidth=1.9, label=label)
    window = curves[0]['window']
    ax.set_xlabel('Indice du token généré (premier token exclu des calculs)')
    ax.set_ylabel(f'Secondes par token — moyenne mobile de {window} étapes')
    ax.set_title('Coût du décodage natif au fil de la génération', loc='left', pad=16, weight='bold')
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.grid(axis='both', color='#DFE4E8', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='best', frameon=False, fontsize=8.5)
    caption = (f'Fenêtres complètes de {window} étapes ; préremplissage du premier token et callback de sauvegarde exclus.\n'
               'Les compteurs de repli du routeur sont cumulatifs ; leur position exacte parmi les tokens n’est pas disponible.')
    fig.text(.10, .035, caption, fontsize=8.3, color='#505A63', ha='left', va='bottom')
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.20)
    return fig


def render(manifest, output, evidence, token_scope):
    if not manifest['points'] and not manifest['curves']:
        raise ValueError('No measured observations available; refusing empty figures')
    # This check does not import pyplot or NumPy in preparation/dry-run mode.
    if importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Matplotlib indisponible dans ce runtime ; aucune figure créée. Aucune installation automatique.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10.5, 'axes.titlesize': 13,
                         'svg.fonttype': 'none', 'savefig.dpi': 220})
    outputs = []
    available = []
    for stem, figure in [('prompt_vs_ttft', plot_primary(plt, manifest, token_scope)),
                         ('decode_seconds_per_token', plot_decode(plt, manifest))]:
        if figure is not None:
            available.append((stem, figure))
            outputs.extend(output / (stem + suffix) for suffix in ('.svg', '.png'))
    manifest_path = output / 'plot_manifest.json'
    inputs = [path for path in evidence.rglob('*') if path.is_file()] if evidence.is_dir() else []
    protect_inputs(inputs, [*outputs, manifest_path])
    output.mkdir(parents=True, exist_ok=True)
    for stem, figure in available:
        for extension in ('.svg', '.png'):
            figure.savefig(output / (stem + extension), bbox_inches='tight', facecolor='white')
        plt.close(figure)
    manifest['render'] = {'matplotlib': matplotlib.__version__, 'prompt_token_scope': token_scope,
                          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'outputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return [*outputs, manifest_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'evidence-real-world')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/real_world/figures')
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--prompt-token-scope', choices=('appended', 'full'), default='appended')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--render', action='store_true', help='Import Matplotlib and export measured SVG/PNG after inference')
    mode.add_argument('--dry-run', action='store_true', help='Validate measured inputs with stdlib only; write nothing (default)')
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly allow rendering with known unfinished reports')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    manifest = collect(args.evidence_dir.resolve(), args.window)
    summary = {'mode': 'render' if args.render else 'dry-run',
               'primary_measured_points': len(manifest['points']),
               'long_measured_curves': len(manifest['curves']),
               'unfinished_runs': manifest['unfinished_runs'], 'missing_runs': manifest['missing_runs'],
               'omitted_observations': manifest['omitted_observations']}
    if args.render:
        if manifest['unfinished_runs'] and not args.allow_incomplete:
            parser.error('Known runs are unfinished; wait for inference or explicitly use --allow-incomplete')
        try:
            summary['outputs'] = list(map(str, render(manifest, args.output_dir, args.evidence_dir, args.prompt_token_scope)))
        except (RuntimeError, ValueError) as error:
            parser.exit(2, str(error) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
