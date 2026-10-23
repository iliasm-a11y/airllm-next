#!/usr/bin/env python3
"""Build French reports from existing real-world evidence; never run inference.

Only the two generated Markdown reports are written. Raw JSON evidence, source
snapshots, prompts and model files are never altered. Missing or unfinished
runs stay visible and are not promoted to completed observations.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ('campaign.json', 'campaign-resumed.json', 'hard-reasoning.json',
             'concision-control.json', 'f2-concision-control.json', 'd2-clarified.json', 'pilot-short-caps.json',
             'long-128.json', 'long-256.json', 'long-512.json', 'long-1024.json',
             'conversation-rebuild.json')
PRIMARY_FILES = ('campaign.json', 'campaign-resumed.json')
PRIMARY_IDS = ('A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2',
               'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'G1', 'G2', 'G3', 'G4', 'G5')
AXES = ('exactitude_factuelle', 'respect_instructions', 'coherence',
        'completude', 'utilite_reelle')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, decimals=2):
    if not number(value):
        return '—'
    return f'{value:,.{decimals}f}'.replace(',', ' ').replace('.', ',')


def cell(value):
    if value is None:
        return '—'
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def yes(value):
    return 'oui' if value is True else 'non' if value is False else '—'


def table(headers, rows):
    result = ['| ' + ' | '.join(map(cell, headers)) + ' |',
              '| ' + ' | '.join('---' for _ in headers) + ' |']
    result.extend('| ' + ' | '.join(map(cell, row)) + ' |' for row in rows)
    return '\n'.join(result)


def fence(text):
    # Generated code may contain backtick fences. Never modify its contents.
    runs = re.findall(r'`+', text)
    marker = '`' * max(3, 1 + max(map(len, runs), default=0))
    return marker + 'text\n' + text + '\n' + marker


def load_snapshot(path):
    if not path.exists():
        return {'path': path, 'status': 'absent', 'data': None, 'sha256': None}
    for attempt in range(3):
        raw = path.read_bytes()
        try:
            return {'path': path, 'status': 'lu', 'data': json.loads(raw), 'sha256': sha(raw)}
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if attempt < 2:
                time.sleep(.02)  # The worker may be replacing its report now.
    return {'path': path, 'status': 'illisible', 'data': None,
            'sha256': sha(raw), 'error': f'{type(error).__name__}: {error}'}


def completed(snapshot):
    data = snapshot.get('data')
    return [turn for turn in data.get('turns', []) if turn.get('status') == 'completed'] if isinstance(data, dict) else []


def effective_status(snapshot, turn=None):
    """Display a final guard stop without mutating or completing raw evidence."""
    data = snapshot.get('data') or {}
    raw = (turn or data).get('status', snapshot.get('status', 'inconnu'))
    if turn is not None and raw == 'completed':
        return raw  # Earlier completed turns remain observations in a stopped run.
    guard = snapshot.get('guard', {}).get('data') or {}
    if type(guard.get('exit_code')) is int:
        if guard.get('violation') is not None:
            return 'interrompu_garde'
        if raw == 'failed':
            return raw
        if guard['exit_code'] != 0 or guard.get('exception'):
            return 'interrompu_processus'
        if raw in ('started', 'in_progress'):
            return 'termine_incomplet'
    return raw


def guard_description(snapshot):
    guard = snapshot.get('guard', {}).get('data') or {}
    if type(guard.get('exit_code')) is not int:
        return []
    return [
        f"Garde final : code de sortie {guard['exit_code']} ; violation `{guard.get('violation') or 'aucune'}` ; "
        f"pression mémoire maximale observée {guard.get('peak_pressure_level', 'indisponible')}. "
        'Ces mesures concernent toute l’exécution supervisée et son arbre de processus.', '',
        table(['Mesure du garde', 'Octets', 'Valeur convertie'], [
            ['Croissance maximale de swap système', guard.get('maximum_swap_growth_bytes'),
             fmt(guard['maximum_swap_growth_bytes'] / 2**20) + ' MiB' if number(guard.get('maximum_swap_growth_bytes')) else '—'],
            ['Limite de croissance de swap du garde', guard.get('limits', {}).get('swap_growth_bytes'),
             fmt(guard['limits']['swap_growth_bytes'] / 2**20) + ' MiB'
             if number(guard.get('limits', {}).get('swap_growth_bytes')) else '—'],
            ['Pic RSS échantillonné', guard.get('sampled_peak_rss_bytes'),
             fmt(guard['sampled_peak_rss_bytes'] / 2**20) + ' MiB' if number(guard.get('sampled_peak_rss_bytes')) else '—'],
            ['Disponible système minimal', guard.get('minimum_available_bytes'),
             fmt(guard['minimum_available_bytes'] / 2**30) + ' GiB' if number(guard.get('minimum_available_bytes')) else '—'],
        ]), '',
    ]


def primary_records(runs):
    """One completed observation per primary ID; raw phases stay separate."""
    chosen, duplicates = {}, []
    for name in PRIMARY_FILES:
        for turn in completed(runs.get(name, {})):
            identifier = turn.get('id')
            if identifier not in PRIMARY_IDS:
                continue
            if identifier in chosen:
                previous_name, previous = chosen[identifier]
                duplicates.append({'id': identifier, 'kept_source': previous_name,
                                   'other_source': name,
                                   'same_ids': previous.get('generated_ids') == turn.get('generated_ids')})
            else:
                chosen[identifier] = (name, turn)
    return [chosen[identifier] for identifier in PRIMARY_IDS if identifier in chosen], duplicates


def display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def protect_inputs(inputs, outputs):
    for index, output in enumerate(outputs):
        for other in [*inputs, *outputs[:index]]:
            if output.resolve() == other.resolve() or (output.exists() and other.exists() and output.samefile(other)):
                raise ValueError(f'Refusing to overwrite an input or alias another output: {output}')


def total(turn):
    return turn.get('response_wall_seconds', turn.get('total_seconds'))


def perceived_ttft_bounds(turn):
    lower = turn.get('perceived_ttft_seconds')
    wall, engine = total(turn), turn.get('total_seconds')
    upper = lower + max(0., wall - engine) if all(number(value) for value in (lower, wall, engine)) else None
    return lower, upper


def aggregate(turns):
    n = sum(len(turn.get('generated_ids', [])) for turn in turns)
    walls = [total(turn) for turn in turns]
    decode = [(max(0, len(turn.get('generated_ids', [])) - 1),
               turn['total_seconds'] - turn['ttft_seconds'])
              for turn in turns if number(turn.get('total_seconds')) and number(turn.get('ttft_seconds'))]
    wall_sum = sum(walls) if all(number(value) for value in walls) else None
    decode_n = sum(pair[0] for pair in decode)
    decode_s = sum(pair[1] for pair in decode)
    logical = [turn.get('logical_read_bytes') for turn in turns]
    storage = [turn.get('process_storage_read_bytes') for turn in turns]
    rss = [turn.get('memory', {}).get('sampled_peak_rss_bytes') for turn in turns]
    swap = [turn.get('memory', {}).get('maximum_swap_growth_bytes') for turn in turns]
    return {
        'turns': len(turns), 'tokens': n, 'wall_seconds': wall_sum,
        'tokens_per_second': n / wall_sum if wall_sum else None,
        'decode_tokens_per_second': decode_n / decode_s if decode_s > 0 and len(decode) == len(turns) else None,
        'logical_read_bytes': sum(logical) if all(number(value) for value in logical) else None,
        'storage_read_bytes': sum(storage) if all(number(value) for value in storage) else None,
        'storage_coverage': sum(number(value) for value in storage),
        'peak_rss_bytes': max((value for value in rss if number(value)), default=None),
        'max_swap_growth_bytes': max((value for value in swap if number(value)), default=None),
    }


def repetition(turn, size):
    stop = set(turn.get('eos_token_ids', []))
    tokens = [token for token in turn.get('generated_ids', []) if token not in stop]
    windows = max(0, len(tokens) - size + 1)
    counts = Counter(tuple(tokens[start:start + size]) for start in range(windows))
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return f'{repeated}/{windows} ({fmt(100 * repeated / windows if windows else 0, 1)} %)'


def router_counter_rows(runs):
    rows = []
    for name, snapshot in runs.items():
        previous = {}
        for turn in completed(snapshot):
            stats = turn.get('metrics', {}).get('weights', {})
            if not any(key.startswith('qwen_router_tie_') for key in stats):
                continue
            current = stats.get('qwen_router_tie_fallback_calls', 0)
            elapsed = stats.get('qwen_router_tie_fallback_seconds', 0.)
            prior = previous.get('calls', 0)
            prior_elapsed = previous.get('seconds', 0.)
            delta = current - prior if number(current) and number(prior) and current >= prior else None
            delta_elapsed = elapsed - prior_elapsed if number(elapsed) and number(prior_elapsed) and elapsed >= prior_elapsed else None
            rows.append([name, turn.get('id'), delta, current,
                         stats.get('qwen_router_tie_fallback_rows', 0), fmt(delta_elapsed, 4),
                         fmt(stats.get('qwen_router_tie_reference_import_seconds', 0.), 4),
                         stats.get('qwen_router_tie_reference_version'),
                         stats.get('qwen_router_tie_reference_device')])
            previous = {'calls': current, 'seconds': elapsed}
    return rows


def d2_sections(runs):
    lines = ['## D2 — arrêt initial, reprise et sémantique de la date', '']
    initial_data = runs['campaign.json'].get('data') or {}
    failed = next((turn for turn in initial_data.get('turns', [])
                   if turn.get('id') == 'D2' and turn.get('status') == 'failed'), None)
    replayed = next((turn for turn in completed(runs['campaign-resumed.json']) if turn.get('id') == 'D2'), None)
    if failed and replayed:
        old = failed.get('partial_generated_ids', [])
        new = replayed.get('generated_ids', [])
        equal = len(new) >= len(old) and new[:len(old)] == old
        lines.extend([table(['Mesure', 'Observation'], [
            ['Tokens avant arrêt initial', len(old)], ['Tokens de la reprise achevée', len(new)],
            ['Préfixe initial exactement conservé dans la reprise', yes(equal)],
            ['Prompt utilisateur identique', yes(failed.get('prompt') == replayed.get('prompt'))],
            ['Arrêt initial', failed.get('error')], ['Arrêt de la reprise', replayed.get('stop_reason')],
        ]), ''])
        old_line = failed.get('partial_text', '').splitlines()[0] if failed.get('partial_text') else ''
        new_line = replayed.get('text', '').splitlines()[0] if replayed.get('text') else ''
        lines.extend(['Première ligne du fragment initial, reproduite exactement :', '', fence(old_line), '',
                      'Première ligne de la reprise achevée :', '', fence(new_line), '',
                      'La date retenue pour la formation est le **22 mai** ; le **8 avril** est la date de version du dossier. '
                      'La résolution de l’arrêt du routeur et l’interprétation de cette date sont deux résultats distincts. '
                      'La consigne initiale « date actuelle » peut prêter à une lecture de la date de version ; '
                      'la référence visait la date de formation. Ces extraits et la revue indépendante permettent de discuter cette ambiguïté '
                      'sans attribuer automatiquement une erreur de compréhension au moteur. '
                      'La formulation ambiguë ne permet pas de conclure à une erreur certaine du modèle sur cette seule réponse.'])
        if old_line == new_line and old_line:
            lines.extend(['', 'La première ligne est inchangée après reprise ; le correctif de routage n’a pas réécrit ce passage antérieur à l’arrêt. Cela ne tranche pas l’ambiguïté de la consigne.'])
    else:
        lines.append('Comparaison en attente : il faut le fragment D2 initial explicitement failed et le tour D2 achevé dans campaign-resumed.json. Aucun préfixe manquant n’est reconstruit.')
    clarified = next(iter(completed(runs['d2-clarified.json'])), None)
    if clarified:
        first = clarified.get('text', '').splitlines()[0] if clarified.get('text') else ''
        lines.extend(['', 'Contrôle lexical séparé `D2_clarified`, première ligne exacte :', '', fence(first), '',
                      'Ce contrôle explicite la date de formation dans la consigne. Il modifie la formulation et ne remplace pas D2 dans les 21 tâches principales.'])
    return lines


def concision_sections(runs, records):
    lines = ['## Contrôle de concision — séparé de A2', '',
             'Le contrôle A2_concis reprend la même question arithmétique avec une consigne système supplémentaire : '
             'deux lignes, 45 mots au maximum, sans préambule ni format Markdown/LaTeX. Il ne remplace pas la sortie A2 principale.', '']
    original = next((turn for _, turn in records if turn.get('id') == 'A2'), None)
    concise = next((turn for turn in completed(runs['concision-control.json']) if turn.get('id') == 'A2_concis'), None)
    rows = []
    for label, turn in [('A2 principal', original), ('A2_concis diagnostic', concise)]:
        if turn:
            rows.append([label, len(turn.get('generated_ids', [])), len(turn.get('text', '').split()),
                         sum(bool(line.strip()) for line in turn.get('text', '').splitlines()),
                         fmt(total(turn)), turn.get('stop_reason')])
    lines.append(table(['Observation', 'Tokens', 'Mots séparés par espaces', 'Lignes non vides', 'Total s', 'Arrêt'], rows)
                 if rows else 'Aucune réponse achevée disponible pour ce contrôle.')
    if original and concise:
        lines.extend(['', f"Question utilisateur exacte identique : {yes(original.get('prompt') == concise.get('prompt'))}. "
                      'Les comptes de mots et de lignes sont descriptifs ; la présence des résultats corrects et le respect global des consignes restent examinés par la revue.'])
    return lines


def source_checks(snapshot, evidence):
    data = snapshot.get('data')
    if not isinstance(data, dict):
        return []
    result = []
    for relative, expected in data.get('identity', {}).get('sources', {}).items():
        path = (ROOT / relative).resolve()
        current = sha(path.read_bytes()) if path.is_relative_to(ROOT) and path.is_file() else None
        archives = []
        if current != expected:
            for candidate in evidence.rglob(Path(relative).name):
                if candidate.is_file() and sha(candidate.read_bytes()) == expected:
                    archives.append(display_path(candidate))
        result.append({'path': relative, 'expected': expected, 'current': current,
                       'matches': current == expected, 'archives': archives})
    return result


def turn_lookup(runs, source, turn_id):
    filename = Path(source or 'campaign.json').name
    snapshot = runs.get(filename)
    if snapshot is None:
        return None
    return next((turn for turn in completed(snapshot) if turn.get('id') == turn_id), None)


def review_sections(runs, checks, review):
    lines = ['## Contrôles objectifs du code', '']
    if not isinstance(checks.get('data'), dict):
        lines.append('Fichier `code-checks.json` absent ou illisible : aucune exécution de code n’est déclarée ici.')
    else:
        data = checks['data']
        source = data.get('responses_file', 'campaign.json')
        rows = []
        verified = 0
        for item in data.get('results', []):
            turn = turn_lookup(runs, source, item.get('id'))
            matches = (sha(turn['text'].encode()) == item.get('response_sha256')) if turn and isinstance(turn.get('text'), str) else None
            verified += matches is True
            identities = [case['same_object_as_first_argument'] for case in item.get('test_results', {}).get('cases', [])
                          if type(case.get('same_object_as_first_argument')) is bool]
            rows.append([item.get('id'), yes(matches), yes(item.get('executed')),
                         yes(item.get('passed')), yes(item.get('extraction', {}).get('format_code_only')),
                         yes(any(identities)) if identities else '—',
                         item.get('error', '')])
        lines.extend(['Le succès des cas exécutés et le respect du format « code seul » sont séparés. '
                      'Une réponse absente ou non exécutée n’est pas une preuve d’échec du modèle.', '',
                      f"Résumé enregistré : {json.dumps(data.get('summary', {}), ensure_ascii=False)}. "
                      f'Empreinte du texte vérifiée pour {verified}/{len(rows)} résultat(s).', ''])
        if rows:
            lines.append(table(['Tour', 'SHA réponse concorde', 'Exécuté', 'Cas réussis', 'Code seul', 'Retourne le même objet', 'Erreur'], rows))
    lines.extend(['', '## Évaluation indépendante', ''])
    if not isinstance(review.get('data'), dict):
        lines.append('Fichier `independent_review.json` absent ou illisible : aucune note qualitative n’est produite automatiquement.')
        return lines
    data = review['data']
    lines.extend([f"État enregistré : **{data.get('status', 'inconnu')}**. {data.get('method', '')}", ''])
    rows = []
    notes = []
    valid = 0
    for item in data.get('reviewed_turns', []):
        turn = turn_lookup(runs, item.get('source_file'), item.get('id'))
        key = item.get('source_response_key', 'text')
        key = key if isinstance(key, str) and key in ('text', 'raw_text') else 'text'
        response = turn.get(key) if turn else None
        matches = sha(response.encode()) == item.get('response_sha256') if isinstance(response, str) else None
        valid += matches is True
        scores = item.get('scores', {})
        values = [scores.get(axis, {}).get('score') if isinstance(scores.get(axis), dict)
                  else scores.get(axis) for axis in AXES]
        rows.append([item.get('id'), Path(item.get('source_file', 'campaign.json')).name,
                     yes(matches), *values])
        notes.extend([f"### {item.get('id')} — justification de l’évaluateur", '',
                      f"Correspondance SHA : {yes(matches)}. Fin : {item.get('termination', 'non précisée')}.", ''])
        for axis in AXES:
            value = scores.get(axis)
            if isinstance(value, dict) and value.get('rationale'):
                notes.append(f"- {axis} ({value.get('score', '—')}/5) : {value['rationale']}")
        for check in item.get('objective_checks', []):
            notes.append(f"- Contrôle : {check.get('check', '')} ; résultat : {check.get('result', '')} ; preuve : {check.get('evidence', '')}")
        for uncertainty in item.get('uncertainties', []):
            notes.append(f'- Limite relevée : {uncertainty}')
        notes.append('')
    lines.append(f'{len(rows)} tour(s) annoté(s), dont {valid} avec texte source complet et empreinte concordante. '
                 'Les notes sont celles de l’évaluateur enregistré, sur 5 ; elles ne sont pas déduites des métriques de répétition.')
    if rows:
        lines.extend(['', table(['Tour', 'Source', 'SHA concorde', 'Exactitude', 'Instructions',
                                 'Cohérence', 'Complétude', 'Utilité'], rows), '', *notes])
    return lines


def annotation_appendices(runs, evidence):
    """Keep ladder fragments and the standalone CLI control outside main scores."""
    def link(path):
        try:
            target = '../../' + path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            target = str(path.resolve())
        return f'[{path.name}]({target})'

    def scores(item):
        values = item.get('scores', {})
        return [values.get(axis, {}).get('score') if isinstance(values.get(axis), dict)
                else values.get(axis) for axis in AXES]

    lines = ['', '## Annexes d’annotation — hors des 21 tâches principales', '']
    ladder = load_snapshot(evidence / 'long_review.json')
    if isinstance(ladder.get('data'), dict):
        data = ladder['data']
        lines.extend(['### Paliers longs et fragment interrompu', '',
                      f"Annotations intégrales : {link(ladder['path'])}. SHA-256 de cette lecture : `{ladder['sha256']}`.", '',
                      'Les lignes achevées décrivent les paliers réellement obtenus. La ligne du fragment L1024 porte '
                      'des notes **conditionnelles aux seuls 244 tokens observés** ; elle ne constitue ni une réponse achevée, '
                      'ni une validation du palier 1 024, et n’entre dans aucun agrégat de réponses achevées.', ''])
        rows = []
        for collection in ('reviewed_turns', 'interrupted_turns'):
            for item in data.get(collection, []):
                source_name = Path(item.get('source_file', '')).name
                snapshot = runs.get(source_name, {})
                candidate = next((turn for turn in (snapshot.get('data') or {}).get('turns', [])
                                  if turn.get('id') == item.get('id')), None)
                key = item.get('source_response_key')
                interrupted = collection == 'interrupted_turns'
                allowed_key = key == 'partial_text' if interrupted else key in ('text', 'raw_text')
                status = effective_status(snapshot, candidate) if candidate else 'source absente'
                correct_status = status in ('interrompu_garde', 'interrompu_processus', 'termine_incomplet') if interrupted else status == 'completed'
                text = candidate.get(key) if candidate and allowed_key else None
                binding = (sha(text.encode()) == item.get('response_sha256') and correct_status
                           if isinstance(text, str) else None)
                rows.append(['fragment interrompu — conditionnel' if interrupted else 'palier achevé',
                             item.get('id'), source_name, status, yes(binding), *scores(item)])
        if rows:
            lines.append(table(['Collection', 'Tour', 'Source', 'Statut effectif', 'SHA / champ / statut concordent',
                                'Exactitude', 'Instructions', 'Cohérence', 'Complétude', 'Utilité'], rows))
    else:
        lines.append('Annotations des paliers longs absentes ou illisibles ; aucune note annexe inventée.')

    baseline = load_snapshot(evidence / 'baseline_review.json')
    if isinstance(baseline.get('data'), dict):
        source = load_snapshot(evidence / 'baseline-chat-math.json')
        guard = load_snapshot(evidence / 'baseline-chat-math.guard.json')
        lines.extend(['', '### Contrôle initial autonome 2+2 — schéma CLI distinct', '',
                      f"Revue : {link(baseline['path'])} ; preuve CLI : {link(source['path'])} ; garde : {link(guard['path'])}.", '',
                      f"SHA-256 de la revue lue : `{baseline['sha256']}` ; de la preuve CLI : `{source['sha256']}`.", '',
                      'La preuve CLI place `text` directement à la racine et ne possède pas de collection `turns`. '
                      'Ce contrôle élémentaire demeure dans cette annexe : il ne devient pas une tâche principale ou un palier long.', ''])
        rows = []
        for item in baseline['data'].get('reviewed_turns', []):
            data = source.get('data') or {}
            guard_data = guard.get('data') or {}
            source_matches = Path(item.get('source_file', '')).name == source['path'].name
            key = item.get('source_response_key')
            text = data.get(key) if key in ('text', 'raw_text') else None
            binding = source_matches and sha(text.encode()) == item.get('response_sha256') if isinstance(text, str) else None
            command = guard_data.get('command', [])
            try:
                guard_prompt = command[command.index('--prompt') + 1]
            except (ValueError, IndexError, TypeError):
                guard_prompt = None
            prompt_matches = guard_prompt == item.get('prompt') if isinstance(guard_prompt, str) else None
            rows.append([item.get('id'), yes(binding), yes(prompt_matches), guard_data.get('exit_code'),
                         guard_data.get('violation') or 'aucune déclarée', *scores(item)])
        if rows:
            lines.append(table(['Contrôle CLI', 'SHA réponse concorde', 'Prompt du garde concorde', 'Code garde',
                                'Violation garde', 'Exactitude', 'Instructions', 'Cohérence', 'Complétude', 'Utilité'], rows))
    else:
        lines.extend(['', 'Revue du contrôle initial autonome absente ou illisible ; aucune note annexe inventée.'])
    for filename, title in (
            ('diagnostic_review.json', 'Contrôles diagnostiques — sans remplacement des réponses principales'),
            ('supplementary_review.json', 'A4 — supplément prospectif, hors des 21 tâches principales')):
        extra = load_snapshot(evidence / filename)
        if isinstance(extra.get('data'), dict):
            sections = review_sections(runs, {'data': None}, extra)
            start = sections.index('## Évaluation indépendante')
            sections[start] = '### ' + title
            lines.extend(['', f"Revue séparée : {link(extra['path'])} ; SHA-256 lu : `{extra['sha256']}`.", '',
                          *sections[start:]])
    return lines


def make_analysis(runs, checks, review, evidence, timestamp, pilot_review=None):
    main_records, duplicates = primary_records(runs)
    main_turns = [turn for _, turn in main_records]
    supplemental = [turn for turn in completed(runs['hard-reasoning.json']) if turn.get('id') == 'A4']
    lines = ['# Analyse des évaluations réelles — AirLLM Next', '',
             f'Instantané généré le {timestamp}. Lecture de fichiers existants ; aucune inférence déclenchée.', '',
             '## Couverture des preuves', '']
    coverage = []
    for name, snapshot in runs.items():
        data = snapshot.get('data')
        turns = data.get('turns', []) if isinstance(data, dict) else []
        statuses = [effective_status(snapshot, turn) for turn in turns]
        terminal = ('interrompu_garde', 'interrompu_processus', 'termine_incomplet')
        coverage.append([name, effective_status(snapshot),
                         data.get('status', 'inconnu') if isinstance(data, dict) else snapshot['status'],
                         len(completed(snapshot)), sum(status == 'failed' for status in statuses),
                         sum(status in terminal for status in statuses),
                         sum(status not in ('completed', 'failed', *terminal) for status in statuses),
                         (data.get('error', '') if isinstance(data, dict) else snapshot.get('error', ''))])
    lines.extend([table(['Fichier', 'État effectif', 'État JSON brut', 'Tours achevés', 'Échecs', 'Interrompus / incomplets', 'En cours', 'Erreur'], coverage), '',
                  'Seuls les tours dont `status=completed` entrent dans les calculs. '
                  'Un garde final signalant un arrêt fait autorité pour l’affichage : un JSON resté `started` après SIGTERM '
                  'est présenté comme interrompu, sans être réécrit ni admis dans les agrégats achevés. '
                  'Une sortie arrêtée par `max_new_tokens` est intégralement conservée, mais sa complétude n’est pas présumée. '
                  'Les diagnostics longs, le pilote et l’ablation sont séparés de la campagne principale. '
                  'Le pilote A1/A2 à 96 tokens a motivé un amendement : reprise des 21 mêmes demandes à 256 tokens par tour. '
                  'Ce choix est postérieur au pilote, pas un paramètre fixé à l’aveugle.', '',
                  'La campagne principale combine les tours achevés de `campaign.json` avant l’arrêt D2 et de '
                  '`campaign-resumed.json` après correction du routeur, à raison d’une observation par ID. '
                  'La copie `campaign-before-router-fix.json` reste une archive et n’ajoute aucune observation. '
                  'Le fragment D2 échoué est conservé comme diagnostic ; seul D2 achevé de la reprise entre dans les 21 tours.', '',
                  table(['Périmètre', 'Observations achevées', 'Cible annoncée'], [
                      ['Principal, IDs uniques', len(main_turns), 21],
                      ['Supplément A4, raisonnement conditionnel', len(supplemental), 1],
                      ['Principal + A4, en explicitant le supplément', len(main_turns) + len(supplemental), 22],
                  ]), '',
                  'IDs principaux encore sans réponse achevée : ' +
                  (', '.join(identifier for identifier in PRIMARY_IDS if identifier not in {turn.get('id') for turn in main_turns}) or 'aucun') + '.', '',
                  '## Définitions et limites de mesure', '',
                  '- TTFT moteur : délai de la première passe de modèle jusqu’au premier token synchronisé ; tokenisation exclue.',
                  '- TTFT perçue encadrée : borne basse = TTFT moteur plus lancement/imports/initialisation pour la première requête du processus. '
                  'La borne haute ajoute tous les frais hors temps moteur : max(0, temps mural requête − temps moteur). '
                  'Cet intervalle couvre la préparation omise de la borne basse sans prétendre mesurer exactement son partage avant/après le premier token.',
                  '- Total perçu enregistré : lancement→prêt plus temps mural de la requête pour la première requête ; '
                  'temps mural de la requête seule ensuite. Le périmètre est cette somme, hors gestes humains et temps entre requêtes.',
                  '- « Premier processus » désigne la première requête de ce lancement ; cela ne prouve pas un cache SSD froid.',
                  '- Temps total par tour : temps mural instrumenté de la requête, incluant sauvegarde des tokens et échantillonnage. '
                  'Le débit de décodage exclut le premier token, mais inclut le callback de sauvegarde dans son dénominateur.',
                  '- Débits agrégés : rapports des sommes ; aucune moyenne non pondérée des débits individuels.',
                  '- Lectures logiques : octets demandés par le lecteur. Stockage : compteur du processus entier, réadahead inclus, '
                  'sans certification de trafic NAND ni attribution à un fichier ou un SSD.',
                  '- RSS : maximum observé par échantillonnage, environ 50 ms. Swap : croissance maximale du swap système durant le tour ; '
                  'elle peut provenir d’autres applications et ne s’additionne pas entre tours.', '',
                  '## Mesures par tour achevé', ''])
    rows = []
    for name, snapshot in runs.items():
        for turn in completed(snapshot):
            memory = turn.get('memory', {})
            lower_ttft, upper_ttft = perceived_ttft_bounds(turn)
            rows.append([name.removesuffix('.json'), turn.get('domain'), turn.get('id'),
                         len(turn.get('generated_ids', [])), fmt(turn.get('ttft_seconds')),
                         'premier processus' if turn.get('cold_first_request') else 'suivant',
                         fmt(lower_ttft), fmt(upper_ttft), fmt(total(turn)),
                         fmt(turn.get('perceived_wait_seconds')), fmt(turn.get('decode_tokens_per_second')),
                         fmt(turn.get('logical_read_bytes', 0) / 2**30) if number(turn.get('logical_read_bytes')) else '—',
                         fmt(turn.get('process_storage_read_bytes', 0) / 2**30) if number(turn.get('process_storage_read_bytes')) else '—',
                         fmt(memory.get('sampled_peak_rss_bytes', 0) / 2**20) if number(memory.get('sampled_peak_rss_bytes')) else '—',
                         fmt(memory.get('maximum_swap_growth_bytes', 0) / 2**20) if number(memory.get('maximum_swap_growth_bytes')) else '—',
                         turn.get('stop_reason'), yes(turn.get('finite_logits_checked'))])
    lines.append(table(['Run', 'Domaine', 'Tour', 'Tokens sortie', 'TTFT moteur s', 'Rang processus', 'TTFT perçue basse s', 'TTFT perçue haute s',
                        'Total s', 'Total perçu s', 'Decode tok/s', 'Logique GiB', 'Stockage GiB',
                        'RSS MiB', 'Swap Δ MiB', 'Arrêt', 'Logits finis vérifiés'], rows) if rows else 'Aucun tour achevé disponible.')
    lines.extend(['', '## Agrégats par exécution et domaine', ''])
    rows = []
    aggregate_sets = [('PRINCIPAL_21_ID_uniques', main_turns)]
    if supplemental:
        aggregate_sets.extend([('SUPPLEMENT_A4', supplemental),
                               ('PRINCIPAL_plus_A4_cible22', main_turns + supplemental)])
    aggregate_sets.extend((name, completed(snapshot)) for name, snapshot in runs.items())
    for name, turns in aggregate_sets:
        if not turns:
            continue
        groups = defaultdict(list)
        for turn in turns:
            groups[turn.get('domain', 'non précisé')].append(turn)
        for domain, grouped in [('GLOBAL', turns), *sorted(groups.items())]:
            values = aggregate(grouped)
            rows.append([name.removesuffix('.json'), domain, values['turns'], values['tokens'],
                         fmt(values['wall_seconds']), fmt(values['tokens_per_second']),
                         fmt(values['decode_tokens_per_second']),
                         fmt(values['logical_read_bytes'] / 2**30) if values['logical_read_bytes'] is not None else '—',
                         fmt(values['storage_read_bytes'] / 2**30) if values['storage_read_bytes'] is not None else '—',
                         f"{values['storage_coverage']}/{values['turns']}",
                         fmt(values['peak_rss_bytes'] / 2**20) if values['peak_rss_bytes'] is not None else '—',
                         fmt(values['max_swap_growth_bytes'] / 2**20) if values['max_swap_growth_bytes'] is not None else '—'])
    lines.append(table(['Run', 'Domaine', 'Tours', 'Tokens', 'Σ total s', 'Σ tokens/Σ total', 'Σ decode/Σ durée decode',
                        'Σ logique GiB', 'Σ stockage GiB', 'Couverture stockage', 'RSS max MiB', 'Swap Δ max MiB'], rows)
                 if rows else 'Agrégats en attente de tours achevés.')
    if duplicates:
        lines.extend(['', 'Doublons achevés écartés de l’agrégat principal (priorité déterministe au premier fichier de phase) :', '',
                      table(['ID', 'Source conservée', 'Autre source', 'Sortie IDs identique'],
                            [[item['id'], item['kept_source'], item['other_source'], yes(item['same_ids'])] for item in duplicates])])

    lines.extend(['', *d2_sections(runs), '', '## Repli de la gate du routeur sur CPU — compteurs enregistrés', '',
                  'Les compteurs des poids sont cumulatifs dans chaque processus. Les deltas ci-dessous sont calculés entre '
                  'tours achevés successifs de la même exécution ; le premier part de zéro. '
                  'Un tour échoué entre deux observations peut donc contribuer au delta suivant. '
                  'Une absence de compteur sur l’ancien moteur n’est pas transformée en mesure nulle. '
                  'Lorsqu’une égalité déclenche le repli, la gate CPU complète est recalculée sur tout le chunk courant '
                  '(F.linear → softmax → topk → normalisation), en FP32 dans la configuration évaluée ; '
                  'seuls les indices et scores des lignes affectées remplacent ceux du routage natif. '
                  'La capture D2 montre un écart FP32 entre Metal et Torch CPU dès les logits, avant le softmax : '
                  'réordonner les seules probabilités natives ne reproduirait donc pas cette gate de référence. '
                  'La référence est le Torch CPU local et versionné, sans garantie de départage identique entre versions ou avec CUDA. '
                  'Les experts et le reste du modèle continuent d’utiliser le moteur natif.', ''])
    fallback_rows = router_counter_rows(runs)
    lines.append(table(['Run', 'Tour', 'Appels repli Δ', 'Appels repli cumul', 'Lignes repli cumul',
                        'Temps repli Δ s', 'Import réf cumul s', 'Version réf', 'Appareil réf'], fallback_rows)
                 if fallback_rows else 'Aucun compteur de repli disponible dans les réponses achevées de cet instantané.')
    lines.extend(['', *concision_sections(runs, main_records)])

    lines.extend(['', '## Échelle de génération longue', '',
                  'Comparaison des IDs réellement générés. Les répétitions de fenêtres de 8 et 16 tokens sont des observations '
                  'descriptives ; elles ne donnent aucune note automatique de qualité ou de cohérence.', ''])
    ladder = []
    ladder_rows = []
    for cap in (128, 256, 512, 1024):
        snapshot = runs[f'long-{cap}.json']
        for turn in completed(snapshot):
            ladder.append((cap, snapshot, turn))
            ladder_rows.append([cap, 'achevé', len(turn.get('generated_ids', [])),
                                len(re.findall(r'\b\w+(?:[’\u0027]\w+)*\b', turn.get('text', ''))),
                                turn.get('stop_reason'), fmt(total(turn)), repetition(turn, 8), repetition(turn, 16)])
        for turn in (snapshot.get('data') or {}).get('turns', []):
            status = effective_status(snapshot, turn)
            if status in ('interrompu_garde', 'interrompu_processus', 'termine_incomplet', 'failed'):
                guard = snapshot.get('guard', {}).get('data') or {}
                ladder_rows.append([cap, status + ' — non validé', len(turn.get('partial_generated_ids', [])),
                                    len(re.findall(r'\b\w+(?:[’\u0027]\w+)*\b', turn.get('partial_text', ''))),
                                    guard.get('violation') or turn.get('error') or 'sortie incomplète',
                                    '—', '—', '—'])
    if ladder_rows:
        lines.append(table(['Plafond', 'Statut', 'Tokens observés', 'Mots observés', 'Arrêt', 'Total achevé s', 'Fenêtres 8 répétées', 'Fenêtres 16 répétées'], ladder_rows))
        lines.extend(['', 'Les fragments interrompus ne valident pas leur palier ; aucun temps de réponse complet, '
                      'débit achevé ou comparaison de préfixe de réponse achevée n’est calculé à partir de ces fragments.'])
    if ladder:
        pairs = []
        for index, (cap, snapshot, shorter) in enumerate(ladder):
            for larger_cap, larger_snapshot, longer in ladder[index + 1:]:
                a = shorter.get('generated_ids', [])
                b = longer.get('generated_ids', [])
                pairs.append([f'{cap} → {larger_cap}', yes(shorter.get('full_input_ids') == longer.get('full_input_ids')),
                              yes(snapshot['data'].get('identity', {}).get('sources') == larger_snapshot['data'].get('identity', {}).get('sources')),
                              len(a), yes(len(b) >= len(a) and b[:len(a)] == a)])
        if pairs:
            lines.extend(['', table(['Paire', 'Entrée IDs identique', 'Sources identiques', 'Longueur comparée', 'Préfixe sortie exact'], pairs)])
    elif not ladder_rows:
        lines.append('Aucun palier long achevé disponible ; aucun débit long ni égalité de préfixe ne peut être conclu.')

    lines.extend(['', '## Ablation de la conversation E — réemploi KV / reconstruction', '',
                  'Les deux exécutions doivent conserver les mêmes messages réellement générés. L’égalité est testée sur les IDs, '
                  'et non sur une ressemblance du texte. Les gains ci-dessous sont des rapports observés ; '
                  'l’ordre des tâches et l’état initial du cache de poids peuvent différer entre processus. '
                  'Ils ne suffisent pas à attribuer causalement tout l’écart au seul KV.', ''])
    base = {turn.get('id'): turn for _, turn in main_records}
    rebuilt = {turn.get('id'): turn for turn in completed(runs['conversation-rebuild.json'])}
    pairs = []
    gain_base = []
    gain_rebuild = []
    for turn_id in ('E1', 'E2', 'E3', 'E4', 'E5'):
        left, right = base.get(turn_id), rebuilt.get(turn_id)
        if not left or not right:
            pairs.append([turn_id, 'en attente', '—', '—', '—', '—', '—', '—', '—'])
            continue
        left_wall, right_wall = total(left), total(right)
        gain = right_wall / left_wall if number(left_wall) and number(right_wall) and left_wall > 0 else None
        reduction = 100 * (1 - left_wall / right_wall) if number(left_wall) and number(right_wall) and right_wall > 0 else None
        same_input = left.get('full_input_ids') == right.get('full_input_ids')
        same_output = left.get('generated_ids') == right.get('generated_ids')
        pairs.append([turn_id, yes(same_input), yes(same_output), left.get('reused_prefix_tokens'),
                      fmt(left_wall), fmt(right_wall), fmt(gain), fmt(reduction),
                      fmt(right.get('ttft_seconds') / left['ttft_seconds'])
                      if number(right.get('ttft_seconds')) and number(left.get('ttft_seconds')) and left['ttft_seconds'] > 0 else '—'])
        if turn_id != 'E1' and same_input and same_output:
            gain_base.append(left)
            gain_rebuild.append(right)
    lines.append(table(['Tour', 'Entrée exacte', 'Sortie exacte', 'Préfixe KV réutilisé', 'KV total s',
                        'Rebuild total s', 'Rebuild/KV ×', 'Réduction observée %', 'TTFT rebuild/KV ×'], pairs))
    if gain_base:
        left = aggregate(gain_base)['wall_seconds']
        right = aggregate(gain_rebuild)['wall_seconds']
        lines.extend(['', f"Paires E2–E5 à entrée et sortie identiques : {len(gain_base)}/4. "
                      f'Rapport des sommes de temps rebuild/KV : {fmt(right / left) if left and right else "—"} × '
                      f'(Σ KV {fmt(left)} s ; Σ rebuild {fmt(right)} s).'])
    lines.extend(['', *review_sections(runs, checks, review)])
    if pilot_review is not None and isinstance(pilot_review.get('data'), dict):
        pilot_sections = review_sections(runs, {'data': None}, pilot_review)
        start = pilot_sections.index('## Évaluation indépendante')
        pilot_sections[start] = '## Évaluation indépendante du pilote — distincte de la reprise'
        lines.extend(['', *pilot_sections[start:]])
    lines.extend(annotation_appendices(runs, evidence))
    lines.extend(['', '## Supervision des exécutions', ''])
    guard_rows = []
    for name, snapshot in runs.items():
        guard = snapshot.get('guard', {}).get('data')
        if isinstance(guard, dict):
            guard_rows.append([name, guard.get('exit_code'), guard.get('violation') or 'aucune déclarée',
                               guard.get('exception', ''), fmt(guard.get('elapsed_seconds')),
                               fmt(guard['sampled_peak_rss_bytes'] / 2**20) if number(guard.get('sampled_peak_rss_bytes')) else '—',
                               fmt(guard['minimum_available_bytes'] / 2**30) if number(guard.get('minimum_available_bytes')) else '—',
                               fmt(guard['maximum_swap_growth_bytes'] / 2**20) if number(guard.get('maximum_swap_growth_bytes')) else '—'])
    lines.append(table(['Run', 'Sortie processus', 'Arrêt du garde', 'Exception', 'Durée s', 'RSS arbre max MiB',
                        'Disponible min GiB', 'Swap Δ max MiB'], guard_rows) if guard_rows
                 else 'Aucun rapport final de supervision disponible pour ces exécutions. Un run en cours ne confirme pas encore ses limites finales.')
    lines.extend(['', '## Traçabilité des instantanés', ''])
    fingerprints = [[display_path(snapshot['path']), snapshot['status'], snapshot['sha256']]
                    for snapshot in [*runs.values(), *(run['guard'] for run in runs.values() if 'guard' in run), checks, review,
                                     *([pilot_review] if pilot_review is not None else [])]]
    lines.extend([table(['Fichier lu', 'Lecture', 'SHA-256 des octets lus'], fingerprints), '',
                  'Les empreintes suivantes comparent les sources déclarées au lancement avec les fichiers actuels. '
                  'Une divergence ne réécrit pas l’histoire du run ; une archive concordante est signalée lorsqu’elle existe.', ''])
    rows = []
    mismatches = []
    for name, snapshot in runs.items():
        checks_source = source_checks(snapshot, evidence)
        if not checks_source:
            continue
        rows.append([name, len(checks_source), sum(check['matches'] for check in checks_source),
                     sum(check['current'] is None for check in checks_source),
                     sum(not check['matches'] and check['current'] is not None for check in checks_source)])
        for check in checks_source:
            if not check['matches']:
                mismatches.append([name, check['path'], check['expected'], check['current'], ', '.join(check['archives']) or 'aucune'])
    lines.append(table(['Run', 'Sources déclarées', 'Concordantes actuellement', 'Absentes', 'Différentes'], rows)
                 if rows else 'Aucune identité de sources disponible.')
    if mismatches:
        lines.extend(['', table(['Run', 'Source', 'SHA enregistré', 'SHA actuel', 'Archive concordante'], mismatches)])
    task_rows = []
    tasks = ROOT / 'research/real_world/tasks.json'
    current_tasks_sha = sha(tasks.read_bytes()) if tasks.is_file() else None
    for name, snapshot in runs.items():
        data = snapshot.get('data')
        if isinstance(data, dict) and data.get('tasks_sha256'):
            expected = data['tasks_sha256']
            archives = [display_path(path) for path in tasks.parent.glob('*tasks*.json')
                        if path.is_file() and sha(path.read_bytes()) == expected]
            task_rows.append([name, expected, yes(expected == current_tasks_sha), ', '.join(archives) or 'aucune'])
    if task_rows:
        lines.extend(['', table(['Run', 'SHA tâches enregistré', 'Identique aux tâches actuelles', 'Copie de tâches concordante'], task_rows)])
    snapshot_paths = {snapshot['path'].resolve() for snapshot in [*runs.values(),
                      *(run['guard'] for run in runs.values() if 'guard' in run), checks, review,
                      *([pilot_review] if pilot_review is not None else [])]}
    archive_rows = []
    if evidence.is_dir():
        for path in sorted(evidence.rglob('*')):
            if path.is_file() and path.resolve() not in snapshot_paths:
                archive_rows.append([display_path(path), path.stat().st_size, sha(path.read_bytes())])
    if archive_rows:
        lines.extend(['', '### Archives, diagnostics et autres preuves', '',
                      'Empreintes complémentaires lues à la génération de ce rapport. Les archives ne sont pas comptées comme '
                      'des réponses supplémentaires ; une copie de la campagne initiale reste une copie de la même observation.', '',
                      table(['Fichier', 'Octets', 'SHA-256'], archive_rows)])
    return '\n'.join(lines).rstrip() + '\n'


def make_responses(runs, timestamp):
    lines = ['# Prompts et réponses mesurés — AirLLM Next', '',
             f'Instantané du {timestamp}. Contenu recopié sans correction des champs source.', '',
             'Les sorties achevées sont reproduites intégralement. Un arrêt par plafond de tokens peut couper la réponse. '
             'Un fragment n’est affiché que pour un tour `failed` ou un arrêt final prouvé par le garde, avec son état incomplet.', '']
    for name, snapshot in runs.items():
        lines.extend([f'## {name}', ''])
        data = snapshot.get('data')
        if not isinstance(data, dict):
            lines.extend([f"Preuve {snapshot['status']}. Aucune réponse disponible.", ''])
            continue
        lines.extend([f"État effectif de l’exécution : **{effective_status(snapshot)}** ; état JSON brut : `{data.get('status', 'inconnu')}`. "
                      f"SHA-256 du fichier lu : `{snapshot['sha256']}`.", ''])
        seen_systems = set()
        for turn in data.get('turns', []):
            status = effective_status(snapshot, turn)
            lines.extend([f"### {turn.get('id', '?')} — {turn.get('domain', 'domaine non précisé')}", '',
                          f"État effectif : **{status}** ; état JSON brut : `{turn.get('status', 'inconnu')}` ; "
                          f"plafond demandé : {turn.get('max_new_tokens', '—')} tokens.", ''])
            messages = turn.get('messages', [])
            system = next((message.get('content') for message in messages if message.get('role') == 'system'), None)
            if isinstance(system, str) and system not in seen_systems:
                seen_systems.add(system)
                lines.extend(['Message système exact :', '', fence(system), ''])
            lines.extend(['Prompt utilisateur exact :', '', fence(turn.get('prompt', '')), ''])
            if turn.get('status') == 'completed':
                lines.extend([f"Sortie intégrale mesurée : {len(turn.get('generated_ids', []))} tokens ; "
                              f"arrêt `{turn.get('stop_reason', 'non précisé')}`. "
                              f"TTFT {fmt(turn.get('ttft_seconds'))} s ; total {fmt(total(turn))} s.", '',
                              fence(turn.get('text', '')), ''])
                if isinstance(turn.get('raw_text'), str) and turn['raw_text'] != turn.get('text'):
                    lines.extend(['Décodage brut intégral, avec jeton d’arrêt et éventuelles balises :', '', fence(turn['raw_text']), ''])
            elif status in ('failed', 'interrompu_garde', 'interrompu_processus', 'termine_incomplet'):
                guard = snapshot.get('guard', {}).get('data') or {}
                reason = (guard.get('violation') or guard.get('exception') or
                          turn.get('error') or data.get('error') or f"sortie processus {guard.get('exit_code', 'inconnue')}")
                lines.extend([f"Arrêt enregistré : {reason}.", '',
                              *guard_description(snapshot),
                              '**Fragment explicitement incomplet — ne constitue pas une réponse achevée :**', '',
                              fence(turn.get('partial_text', '')), '',
                              f"IDs partiels conservés : {len(turn.get('partial_generated_ids', []))}.", ''])
            else:
                lines.extend(['Tour non achevé dans cet instantané ; aucun fragment présenté comme réponse.', ''])
    return '\n'.join(lines).rstrip() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'evidence-real-world')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/real_world')
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    runs = {name: load_snapshot(evidence / name) for name in RUN_FILES}
    for name, snapshot in runs.items():
        snapshot['guard'] = load_snapshot((evidence / name).with_suffix('.guard.json'))
    checks = load_snapshot(evidence / 'code-checks.json')
    review = load_snapshot(evidence / 'independent_review.json')
    pilot_review = load_snapshot(evidence / 'pilot_review.json')
    outputs = [args.output_dir / 'ANALYSIS_FR.md', args.output_dir / 'RESPONSES_FR.md']
    inputs = list(evidence.rglob('*')) if evidence.is_dir() else []
    inputs = [path for path in inputs if path.is_file()] + [ROOT / 'research/real_world/tasks.json']
    protect_inputs(inputs, outputs)
    timestamp = datetime.now(ZoneInfo('Europe/Paris')).strftime('%d/%m/%Y à %H:%M:%S %Z')
    analysis = make_analysis(runs, checks, review, evidence, timestamp, pilot_review)
    responses = make_responses(runs, timestamp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(outputs, (analysis, responses)):
        path.write_text(content)
    print(json.dumps({'outputs': list(map(str, outputs)),
                      'completed_turns': {name: len(completed(snapshot)) for name, snapshot in runs.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
