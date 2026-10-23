#!/usr/bin/env python3
"""Independent stdlib-only arithmetic/fixture validation; no model execution.

The implementations below are the evaluator's own reference calculations, not
extracted or repaired model output. This program never modifies tasks.json.
"""
import argparse
import copy
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def stable_first_occurrences(values):
    # Compare with all earlier input positions, independently of an accumulator.
    return [value for index, value in enumerate(values)
            if not any(value == previous for previous in values[:index])]


def even_sum(values):
    return sum(value for value in values if value % 2 == 0)


def validate(tasks_path):
    raw = tasks_path.read_bytes()
    sessions = json.loads(raw)
    turns = {turn['id']: turn for session in sessions for turn in session['turns']}
    solutions = []
    for lina, marc, nora in permutations(range(3)):
        if lina < marc and lina != 0 and nora != 2 and marc != 1:
            solutions.append({'Lina': ('lundi', 'mardi', 'mercredi')[lina],
                              'Marc': ('lundi', 'mardi', 'mercredi')[marc],
                              'Nora': ('lundi', 'mardi', 'mercredi')[nora]})
    assert solutions == [{'Lina': 'mardi', 'Marc': 'mercredi', 'Nora': 'lundi'}]
    D = Decimal
    kits = D(18) * D(24)
    discounted = kits * (1 - D(15) / 100)
    a2_total = discounted + D(36)
    assert a2_total == D('403.20') and D(450) - a2_total == D('46.80')
    d2_total = 480 + 24 * 18 + 720
    assert d2_total == 1632 and 1800 - d2_total == 168
    e_total = 8 * 14 + 12
    assert e_total == 124 and e_total - 120 == 4
    assert D(8) * D('13.50') + 12 == 120 and 8 * 14 + 8 == 120
    contributions = {'A': 12 * (25 - 15), 'B': 8 * (40 - 22)}
    revenue = 12 * 25 + 8 * 40
    margin = sum(contributions.values())
    assert (revenue, margin, margin - 90) == (620, 264, 174)
    options = [{'option': name, 'cost_for_requested_10': 10 * price + delivery,
                'capacity': capacity, 'feasible_for_10': capacity >= 10 and 10 * price + delivery <= 500}
               for name, price, delivery, capacity in [('A', 38, 80, 10), ('B', 44, 20, 12), ('C', 35, 40, 8)]]
    winner = min((option for option in options if option['feasible_for_10']),
                 key=lambda option: (option['cost_for_requested_10'], -option['capacity']))
    assert winner['option'] == 'B'
    strict_json = turns['G4']['checks']['json_exact']
    assert set(strict_json) == {'projet', 'total_euros', 'confirme'}
    assert strict_json['projet'] == 'Iris'
    assert type(strict_json['total_euros']) in (int, float) and strict_json['total_euros'] == 3 * 17 == 51
    assert type(strict_json['confirme']) is bool and strict_json['confirme'] is False
    assert 7 * 8 == 56
    start, end = datetime(2000, 1, 1, 10), datetime(2000, 1, 1, 17)
    pause_start, pause_end = datetime(2000, 1, 1, 12, 30), datetime(2000, 1, 1, 13, 30)
    slots = []
    while start + timedelta(minutes=30) <= end:
        if start >= pause_end or start + timedelta(minutes=30) <= pause_start:
            slots.append(start.strftime('%H:%M'))
        start += timedelta(minutes=30)
    assert len(slots) == 12 and len(slots) * 4 == 48
    long_expenses = sum((240, 90, 160, 60))
    assert long_expenses == 550 and 650 - long_expenses == 100
    long_prompts = [turns[f'L{cap}']['prompt'] for cap in (128, 256, 512, 1024)]
    assert len(set(long_prompts)) == 1
    code_checks = []
    for identifier, function in [('B1', stable_first_occurrences), ('B2', even_sum)]:
        cases = []
        for case in turns[identifier]['checks']['python_tests']['cases']:
            args = copy.deepcopy(case['args'])
            before = copy.deepcopy(args)
            actual = function(*args)
            entry = {'args': before, 'expected': case['expected'], 'independently_computed': actual,
                     'matches': actual == case['expected'], 'input_unchanged': args == before}
            assert entry['matches'] and entry['input_unchanged']
            cases.append(entry)
        code_checks.append({'id': identifier, 'cases': cases})
    d2_regex = turns['D2']['checks']['regex_all'][0]
    regex_formats = {value: bool(re.search(d2_regex, value)) for value in ('1632', '1 632', '1\u00a0632')}
    assert all(regex_formats.values())
    return {
        'kind': 'independent_fixture_ground_truth_validation',
        'source_tasks': str(tasks_path), 'tasks_sha256': hashlib.sha256(raw).hexdigest(),
        'scope': 'Exact input prompts read by evaluator; own stdlib reference calculations, never model output.',
        'turn_count': len(turns),
        'prompt_sha256': {identifier: hashlib.sha256(turn['prompt'].encode()).hexdigest() for identifier, turn in turns.items()},
        'A1': {'permutations_checked': 6, 'solutions': solutions},
        'A2': {'kits_before_discount': str(kits), 'kits_after_discount': str(discounted),
               'total': str(a2_total), 'remaining_budget': str(D(450) - a2_total)},
        'D2': {'meals': 24 * 18, 'total': d2_total, 'remaining_budget': 1800 - d2_total,
               'regex_formats_accepted': regex_formats},
        'E4_E5': {'people': 8, 'formula': 14, 'delivery': 12, 'total': e_total, 'over_budget': 4,
                  'individually_valid_alternatives': [{'formula': '13.50', 'delivery': 12, 'total': '120.00'},
                                                     {'formula': 14, 'delivery': 8, 'total': 120}]},
        'F1': {'revenue': revenue, 'contributions': contributions, 'margin': margin, 'result': margin - 90},
        'F2': {'options': options, 'winner': winner['option'],
               'C_cost_for_available_8_people': 8 * 35 + 40},
        'G4': strict_json, 'G5': {'product': 7 * 8},
        'L': {'expenses': long_expenses, 'remaining_budget': 650 - long_expenses, 'slot_start_times': slots,
              'slot_count': len(slots), 'theoretical_visitors_if_one_slot_each': len(slots) * 4,
              'stated_visitor_limit': 40, 'repair_count': None, 'identical_prompts_across_caps': True},
        'code_case_validation': code_checks,
        'annotations': [
            'D2: written speaker confirmation and allergy details are the intended open dependencies; unspecified wording also permits mentioning the undecided report format.',
            'F2: C=390 is a hypothetical quote for all 10 despite insufficient capacity; C=320 for the available 8 is defensible if clearly labelled infeasible for 10. B remains the unique choice.',
            'G3: Julie and Emma are possible antecedents, not necessarily an exhaustive list of external referents.',
            'B1: declared pass conditions do not enforce returned-list identity; the checker separately records an identity diagnostic. Mixed/nested input coverage remains finite.',
            'B2: input nonmutation is tested but not explicitly requested in the prompt.',
            'Text contains/regex checks are hints; semantic equivalence and alternative valid proposals must not be rejected solely for missing strings.',
        ],
        'status': 'All independently recomputed numeric values and eight declared code test outputs agree; semantic qualifications recorded separately.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=ROOT / 'research/real_world/tasks.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and (args.output.resolve() == args.tasks.resolve() or
                        (args.output.exists() and args.tasks.exists() and args.output.samefile(args.tasks))):
        parser.error('Output aliases protected tasks input')
    result = validate(args.tasks)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({'status': result['status'], 'output': str(args.output),
                          'tasks_sha256': result['tasks_sha256']}, ensure_ascii=False))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
