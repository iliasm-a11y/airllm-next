"""Strict metadata-only admission and retained-log gates for bounded dense mlock.

This does not read tensor payloads, hash the 110GB model again, or prove NAND/cache
behavior. The selector binds previously checked inventory/identity evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

FEATURE = 'bounded_dense_cpu_mlock_v1'
ENV_KEYS = ('AIRLLM_DENSE_LOCK', 'AIRLLM_DENSE_LOCK_MIB', 'AIRLLM_DENSE_LOCK_NAMES')
FAMILIES = {'hc_attn_fn.weight', 'hc_ffn_fn.weight', 'ffn_gate_inp.weight',
    'ssm_beta.weight', 'ssm_f_a.weight', 'ssm_f_b.weight', 'ssm_g_a.weight', 'ssm_g_b.weight',
    'attn_k_b.weight', 'attn_v_b.weight', 'attn_kv_a_mqa.weight', 'indexer.attn_q_b.weight',
    'indexer.attn_k.weight', 'indexer.proj.weight', 'indexer_compressor_gate.weight'}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def _stat(path):
    row = Path(path).stat()
    return {'device': row.st_dev, 'inode': row.st_ino, 'size': row.st_size,
            'mtime_ns': row.st_mtime_ns, 'ctime_ns': row.st_ctime_ns}


def lock_environment(plan):
    return {'AIRLLM_DENSE_LOCK': '1', 'AIRLLM_DENSE_LOCK_MIB': str(plan['budget_mib']),
            'AIRLLM_DENSE_LOCK_NAMES': ','.join(plan['names'])}


def lock_environment_matches(environment, plan):
    actual = {key: value for key, value in environment.items() if key.startswith('AIRLLM_DENSE_LOCK')}
    return actual == (lock_environment(plan) if plan else {})


def load_plan(path, model):
    path = Path(path).resolve()
    before = _stat(path)
    if not 0 < before['size'] <= 2 * 1024**2:
        raise ValueError('dense lock selector manifest exceeds metadata budget')
    raw = path.read_bytes()
    if before != _stat(path):
        raise ValueError('dense lock selector changed while reading')
    plan = json.loads(raw)
    if not isinstance(plan, dict) or type(plan.get('schema_version')) is not int or plan['schema_version'] != 1 or plan.get('feature') != FEATURE:
        raise ValueError('unsupported dense lock selector schema')
    mib, page = plan.get('budget_mib'), plan.get('page_size')
    if (type(mib) is not int or mib not in (128, 256, 512) or not _int(page, 1) or
            page & (page-1) or page != os.sysconf('SC_PAGESIZE')):
        raise ValueError('dense lock budget/page size invalid')
    names = plan.get('names')
    if not isinstance(names, list) or not 1 <= len(names) <= 128 or any(not isinstance(n, str) for n in names):
        raise ValueError('dense lock names must be a bounded list')
    if names != sorted(set(names)) or len(','.join(names).encode()) > 16384:
        raise ValueError('dense lock names must be sorted, unique and <=16KiB')
    for name in names:
        match = re.fullmatch(r'blk\.\d+\.(.+)', name)
        if match is None or match[1] not in FAMILIES:
            raise ValueError('tensor is outside exact F32/BF16 dense allowlist: ' + name)
    files = {p.name: _stat(p) for p in Path(model).resolve().parent.glob('*.gguf')}
    if not files or plan.get('model_file_stats') != files:
        raise ValueError('dense lock selector model file identity/stats differ')
    hashes = plan.get('model_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(files) or not all(_sha(v) for v in hashes.values()):
        raise ValueError('dense lock selector needs prior verified per-shard SHA256 binding')
    if not _sha(plan.get('identity_manifest_sha256')) or not _sha(plan.get('inventory_sha256')):
        raise ValueError('dense lock selector needs identity/inventory evidence hashes')
    if plan.get('model_revision') != '119034bc3d9d76eb41f91508e6f9fb19fde42672':
        raise ValueError('dense lock selector does not identify the original GLM revision')
    alignment = plan.get('source_alignment') or {}
    if (not _sha(alignment.get('source_plan_sha256')) or not _sha(alignment.get('operations_sha256')) or
            alignment.get('source_step') != 1 or alignment.get('names_count') != 684 or
            alignment.get('old_decode_steps') != [1, 2, 3] or alignment.get('ordered_matrix_sizes_verified') is not True):
        raise ValueError('selector lacks checked original dense operation/inventory alignment')
    if plan.get('current_full_hash_reverified') is not False:
        raise ValueError('selector must describe historical SHA binding without claiming current full hashing')
    if not isinstance(plan.get('identity_manifest_path'), str) or not plan['identity_manifest_path']:
        raise ValueError('historical checksum evidence path missing')
    identity_path = Path(plan['identity_manifest_path']).resolve()
    identity_stat = _stat(identity_path)
    if not 0 < identity_stat['size'] <= 2*1024**2:
        raise ValueError('historical checksum evidence exceeds metadata budget')
    identity_raw = identity_path.read_bytes()
    if (_stat(identity_path) != identity_stat or
            hashlib.sha256(identity_raw).hexdigest() != plan['identity_manifest_sha256']):
        raise ValueError('historical checksum evidence changed')
    historical = json.loads(identity_raw)
    history_rows = historical.get('files', [])
    history = {row.get('name'): row for row in history_rows}
    if (historical.get('integrity_verified') is not True or historical.get('revision') != plan['model_revision'] or
            historical.get('quantization') != 'Q2_K_S' or set(history) != set(files) or len(history) != len(history_rows)):
        raise ValueError('historical checksum manifest does not identify original verified model')
    history_scope = {}
    for filename, current in files.items():
        row = history[filename]
        identity = row.get('identity')
        if (row.get('integrity_verified') is not True or row.get('actual_sha256') != hashes[filename] or
                row.get('sha256') != hashes[filename] or row.get('bytes') != current['size'] or
                not isinstance(identity, list) or len(identity) != 5 or
                identity[1:4] != [current['inode'], current['size'], current['mtime_ns']]):
            raise ValueError('current metadata no longer links to historical checksum evidence')
        history_scope[filename] = {'historical_device': identity[0], 'current_device': current['device'],
            'historical_ctime_ns': identity[4], 'current_ctime_ns': current['ctime_ns'],
            'inode_size_mtime_match': True, 'current_full_sha256_recomputed': False,
            'complete_five_field_historical_identity_not_proven': True}
    tensors = plan.get('selected_tensors')
    if (not isinstance(tensors, list) or len(tensors) != len(names) or
            any(not isinstance(row, dict) for row in tensors) or [r.get('name') for r in tensors] != names):
        raise ValueError('selected tensor rows differ from exact ordered names')
    raw_bytes, spans = 0, {}
    for row in tensors:
        filename, dtype, dims = row.get('file'), row.get('type'), row.get('dimensions')
        start, size = row.get('absolute_start'), row.get('payload_bytes')
        if filename not in files or dtype not in ('F32', 'BF16') or not isinstance(dims, list) or not 1 <= len(dims) <= 4:
            raise ValueError('invalid selected tensor file/type/dimensions')
        if any(not _int(d, 1) for d in dims) or not _int(start) or not _int(size, 1) or not _int(row.get('tensor_offset')):
            raise ValueError('invalid selected tensor offsets or lengths')
        elements = 1
        for dimension in dims:
            elements *= dimension
        if elements * (4 if dtype == 'F32' else 2) != size or start < row['tensor_offset'] or start+size > files[filename]['size']:
            raise ValueError('tensor span/dtype size exceeds original GGUF')
        low, high = start-start%page, ((start+size+page-1)//page)*page
        if row.get('page_start') != low or row.get('page_end') != high:
            raise ValueError('tensor full-page span does not match offsets')
        raw_bytes += size
        spans.setdefault(filename, []).append((low, high))
    unions = []
    for filename in sorted(spans):
        merged = []
        for low, high in sorted(spans[filename]):
            if merged and low <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
            else:
                merged.append((low, high))
        unions.extend({'file': filename, 'start': low, 'end': high} for low, high in merged)
    union_bytes = sum(row['end']-row['start'] for row in unions)
    if (plan.get('page_unions') != unions or plan.get('selected_raw_bytes') != raw_bytes or
            plan.get('selected_page_union_bytes') != union_bytes or not 0 < union_bytes <= mib*1024**2):
        raise ValueError('dense lock full-page union accounting does not match bounded budget')
    return {'path': str(path), 'stat': before, 'sha256': hashlib.sha256(raw).hexdigest(),
            'budget_mib': mib, 'page_size': page, 'names': names, 'selected_raw_bytes': raw_bytes,
            'selected_page_union_bytes': union_bytes, 'model_file_stats': files,
            'model_sha256': hashes, 'identity_manifest_sha256': plan['identity_manifest_sha256'],
            'identity_manifest_path': str(identity_path), 'identity_manifest_stat': identity_stat,
            'historical_identity_scope': history_scope,
            'inventory_sha256': plan['inventory_sha256'],
            'integrity_scope': ('current selector/file stats plus historical verified SHA/inventory binding; '
                'device/ctime may differ; no current full model payload SHA or five-field identity proof'),
            'manifest': plan}


def check_plan(plan, model):
    current = load_plan(plan['path'], model)
    if (current['sha256'] != plan['sha256'] or current['stat'] != plan['stat'] or
            current['identity_manifest_stat'] != plan['identity_manifest_stat']):
        raise ValueError('dense lock selector identity changed after planning')
    return {'passed': True, 'sha256': current['sha256'], 'scope': current['integrity_scope']}


def _retained_rows(rows, plan):
    expected = {'names': len(plan['names']), 'raw_bytes': plan['selected_raw_bytes'],
        'page_union_bytes': plan['selected_page_union_bytes'], 'locked_bytes': plan['selected_page_union_bytes'],
        'verified_pages_at_lock': plan['selected_page_union_bytes']//plan['page_size'], 'error': 0, 'unlock_error': 0}
    if not isinstance(rows, list) or not rows or len(rows) > 16 or any(not isinstance(row, dict) for row in rows):
        return False, expected
    retained = [r for r in rows if r.get('status') == 'retained' and r.get('retained') == '1']
    passed = len(retained) == 1 and all(r.get('status') in ('retained', 'released') for r in rows)
    if passed:
        passed = all(retained[0].get(key) == str(value) for key, value in expected.items())
        calls = retained[0].get('lock_calls', '')
        passed = passed and isinstance(calls, str) and calls.isdigit() and 0 < int(calls) <= len(plan['names'])
    return bool(passed), expected


def retained_log(path, plan):
    rows = []
    path = Path(path)
    if not path.is_file():
        return {'passed': False, 'reason': 'missing native dense lock log', 'rows': rows}
    if path.stat().st_size > 128*1024**2:
        return {'passed': False, 'reason': 'native dense lock log exceeds parsing budget', 'rows': rows}
    with path.open(errors='replace') as stream:
        while line := stream.readline(16385):
            if len(line) > 16384:
                return {'passed': False, 'reason': 'native log line exceeds parsing budget', 'rows': rows}
            if 'AIRLLM_DENSE_LOCK ' in line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line.split('AIRLLM_DENSE_LOCK ', 1)[1]))
                rows.append(fields)
                if len(rows) > 16:
                    return {'passed': False, 'reason': 'ambiguous repeated dense lock records', 'rows': rows}
    passed, expected = _retained_rows(rows, plan)
    return {'passed': bool(passed), 'rows': rows, 'expected': expected, 'plan_sha256': plan['sha256'],
            'reason': None if passed else 'native log did not prove exact retained names/full-page union',
            'scope': 'verified residency at acquisition and retained user locks; not a speed or NAND claim'}


def evidence_matches(evidence, actual_plan, expected_plan):
    valid_rows, expected = _retained_rows(evidence.get('rows') if isinstance(evidence, dict) else None, expected_plan)
    return bool(isinstance(evidence, dict) and evidence.get('passed') is True and valid_rows and
                isinstance(actual_plan, dict) and actual_plan.get('sha256') == expected_plan['sha256'] and
                evidence.get('plan_sha256') == expected_plan['sha256'] and
                evidence.get('expected') == expected)
