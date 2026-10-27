#!/usr/bin/env python3
"""Read-only macOS memory inventory; never imports MLX or changes resources.

RSS rankings and physical-footprint accounting are separate observations. The
tool does not diagnose a process as abandoned from its name or parent PID.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from m5_process_inventory import inventory, expected_read_error


def command(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(command=argv, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=argv, error=f"{type(exc).__name__}: {exc}")


def vm_pages(raw):
    match = re.search(r"page size of (\d+) bytes", raw)
    size = int(match.group(1)) if match else None
    counts = {}
    for line in raw.splitlines()[1:]:
        match = re.fullmatch(r'\s*(.+?):\s+(\d+)\.?\s*', line)
        if match:
            counts[match.group(1).strip('"')] = int(match.group(2))
    keys = {
        'free_bytes': 'Pages free', 'active_bytes': 'Pages active',
        'inactive_bytes': 'Pages inactive', 'speculative_bytes': 'Pages speculative',
        'wired_bytes': 'Pages wired down', 'purgeable_bytes': 'Pages purgeable',
        'file_backed_resident_bytes': 'File-backed pages',
        'anonymous_resident_bytes': 'Anonymous pages',
        'compressor_physical_bytes': 'Pages occupied by compressor',
        'compressor_logical_uncompressed_bytes': 'Pages stored in compressor',
    }
    return dict(page_size_bytes=size, counts=counts,
                derived_bytes={name: counts[key] * size for name, key in keys.items()
                               if key in counts and size})


def pressure():
    raw = command(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'])
    try:
        return int(raw['stdout'].strip())
    except (KeyError, ValueError):
        return None


def snapshot():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                virtual_memory=memory._asdict(), swap_memory=swap._asdict(),
                pressure_level=pressure())


def process_inventory(include_command_lines=False, *, diagnostics=None):
    """Enrich trusted identities with optional per-process accounting fields."""
    rows = []
    observed = inventory(include_commands=True)
    errors = list(observed['errors'])
    for identity in observed['processes']:
        try:
            proc = psutil.Process(identity['pid'])
            if proc.create_time() != identity['created']:
                errors.append(dict(pid=identity['pid'],reason='pid_reused_before_accounting'))
                continue
            accounting = {}
            for field in ('ppid','status','memory_info'):
                try:
                    accounting[field] = getattr(proc,field)()
                except Exception as error:
                    if not expected_read_error(error):
                        raise
                    errors.append(dict(pid=proc.pid,field=field,error_type=type(error).__name__))
                    accounting[field] = None
            if psutil.Process(proc.pid).create_time() != identity['created']:
                errors.append(dict(pid=proc.pid,reason='pid_reused_during_accounting'))
                continue
            cmd = identity['command'] or []
            executable = identity['executable'] or ''
            full = ' '.join(cmd)
            if executable in ('node', 'npm', 'npx') or executable.startswith('npm '):
                category = 'node/npm'
            elif 'ChatGPT.app/' in full or 'Codex.app/' in full:
                category = 'Codex/ChatGPT'
            else:
                category = 'other'
            row = dict(pid=identity['pid'], ppid=accounting['ppid'], name=executable,
                       exe=executable, create_time=identity['created'],
                       status=accounting['status'], rss_bytes=(None if accounting['memory_info'] is None
                                                              else accounting['memory_info'].rss),
                       category=category, inference_candidate=identity['inference_candidate'])
            if include_command_lines:
                row.update(command=cmd, full_executable=identity['full_executable'])
            rows.append(row)
        except Exception as error:
            if not expected_read_error(error):
                raise
            errors.append(dict(pid=identity['pid'],field='accounting_identity',error_type=type(error).__name__))
    rows.sort(key=lambda row: row['rss_bytes'] if row['rss_bytes'] is not None else -1, reverse=True)
    if diagnostics is not None:
        diagnostics.update(errors=errors,uncertain=observed['uncertain'],
                           complete=observed['complete'],trusted_identity_count=len(observed['processes']))
    return rows


def integer_expr(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Mult, ast.Pow, ast.LShift):
        op = {ast.Mult: operator.mul, ast.Pow: operator.pow, ast.LShift: operator.lshift}[type(node.op)]
        return op(integer_expr(node.left), integer_expr(node.right))
    raise ValueError('Not an integer literal expression')


def configured_budgets():
    result = {}
    for relative in ('air_llm/airllm/nextgen/experiment_guard.py',
                     'benchmarks/benchmark_research51.py'):
        data = (ROOT / relative).read_bytes()
        tree = ast.parse(data.decode())
        values = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ExperimentLimits':
                for member in node.body:
                    if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                        try:
                            values[member.target.id] = integer_expr(member.value)
                        except ValueError:
                            pass
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'NextConfig':
                for keyword in node.keywords:
                    try:
                        values[keyword.arg] = integer_expr(keyword.value)
                    except ValueError:
                        pass
        result[relative] = dict(sha256=hashlib.sha256(data).hexdigest(),
                                literal_integer_budgets=values)
    return result


def collect(footprint_count=5, include_command_lines=False):
    before = snapshot()
    raw_vm = command(['vm_stat'])
    inventory_diagnostics = {}
    rows = process_inventory(include_command_lines,diagnostics=inventory_diagnostics)
    groups = {}
    for row in rows:
        group = groups.setdefault(row['category'], dict(count=0, summed_rss_bytes=0,unavailable_rss_count=0))
        group['count'] += 1
        if row['rss_bytes'] is None:
            group['unavailable_rss_count'] += 1
        else:
            group['summed_rss_bytes'] += row['rss_bytes']
    targets = [row for row in rows if row['pid'] != os.getpid()][:footprint_count]
    footprint = command(['footprint', '-f', 'bytes', '--noCategories', '--swapped', '--wired',
                         *[str(row['pid']) for row in targets]]) if targets else None
    identities_after = {}
    for row in targets:
        try:
            identities_after[str(row['pid'])] = psutil.Process(row['pid']).create_time() == row['create_time']
        except Exception as error:
            if not expected_read_error(error):
                raise
            identities_after[str(row['pid'])] = False
    budgets = configured_budgets()
    guard = budgets['air_llm/airllm/nextgen/experiment_guard.py']['literal_integer_budgets']
    available = before['virtual_memory']['available']
    reasons = []
    if before['pressure_level'] != 1:
        reasons.append('macOS pressure is not verified normal')
    required = guard['reserve_bytes'] + guard['admitted_payload_bytes']
    if available < required:
        reasons.append('available memory below payload plus system reserve')
    raw_system = dict(
        raw_memory_pressure=command(['memory_pressure', '-Q']),
        raw_swap=command(['sysctl', 'vm.swapusage']),
        hardware=command(['sysctl', 'hw.memsize', 'hw.model', 'machdep.cpu.brand_string']),
        physical_disks=command(['diskutil', 'list', 'physical']),
        volumes=command(['ls', '-la', '/Volumes']), disk_space=command(['df', '-h', '/']))
    return dict(
        schema='airllm.m5.memory-diagnostic.v1', read_only=True,
        before=before, after=snapshot(), raw_vm_stat=raw_vm,
        vm=vm_pages(raw_vm.get('stdout', '')),
        **raw_system,
        processes=rows, process_groups=groups, top_physical_footprint=footprint,
        process_inventory_diagnostics=inventory_diagnostics,
        process_inventory_privacy=dict(command_lines_included=include_command_lines,
            executable_paths_included=include_command_lines,
            note='Full process arguments are used transiently for classification; omitted from default reports.'),
        footprint_target_identity_unchanged=identities_after,
        inference_process_candidates=[row for row in rows if row['inference_candidate']],
        configuration_budgets=budgets,
        standard_model_admission=dict(admissible=not reasons, reasons=reasons,
                                      required_available_bytes=required,
                                      available_bytes=available,
                                      recovery_window_headroom_bytes=256 << 20),
        mlx_memory=dict(owned_inference_worker_found=False,
                       current_process_mlx_imported=False,
                       global_gpu_allocation_bytes=None, engine_cache_bytes=None,
                       reason='No inference worker ownership established. MLX counters are process-local; no GPU context created by this diagnostic.'),
        interpretation_limits=[
            'Snapshots and commands are sequential observations, not an atomic system partition.',
            'Summed RSS includes shared pages and omits compressed ownership; it is not exclusive physical consumption.',
            'macOS physical footprint is kernel accounting including compressed ownership, not resident RSS and not proof that closing a process releases that exact amount.',
            'File-backed pages overlap active/inactive classifications and are not all immediately reclaimable file cache.',
            'Compressor physical bytes and its logical uncompressed bytes describe the same content and must not be added.',
            'memory_pressure -Q free percentage is not psutil available memory nor proof that pressure sysctl is normal.',
            'Inference-name matches do not establish abandoned state or mission ownership; no signals sent.',
            'No system cache was purged, no personal application closed, no resource guard changed.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--footprint-count', type=int, default=5, choices=range(0, 11))
    parser.add_argument('--include-command-lines', action='store_true',
                        help='Include private process arguments and full executable paths; keep the report private')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation rejects an existing output, including a symlink.
    with args.output.open('x') as stream:
        report = collect(args.footprint_count, args.include_command_lines)
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()),
                          pressure_level=report['before']['pressure_level'],
                          available_bytes=report['before']['virtual_memory']['available'],
                          standard_model_admission=report['standard_model_admission'],
                          inference_candidates=len(report['inference_process_candidates']))))


if __name__ == '__main__':
    main()
