#!/usr/bin/env python3
"""Independently verify M5 campaign bytes, guards and full recorded logits.

Only Python's standard library and the retained standard-library speculative
recording verifier are used. A manifest proves consistency, not authenticity.
No engine, MLX, NumPy, process control or current checkpoint read is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
import struct
import sys

from verify_gguf_speculation import (verify as verify_logits, _recording,
                                    RejectedEvidence as LogitEvidenceRejection)


class RejectedEvidence(ValueError):
    pass


class VerifiedFidelityRejection(RejectedEvidence):
    """Valid completed recordings disagree; this never accepts their speed."""
    def __init__(self, result):
        self.result = result
        super().__init__('Verified fidelity rejection: '+', '.join(
            item['checker_reason'] for item in result['rejected_pairs']))


def require(condition, reason):
    if not condition:
        raise RejectedEvidence(reason)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(RejectedEvidence(x)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


RETAINED_LIMITS = dict(rss_bytes=2 << 30, reserve_bytes=3 << 30,
    swap_growth_bytes=128 << 20, admitted_payload_bytes=768 << 20,
    timeout_seconds=600., interval_seconds=.05, max_processes=32)
KERNEL_POLICY = dict(version=2, tensorops='strict_scalar_and_reference',
    reuse='strict_scalar_and_reference_or_full_grid_bitwise_reference_with_recorded_c_diagnosis',
    full_model_fidelity_required=True, scalar_tolerances_unchanged=True)
FIDELITY_FAILURE_POLICY = dict(version=1,
    continue_only_on=['verified_generated_tokens_mismatch','verified_ordered_expert_routes_mismatch',
                      'verified_full_logits_mismatch'],
    malformed_nonfinite_source_resource_or_input_errors='stop_campaign',
    rejected_candidates_cannot_supply_speedup_or_prerequisites=True)
VERIFIED_FIDELITY_MESSAGES = {
    'Turn transcripts or EOS differ: generated': 'verified_generated_tokens_mismatch',
    'Accepted-prefix expert routes differ': 'verified_ordered_expert_routes_mismatch',
    'Emitted full logits exceed numerical tolerance': 'verified_full_logits_mismatch'}


def validate_config(config):
    """Hashes never authorize weakening the mission's retained resource policy."""
    require(config.get('schema_version') == 1, 'Unsupported campaign schema')
    require(isinstance(config.get('origin'),str) and Path(config['origin']).is_absolute(), 'Missing campaign origin')
    limits = config.get('limits')
    require(isinstance(limits, dict) and limits == RETAINED_LIMITS,
            'Campaign safety policy differs from retained resource limits')
    require(all(type(limits[k]) is int for k in ('rss_bytes','reserve_bytes','swap_growth_bytes',
                'admitted_payload_bytes','max_processes')), 'Resource byte/process limits must be integers')
    require(type(config.get('pairs')) is int and config['pairs'] >= 2, 'At least two pairs are required')
    require(re.fullmatch('[0-9a-f]{64}', config.get('expected_checkpoint_sha256','')) is not None,
            'Missing expected full checkpoint digest')
    model = config.get('model_identity', {})
    require(isinstance(model.get('path'),str) and Path(model['path']).is_absolute(), 'Checkpoint path must be absolute')
    require(all(type(model.get(k)) is int and model[k] > 0 for k in ('bytes','inode','mtime_ns','ctime_ns')),
            'Invalid checkpoint identity')
    roots, manifests = config.get('engine_roots',{}), config.get('engine_manifests',{})
    require(set(roots) == {'h1','h2','candidate'} and set(manifests) == set(roots.values()), 'Invalid engine roots')
    for label, sha in (('h1','c8f81e57463ed3d7fd3d7dba226a65500439a9ce'),
                       ('h2','5e8234181e736c0319a64df25e6813e92dd3e6da')):
        require(manifests[roots[label]].get('git_head') == sha, 'Reference revision differs: '+label)
    require(config.get('numerical_tolerances') == dict(atol=1e-4,rtol=1e-4), 'Full-logit tolerances changed')
    require(config.get('kernel_qualification_policy') == KERNEL_POLICY, 'Kernel qualification policy is not the reviewed version')
    require(config.get('fidelity_failure_policy') == FIDELITY_FAILURE_POLICY,
            'Fidelity failure policy is not the reviewed version')
    protocol = config.get('protocol_sha256',{})
    require(isinstance(protocol,dict) and protocol and all(isinstance(k,str) and Path(k).is_absolute()
            and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in protocol.items()), 'Invalid protocol hashes')
    experiments = config.get('experiments')
    require(isinstance(experiments,list) and experiments, 'No frozen experiments')
    names = set()
    switches = {'--numerical-evidence','--request-trace','--hyper-prefill','--expert-batch','--concat-tiles',
                '--packed-copy-free','--packed-direct-io','--bulk-metadata','--zero-copy-weights',
                '--expert-pipeline','--grouped-direct-io'}
    values = {'--rows','--prefill-chunk','--hyper-prefill-max-tokens','--cache-policy','--io-mode',
              '--timeout','--string-buffer-bytes','--metal-batch-backend','--metal-token-tile','--speculative-depth',
              '--router-reference-chunk'}
    for experiment in experiments:
        name = experiment.get('name','')
        require(isinstance(name,str) and re.fullmatch('[A-Za-z0-9_-]+',name) and name not in names,
                'Unsafe or duplicate experiment name')
        names.add(name)
        plan = experiment.get('plan')
        require(isinstance(plan,list) and len(plan) == config['pairs']*2, 'Invalid planned worker count')
        ids = set()
        for job in plan:
            label = job.get('id','')
            require(isinstance(label,str) and re.fullmatch('(baseline|candidate)-p[0-9]+',label)
                    and label not in ids, 'Unsafe or duplicate worker ID')
            ids.add(label)
            require(job.get('engine_root') in manifests and job.get('tasks') in config.get('tasks_by_path',{})
                    and job['tasks'] in protocol, 'Unfrozen engine or task')
            args = job.get('args')
            require(isinstance(args,list) and all(isinstance(x,str) for x in args), 'Invalid worker arguments')
            seen, cursor = set(), 0
            while cursor < len(args):
                flag = args[cursor]
                require(flag not in seen and flag in switches|values, 'Unknown or repeated worker control')
                seen.add(flag)
                cursor += 1
                if flag in values:
                    require(cursor < len(args) and not args[cursor].startswith('--'), 'Missing worker control value')
                    value = args[cursor]
                    cursor += 1
                    if flag == '--timeout':
                        require(value == '600' or value == '600.0', 'Worker timeout differs from retained limit')
                    elif flag in ('--prefill-chunk','--hyper-prefill-max-tokens'):
                        require(value in ('16','32','64','128'), 'Prefill control outside admitted campaign sizes')
                    elif flag == '--metal-batch-backend':
                        require(value in ('off','reuse','tensorops'), 'Unknown matrix backend')
                    elif flag in ('--metal-token-tile','--speculative-depth'):
                        require(value in ('0','2','4','8','16') and (flag != '--metal-token-tile' or value != '0'),
                                'Unknown speculation/tile depth')
                    elif flag == '--router-reference-chunk':
                        require(value in ('0','16'),'Unknown router reference chunk')
                    elif flag == '--cache-policy':
                        require(value in ('lru','adaptive'), 'Unknown cache policy')
                    elif flag == '--io-mode':
                        require(value in ('pread','pread_nocache'), 'Unknown I/O policy')
                    elif flag == '--string-buffer-bytes':
                        require(value.isdigit() and 0 <= int(value) <= 65536, 'Unbounded string buffer')
                    elif flag == '--rows':
                        require(value.isdigit() and 1 <= int(value) <= 8192, 'Unbounded row tile')
            require('--numerical-evidence' in seen, 'Full recorded logits are mandatory')
            require(not {'--packed-copy-free','--packed-direct-io'} <= seen, 'Conflicting packed transport')
            require('--grouped-direct-io' not in seen or {'--expert-batch','--packed-direct-io'} <= seen,
                    'Grouped transport prerequisites missing')
    for experiment in experiments:
        require(all(p in names and p != experiment['name'] for p in experiment.get('requires_prior_pass',[])),
                'Invalid combination prerequisites')
    return config


def inside(folder, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Invalid evidence path')
    path = (folder/relative).resolve(strict=True)
    require(path.is_relative_to(folder), 'Evidence path escapes campaign')
    require(not (folder/relative).is_symlink(), 'Evidence symlink is not accepted')
    return path


def evidence_path(folder, path, config):
    """Relocate recorded paths through the immutable original campaign root."""
    recorded = Path(path)
    if recorded.is_absolute():
        try:
            relative = str(recorded.relative_to(config['origin']))
        except ValueError as error:
            raise RejectedEvidence('Recorded report escapes original campaign') from error
    else:
        relative = str(recorded)
    return inside(Path(folder).resolve(), relative)


FINDER_METADATA_NAME='.DS_Store'
FINDER_METADATA_MAGIC=b'\x00\x00\x00\x01Bud1'


def require_no_finder_evidence(files):
    """Never discard a path that an existing index declares as evidence."""
    require(isinstance(files,dict),'Missing evidence file hashes')
    require(not any(Path(name).name==FINDER_METADATA_NAME for name in files),
            'A manifest cannot relabel declared evidence as ignored Finder metadata')


def finder_metadata_ignored(path):
    """Only ordinary, unaliased Finder databases are non-evidence metadata.

    The exact name and observed Apple database header are mandatory. Existing
    manifest entries never use this rule to waive an immutable evidence hash.
    Read only the 8-byte signature; a symlink/device/FIFO must never be followed.
    """
    path=Path(path)
    if path.name!=FINDER_METADATA_NAME:
        return False
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1,
            'Finder metadata must be an unaliased regular file')
    require(hasattr(os,'O_NOFOLLOW') and hasattr(os,'O_NONBLOCK'),
            'Safe nonfollowing Finder metadata reads are unavailable')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        opened=os.fstat(descriptor)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink==1
                and (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),
                'Finder metadata identity changed before classification')
        require(os.read(descriptor,len(FINDER_METADATA_MAGIC))==FINDER_METADATA_MAGIC,
                'A non-Finder file cannot be excluded as metadata')
        after=path.lstat()
        require(stat.S_ISREG(after.st_mode) and after.st_nlink==1
                and (after.st_dev,after.st_ino)==(opened.st_dev,opened.st_ino),
                'Finder metadata identity changed during classification')
    finally:
        os.close(descriptor)
    return True


def verify_manifest(folder, *, allow_interrupted=False):
    folder = Path(folder).resolve(strict=True)
    manifest = read_json(folder/'manifest.json')
    files = manifest.get('files')
    require(isinstance(files, dict) and files, 'Missing evidence file hashes')
    require_no_finder_evidence(files)
    paths=[path for path in folder.rglob('*') if not finder_metadata_ignored(path)]
    actual = {str(p.relative_to(folder)) for p in paths if p.is_file()
              and p.name not in ('manifest.json', '.runner.lock')}
    require(set(files) <= actual, 'Evidence manifest has missing files')
    if not allow_interrupted:
        require(actual == set(files), 'Evidence manifest has unlisted files')
    for relative, expected in files.items():
        path = inside(folder, relative)
        # Atomic state may be newer than the last finalized manifest after an
        # abrupt controller death. Immutable prior evidence is still checked;
        # every completed job is revalidated before a resume can skip it.
        if allow_interrupted and relative == 'state.json':
            continue
        require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'],
                'Evidence digest or length differs: ' + relative)
    return len(files)


def guard_accepted(guard, limits, command):
    require(guard.get('command') == command, 'Supervisor command differs from frozen job')
    require(guard.get('limits') == limits, 'Supervisor limits differ from retained limits')
    require(type(guard.get('exit_code')) is int and guard['exit_code'] == 0,
            'Worker did not exit successfully')
    require('violation' in guard and guard['violation'] is None and not guard.get('exception')
            and not guard.get('cleanup_error'), 'Resource guard or cleanup failed')
    require(type(guard.get('sample_count')) is int and guard['sample_count'] > 0, 'No resource observations')
    for key in ('sampled_peak_rss_bytes','minimum_available_bytes','maximum_swap_growth_bytes','elapsed_seconds'):
        value = guard.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'Invalid resource metric: '+key)
    require(type(guard.get('sampled_peak_processes')) is int and guard['sampled_peak_processes'] >= 1,
            'Invalid observed process count')
    require(guard.get('sampled_peak_rss_bytes', math.inf) <= limits['rss_bytes'], 'RSS ceiling exceeded')
    require(guard.get('minimum_available_bytes', 0) >= limits['reserve_bytes'], 'System reserve violated')
    require(guard.get('maximum_swap_growth_bytes', math.inf) <= limits['swap_growth_bytes'], 'Swap growth exceeded')
    require(guard.get('sampled_peak_processes', math.inf) <= limits['max_processes'], 'Process ceiling exceeded')
    require(guard.get('peak_pressure_level') == 1, 'Normal macOS pressure not established')
    require(0 <= guard.get('elapsed_seconds', math.inf) <= limits['timeout_seconds'] + 10,
            'Guard elapsed time invalid')
    for label in ('before', 'after'):
        snap = guard.get(label, {})
        require(all(type(snap.get(k)) is int and snap[k] >= 0 for k in ('available_bytes','swap_used_bytes')),
                'Invalid resource snapshot: '+label)
        require(snap.get('pressure_level') == 1 and snap.get('available_bytes', 0) >= limits['reserve_bytes'],
                'Missing normal resource snapshot: ' + label)
    require(guard['before']['available_bytes'] >= limits['reserve_bytes']+limits['admitted_payload_bytes'],
            'Initial payload could not be admitted')
    # The retained sampler records its final snapshot after its loop/minimum.
    # The independent final reserve check above still applies to that snapshot.
    require(guard['minimum_available_bytes'] <= guard['before']['available_bytes'],
            'Minimum available memory contradicts initial snapshot')
    require(guard['after']['swap_used_bytes']-guard['before']['swap_used_bytes'] <= guard['maximum_swap_growth_bytes'],
            'Declared swap growth contradicts resource snapshots')


def option(command, name):
    require(command.count(name) == 1, 'Missing or repeated command argument: ' + name)
    index = command.index(name)
    require(index+1 < len(command), 'Missing command value: ' + name)
    return command[index+1]


def validate_kernel_report(path, config, folder=None):
    recorded_path = Path(path)
    path = evidence_path(folder, path, config) if folder is not None else recorded_path
    report = read_json(path)
    guard = read_json(path.with_suffix('.guard.json'))
    require(report.get('schema') == 'airllm-m5-kernels-v1' and report.get('status') == 'completed',
            'Kernel validation did not complete')
    require(report.get('strict_fp32') is True and report.get('sources_unchanged') is True,
            'Kernel report is not strict FP32 with unchanged sources')
    require(report.get('sources_before') == report.get('sources_after'), 'Kernel source identities changed')
    require(report.get('versions',{}).get('mlx') == '0.32.4', 'Kernel report did not test required MLX 0.32.4')
    source_names = ('benchmarks/validate_m5_kernels.py', 'air_llm/airllm/nextgen/gguf_m5.py',
                    'air_llm/airllm/nextgen/gguf_metal.py', 'air_llm/airllm/nextgen/experiment_guard.py')
    declared_output = option(guard.get('command', []), '--output')
    expected_output = config.get('kernel_report_origin') if path.parent.name == 'provided-kernels' else str(recorded_path)
    require(declared_output == expected_output, 'Kernel report output differs')
    for name in source_names:
        expected = config['protocol_sha256'].get(str(Path(config['engine_roots']['candidate'])/name))
        value = report.get('sources_before', {}).get(name)
        require(value == expected and value is not None, 'Kernel source does not match frozen candidate: '+name)
    limits = dict(config['limits'], rss_bytes=512 << 20, admitted_payload_bytes=256 << 20)
    require(report.get('limits') == limits, 'Kernel report changed admitted memory limits')
    command = guard.get('command', [])
    require(len(command) in (7,9) and command[1] == str(Path(config['engine_roots']['candidate'])/'benchmarks/validate_m5_kernels.py')
            and command[2:4] == ['--output',expected_output] and command[4] == '--repetitions'
            and command[5].isdigit() and int(command[5]) >= 2 and command[6] == '--worker'
            and (len(command) == 7 or command[7] == '--capture'), 'Wrong full-grid kernel supervisor command')
    guard_accepted(guard, limits, command)
    require(isinstance(report.get('cases'), list) and report['cases'], 'No actual kernel cases')
    validate_kernel_grid(report)
    report['_verified_raw_reference_bitwise'] = validate_kernel_artifacts(report,path.parent)
    report['_recorded_c_diagnosis_verified'] = False
    diagnostic_path = config.get('kernel_diagnosis')
    if diagnostic_path is not None:
        diagnostic_path = (evidence_path(folder,diagnostic_path,config) if folder is not None else Path(diagnostic_path))
        validate_kernel_diagnosis(diagnostic_path,path,report,config)
        report['_recorded_c_diagnosis_verified'] = True
    report['_reviewed_qualification_policy'] = config.get('kernel_qualification_policy')
    return report


def validate_kernel_grid(report):
    shapes = [(19,256,b) for b in (2,16,32,64,128)]+[(64,512,16),(64,512,64),(128,1024,128)]
    expected = {(kind,grouped,rows,width,batch,mode,tile) for kind in (12,14)
                for grouped in (False,True) for rows,width,batch in shapes
                for mode,tile in [('reuse',2),('reuse',4),('reuse',8),('reuse',16),('tensorops',4)]}
    actual = []
    for case in report['cases']:
        key = tuple(case.get(k) for k in ('kind','grouped','rows','width','batch','backend','token_tile'))
        require(key in expected, 'Kernel case is outside required full grid')
        actual.append(key)
        for name in ('reference_vs_scalar','candidate_vs_scalar','candidate_vs_reference'):
            comparison = case.get(name)
            if name != 'reference_vs_scalar' and not case.get('compiled_and_executed'):
                continue
            require(isinstance(comparison,dict) and type(comparison.get('accepted')) is bool
                    and comparison.get('atol') == 1e-4 and comparison.get('rtol') == 5e-5,
                    'Kernel numerical comparison changed precision/tolerances')
            error = comparison.get('max_absolute_error')
            require(type(error) in (int,float) and math.isfinite(error) and error >= 0,
                    'Kernel numerical comparison is missing a finite error')
    require(len(actual) == len(set(actual)) == len(expected) and set(actual) == expected,
            'Kernel report is quick, incomplete or duplicates required full-grid cases')


def kernel_case_key(case):
    return json.dumps([case[k] for k in ('kind','grouped','rows','width','batch','backend','token_tile')],separators=(',',':'))


def validate_kernel_artifacts(report,base):
    """Recompute recorded FP32 comparisons and bitwise reference parity."""
    base = Path(base).resolve()
    checked = {}
    exact = {}
    def tensor(descriptor,shape,dtype):
        require(isinstance(descriptor,dict) and descriptor.get('shape') == shape and descriptor.get('dtype') == dtype,
                'Raw kernel tensor shape or dtype differs')
        size = math.prod(shape)*(1 if dtype == '|u1' else 4)
        require(0 < size <= 16 << 20 and descriptor.get('bytes') == size,'Raw kernel tensor length is invalid')
        path = inside(base,descriptor.get('file'))
        require(path.stat().st_size == size,'Raw kernel tensor length differs')
        raw = path.read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        require(value == descriptor.get('sha256') and (path not in checked or checked[path] == value),
                'Raw kernel tensor digest differs')
        checked[path] = value
        return raw
    def values(raw):
        result = tuple(x[0] for x in struct.iter_unpack('<f',raw))
        require(all(math.isfinite(x) for x in result),'Raw kernel tensor is nonfinite')
        return result
    def compare(actual,expected,recorded):
        accepted = all(abs(a-b) <= 1e-4+5e-5*abs(b) for a,b in zip(actual,expected))
        maximum = max(abs(a-b) for a,b in zip(actual,expected))
        require(recorded['accepted'] is accepted,'Recorded kernel numerical decision contradicts raw tensors')
        # NumPy records a rounded FP32 subtraction; Python subtracts in FP64.
        # This comparison tolerates metadata rounding, never a looser allclose.
        require(math.isclose(recorded['max_absolute_error'],maximum,rel_tol=2e-7,abs_tol=1e-12),
                'Recorded kernel maximum error contradicts raw tensors')
    for case in report['cases']:
        rows,width,batch = (case[k] for k in ('rows','width','batch'))
        prefix = [2] if case['grouped'] else []
        shape = prefix+[batch,rows]
        evidence = case.get('numeric_evidence',{})
        tensor(evidence.get('packed'),prefix+[rows,width//256*(144 if case['kind'] == 12 else 210)],'|u1')
        values(tensor(evidence.get('input'),prefix+[batch,width],'<f4'))
        scalar_raw = tensor(evidence.get('scalar'),shape,'<f4')
        reference_raw = tensor(evidence.get('reference'),shape,'<f4')
        scalar,reference = values(scalar_raw),values(reference_raw)
        compare(reference,scalar,case['reference_vs_scalar'])
        bitwise = False
        if case.get('compiled_and_executed') is True:
            candidate_raw = tensor(case.get('candidate_evidence'),shape,'<f4')
            candidate = values(candidate_raw)
            compare(candidate,scalar,case['candidate_vs_scalar'])
            compare(candidate,reference,case['candidate_vs_reference'])
            bitwise = candidate_raw == reference_raw
        exact[kernel_case_key(case)] = bitwise
    for path,expected in checked.items():
        require(digest(path) == expected,'Raw kernel tensor changed during verification')
    return exact


def validate_kernel_diagnosis(path,kernel_path,report,config):
    """Validate provenance/coverage of recorded C-codec evidence; do not execute C."""
    diagnosis = read_json(path)
    require(digest(path) == config.get('kernel_diagnosis_sha256'),'Kernel diagnosis digest differs')
    require(diagnosis.get('schema') == 'airllm-m5-numerical-diagnosis-v1'
            and diagnosis.get('report_sha256') == digest(kernel_path)
            and diagnosis.get('compiler_exit_code') == 0,'C-codec diagnosis does not match kernel evidence')
    candidate = Path(config['engine_roots']['candidate'])
    for field,relative in (('c_decoder_sha256','tests_next/reference/ggml_codec_reference.c'),
                           ('diagnostic_source_sha256','benchmarks/diagnose_m5_numerics.py'),
                           ('scalar_oracle_source_sha256','benchmarks/validate_m5_kernels.py')):
        expected = config['protocol_sha256'].get(str(candidate/relative))
        require(expected is not None and diagnosis.get(field) == expected,'C-codec diagnosis source differs: '+field)
    expected_shapes = {tuple(c[k] for k in ('kind','grouped','rows','width','batch')) for c in report['cases']}
    shapes = diagnosis.get('shapes')
    require(isinstance(shapes,list) and len(shapes) == len(expected_shapes),'C-codec diagnosis is incomplete')
    actual = [tuple(s.get(k) for k in ('kind','grouped','rows','width','batch')) for s in shapes]
    require(len(set(actual)) == len(actual) and set(actual) == expected_shapes,'C-codec diagnosis shape coverage differs')
    require(all(s.get('independent_c_vs_scalar_weights_exact') is True
                and s.get('independent_c_vs_vectorized_weights_exact') is True for s in shapes)
            and diagnosis.get('all_scalar_weights_match_independent_c') is True
            and diagnosis.get('all_vectorized_weights_match_independent_c') is True,
            'Recorded C-codec comparison did not preserve all decoded weights')


def kernel_eligibility(report, experiment):
    """Require a tested numerically valid shape and observed host benefit.

    Unselected shapes still use runtime reference fallback. Miniatures never
    establish real-model parity, which is independently checked after each pair.
    """
    validate_kernel_grid(report)
    for job in experiment['plan']:
        args = job['args']
        if '--metal-batch-backend' not in args:
            continue
        mode = args[args.index('--metal-batch-backend')+1]
        tile = int(args[args.index('--metal-token-tile')+1]) if '--metal-token-tile' in args else 4
        cases = [c for c in report['cases'] if c.get('backend') == mode
                 and (mode != 'reuse' or c.get('token_tile') == tile)]
        strict = [c for c in cases if c.get('compiled_and_executed') is True
                 and c.get('reference_vs_scalar', {}).get('accepted') is True
                 and c.get('candidate_vs_scalar', {}).get('accepted') is True
                 and c.get('candidate_vs_reference', {}).get('accepted') is True]
        alternative = (mode == 'reuse' and report.get('_reviewed_qualification_policy') == KERNEL_POLICY
            and report.get('_recorded_c_diagnosis_verified') is True and all(
                c.get('compiled_and_executed') is True and report.get('_verified_raw_reference_bitwise',{}).get(kernel_case_key(c)) is True
                for c in cases))
        valid = cases if alternative else strict
        qualifier = ('reference_exact_with_inherited_scalar_disagreement' if alternative and len(strict) != len(cases)
                     else 'strict_scalar_and_reference')
        report.setdefault('_qualification_modes',{})[mode+'/'+str(tile)] = qualifier
        if not valid:
            return 'No compiled numerically accepted kernel case for '+mode+'/'+str(tile)
        # Once admitted, every tested kind/grouped/shape for this backend must
        # retain its numerical acceptance. Runtime still calibrates real shapes.
        if len(valid) != len(cases):
            return 'Some required full-grid cases failed compilation or numerics for '+mode+'/'+str(tile)
        timed = [c for c in valid if 'host_seconds' in c]
        if alternative:
            require(all(c.get('selected') is False for c in valid if 'host_seconds' not in c),
                    'Untimed inherited scalar disagreement cannot claim speed benefit')
        else:
            require(len(timed) == len(valid),'Missing paired kernel host timings')
        for case in timed:
            timings = case.get('host_seconds')
            require(isinstance(timings,list) and len(timings) == 2 and all(isinstance(x,list) for x in timings)
                    and len(timings[0]) == len(timings[1]) >= 2, 'Missing paired kernel host timings')
            require(all(type(x) in (int,float) and math.isfinite(x) and x > 0 for side in timings for x in side),
                    'Invalid kernel host timings')
            measured = statistics.median(timings[0])/statistics.median(timings[1])
            require(type(case.get('host_median_speedup')) in (int,float)
                    and math.isclose(case['host_median_speedup'], measured, rel_tol=1e-12)
                    and case.get('selected') is (measured > 1.03), 'Kernel benefit contradicts raw timings')
            require(case.get('timing_order') == [[0,1] if n%2 == 0 else [1,0] for n in range(len(timings[0]))],
                    'Kernel timing order was not counterbalanced')
        if not any(c.get('selected') is True for c in timed):
            return 'No observed host latency benefit for '+mode+'/'+str(tile)+'; reference retained'
    return None


def validate_job(folder, config, job, record):
    folder = Path(folder).resolve()
    output = inside(folder, record['output'])
    report = read_json(output)
    guard = read_json(output.with_suffix('.guard.json'))
    command = record.get('command')
    require(isinstance(command, list) and len(command) > 4, 'No frozen worker command')
    require(Path(command[1]).name == 'benchmark_research51.py', 'Wrong worker entrypoint')
    require(option(command, '--model') == config['model_identity']['path'], 'Wrong model path')
    require(option(command, '--tasks') == job['tasks'] and option(command, '--engine-root') == job['engine_root']
            and option(command, '--output') == record.get('output_absolute', str(output)) and option(command, '--label') == job['id'],
            'Worker identity arguments differ')
    source = config['engine_manifests'][job['engine_root']]
    require(option(command, '--expected-engine-head') == source['git_head'], 'Wrong expected engine commit')
    require(json.loads(option(command, '--expected-engine-sources-json')) == source['source_sha256'],
            'Wrong expected source hashes')
    require(json.loads(option(command, '--expected-protocol-json')) == config['protocol_sha256'],
            'Wrong expected protocol hashes')
    # Comparison flags are an exact suffix, with no undocumented trailing switch.
    marker = command.index('--expected-engine-sources-json')+2
    require(command[marker:] == job['args'], 'Comparison flags differ from the frozen experiment')
    guard_accepted(guard, config['limits'], [*command, '--worker'])
    require(record.get('exit_code') == 0, 'Controller did not observe successful parent exit')
    duration = record.get('controller_wall_seconds')
    require(type(duration) in (int,float) and math.isfinite(duration) and duration > 0,
            'Invalid controller process duration')
    require(report.get('status') == 'completed', 'Incomplete worker')
    for key in ('source_unchanged', 'protocol_unchanged', 'model_stat_unchanged', 'engine_head_unchanged'):
        require(report.get(key) is True, 'Worker input changed: '+key)
    require(report.get('git_head') == source['git_head'] == report.get('git_head_after'), 'Engine revision differs')
    require(report.get('source_sha256') == source['source_sha256'], 'Engine source differs')
    require(report.get('protocol_sha256') == config['protocol_sha256'], 'Protocol differs')
    for key in ('worker_wall_seconds','initialization_seconds','tokenizer_seconds','model_initialization_seconds'):
        value = report.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0,
                'Invalid worker stage duration: '+key)
    require(report['worker_wall_seconds'] > 0, 'Missing complete worker duration')
    model = config['model_identity']
    require(report.get('model') == {k: model[k] for k in ('path', 'bytes', 'inode', 'mtime_ns')},
            'Worker checkpoint identity differs')
    tasks = config['tasks_by_path'][job['tasks']]
    expected_task_hash = config['protocol_sha256'].get(job['tasks'])
    require(expected_task_hash == report.get('tasks_sha256'), 'Task digest differs')
    require(report.get('tasks_snapshot') == tasks, 'Task snapshot differs')
    turns = report.get('turns')
    require(isinstance(turns, list) and len(turns) == len(tasks) and turns, 'Missing task outputs')
    for task, turn in zip(tasks, turns):
        require(all(turn.get(k) == task[k] for k in ('id', 'prompt', 'max_new_tokens')), 'Output task differs')
        tokens = turn.get('generated_ids', [])
        require(turn.get('status') == 'completed' and isinstance(tokens, list) and tokens,
                'No completed token sequence')
        require(len(tokens) <= task['max_new_tokens'], 'Output exceeds maximum token count')
        stops = turn.get('eos_token_ids', [])
        stop = 'eos' if tokens[-1] in stops else 'max_new_tokens'
        require(turn.get('stop_reason') == stop and (stop == 'eos' or len(tokens) == task['max_new_tokens']),
                'Stop conditions differ from generated IDs')
        for name in ('total_seconds', 'ttft_seconds', 'wall_seconds', 'logical_read_bytes', 'read_calls'):
            value = turn.get(name)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Invalid measured metric: '+name)
    return report


def summarize_experiment(folder, config, experiment, jobs):
    folder = Path(folder).resolve()
    by_pair = {}
    immutable_recordings = {}
    for ordinal, job in enumerate(experiment['plan']):
        variant, pair_text = job['id'].split('-p')
        pair = int(pair_text)
        expected = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
        require(variant == expected[ordinal % 2], 'AB/BA order was not counterbalanced')
        key = experiment['name']+'/'+job['id']
        require(key in jobs, 'Incomplete paired experiment')
        record = jobs[key]
        report = validate_job(folder, config, job, record)
        # Validate every raw recording before any mismatch may be classified as
        # a negative fidelity result. A first-pair route mismatch must never hide
        # a later worker's corrupt/nonfinite/incoherent evidence.
        recording = _recording(inside(folder,record['output']),serial=variant == 'baseline')
        immutable_recordings[recording['path']] = recording['sha256']
        for path,expected_digest in recording['artifacts'].items():
            require(path not in immutable_recordings or immutable_recordings[path] == expected_digest,
                    'Recording artifacts changed during experiment prevalidation')
            immutable_recordings[path] = expected_digest
        by_pair.setdefault(pair, {})[variant] = (record, report)
    require(len(by_pair) >= config['pairs'] >= 2, 'Insufficient paired repetitions')
    results = []
    rejected = []
    for pair in sorted(by_pair):
        group = by_pair[pair]
        require(set(group) == {'baseline', 'candidate'}, 'Incomplete pair')
        (a, ar), (b, br) = group['baseline'], group['candidate']
        try:
            parity = verify_logits(inside(folder, a['output']), inside(folder, b['output']),
                                   **config['numerical_tolerances'])
        except LogitEvidenceRejection as error:
            reason = str(error)
            if reason not in VERIFIED_FIDELITY_MESSAGES:
                raise
            rejected.append(dict(pair=pair,category=VERIFIED_FIDELITY_MESSAGES[reason],
                checker_reason=reason,baseline_report_sha256=immutable_recordings[inside(folder,a['output'])],
                candidate_report_sha256=immutable_recordings[inside(folder,b['output'])]))
            continue
        baseline = sum(x['wall_seconds'] for x in ar['turns'])
        candidate = sum(x['wall_seconds'] for x in br['turns'])
        require(baseline > 0 and candidate > 0, 'Zero request latency')
        results.append(dict(pair=pair, full_recorded_logits=parity,
            baseline_request_seconds=baseline, candidate_request_seconds=candidate,
            total_speedup=baseline/candidate,
            baseline_worker_seconds=ar['worker_wall_seconds'],
            candidate_worker_seconds=br['worker_wall_seconds'],
            worker_total_speedup=ar['worker_wall_seconds']/br['worker_wall_seconds'],
            baseline_process_wall_seconds=a.get('controller_wall_seconds'),
            candidate_process_wall_seconds=b.get('controller_wall_seconds'),
            process_total_speedup=a['controller_wall_seconds']/b['controller_wall_seconds'],
            baseline_initialization_seconds=ar['initialization_seconds'],
            candidate_initialization_seconds=br['initialization_seconds'],
            baseline_ttft_seconds=[t['ttft_seconds'] for t in ar['turns']],
            candidate_ttft_seconds=[t['ttft_seconds'] for t in br['turns']],
            baseline_logical_read_bytes=sum(t['logical_read_bytes'] for t in ar['turns']),
            candidate_logical_read_bytes=sum(t['logical_read_bytes'] for t in br['turns']),
            baseline_read_calls=sum(t['read_calls'] for t in ar['turns']),
            candidate_read_calls=sum(t['read_calls'] for t in br['turns']),
            baseline_decode_tokens_per_second=[t.get('decode_tokens_per_second') for t in ar['turns']],
            candidate_decode_tokens_per_second=[t.get('decode_tokens_per_second') for t in br['turns']],
            baseline_worker_cpu_seconds=ar.get('cpu_seconds'),
            candidate_worker_cpu_seconds=br.get('cpu_seconds'),
            candidate_logical_bytes_per_emitted_token=sum(t['logical_read_bytes'] for t in br['turns'])/
                sum(len(t['generated_ids']) for t in br['turns']),
            candidate_first_forward_seconds=[t['step_checks'][0]['forward_seconds'] for t in br['turns']],
            prefill_scope='first forward contains prefill; TTFT and full request are separately retained',
            emitted_tokens=[len(t['generated_ids']) for t in br['turns']],
            sustained_decode_evidence=all(len(t['generated_ids']) >= 64 for t in br['turns'])))
    # The retained checker stops before its final digest recheck on a fidelity
    # mismatch. Close that gap explicitly, including pairs after the first one.
    for path,expected_digest in immutable_recordings.items():
        require(digest(path) == expected_digest,'Recording evidence changed during fidelity comparison')
    if rejected:
        raise VerifiedFidelityRejection(dict(status='rejected_fidelity',
            policy_version=FIDELITY_FAILURE_POLICY['version'],tasks_kind=experiment['tasks_kind'],
            rejected_pairs=rejected,accepted_pair_numbers=[row['pair'] for row in results],
            all_worker_controls_and_recordings_validated=True,
            performance_comparison_accepted=False,eligible_as_combination_prerequisite=False,
            numerical_tolerances=config['numerical_tolerances'],
            scope='Recorded strict fidelity mismatch; cause is not inferred, routes retain their original order'))
    return dict(status='accepted', tasks_kind=experiment['tasks_kind'], pairs=results,
                median_total_speedup=statistics.median(x['total_speedup'] for x in results),
                median_worker_total_speedup=statistics.median(x['worker_total_speedup'] for x in results),
                median_process_total_speedup=statistics.median(x['process_total_speedup'] for x in results),
                timing_scope='instrumented request wall latency; includes draft, calibration, verification and replay',
                pure_gpu_seconds=None, physical_ssd_certified=False,
                full_recurrent_state_parity_validated=False)


def verify_hash(folder, path, config):
    output = evidence_path(folder, path, config)
    record = read_json(output)
    guard = read_json(output.with_suffix('.guard.json'))
    require(record.get('sha256') == config['expected_checkpoint_sha256']
            and record.get('stable_file') is True
            and record.get('bytes') == config['model_identity']['bytes'], 'Checkpoint hash differs')
    limits = dict(config['limits'], admitted_payload_bytes=64 << 20, timeout_seconds=120.)
    command = guard.get('command', [])
    require(len(command) == 7 and Path(command[1]).name == 'hash_checkpoint_guarded.py'
            and command[2:] == ['--model', config['model_identity']['path'], '--output', str(path), '--worker'],
            'Checkpoint hash command differs')
    guard_accepted(guard, limits, command)
    return record['sha256']


def verify(folder):
    folder = Path(folder).resolve(strict=True)
    count = verify_manifest(folder)
    config = read_json(folder/'config.json')
    state = read_json(folder/'state.json')
    require(state.get('config_sha256') == digest(folder/'config.json'), 'Frozen configuration changed')
    require(state.get('status') in ('completed', 'completed_with_exclusions','completed_evidence_with_rejections'),
            'Campaign is incomplete or deferred')
    validate_config(config)
    completed_sessions = [s for s in state['sessions'] if s.get('status') == 'completed']
    require(completed_sessions, 'No complete integrity session')
    for session in completed_sessions:
        verify_hash(folder, session['hash_before'], config)
        verify_hash(folder, session['hash_after'], config)
    sealed = {}
    for session in state['sessions']:
        for block in session.get('integrity_blocks', []):
            if block.get('status') != 'sealed':
                continue
            key = (session['index'],block['index'])
            require(key not in sealed, 'Duplicate integrity block identity')
            verify_hash(folder, block['hash_before'], config)
            verify_hash(folder, block['hash_after'], config)
            sealed[key] = (session,block)
    results = {}
    skipped = {}
    rejections = {}
    for experiment in config['experiments']:
        name = experiment['name']
        recorded = state['experiment_results'].get(name, {})
        session = next((s for s in state['sessions'] if s['index'] == recorded.get('session')), None)
        kernel_path = config.get('kernel_report') or (session or {}).get('kernel_report')
        kernel_report = None
        if kernel_path:
            kernel_report = validate_kernel_report(kernel_path, config, folder)
        if recorded.get('status') == 'excluded':
            reason = kernel_eligibility(kernel_report, experiment) if kernel_report else None
            unmet = [p for p in experiment.get('requires_prior_pass', []) if p not in results
                     or results[p]['median_total_speedup'] <= 1 or results[p]['median_worker_total_speedup'] <= 1]
            require(reason is not None or unmet, 'Excluded experiment has no verified reason')
            skipped[name] = dict(status='excluded', reason=reason, unmet_prerequisites=unmet)
            continue
        qualification_modes = {}
        for job in experiment['plan']:
            record = state['jobs'].get(name+'/'+job['id'], {})
            pair = sealed.get((record.get('session'),record.get('integrity_block')))
            require(pair is not None, 'Worker lacks a sealed before/after checkpoint hash bracket')
            bracket_session, bracket = pair
            require(name+'/'+job['id'] in bracket.get('jobs',[]), 'Worker is not covered by its integrity block')
            before, after = bracket.get('hash_before_completed_unix'), bracket.get('hash_after_started_unix')
            started, ended = record.get('started_unix'), record.get('ended_unix')
            require(all(type(x) in (int,float) and math.isfinite(x) for x in (before,after,started,ended))
                    and before <= started <= ended <= after, 'Worker is outside its recorded integrity hash bracket')
            if any('--metal-batch-backend' in j['args'] for j in experiment['plan']):
                path = config.get('kernel_report') or bracket_session.get('kernel_report')
                require(path is not None, 'Kernel experiment lacks admitted prerequisite')
                report = validate_kernel_report(path, config, folder)
                require(kernel_eligibility(report, experiment) is None, 'Kernel experiment lacked measured numerical prerequisite')
                qualification_modes.update(report.get('_qualification_modes',{}))
        for prereq in experiment.get('requires_prior_pass', []):
            require(prereq in results and results[prereq]['median_total_speedup'] > 1
                    and results[prereq]['median_worker_total_speedup'] > 1,
                    'Combination prerequisite lacks verified real measured benefit')
        try:
            result = summarize_experiment(folder, config, experiment, state['jobs'])
        except VerifiedFidelityRejection as error:
            # Recalculate the recorded failure from intact raw proofs; a stored
            # status/message never authorizes skipping fidelity or guard checks.
            require(recorded.get('status') == 'rejected_fidelity',
                    'Recorded experiment conceals an independently verified fidelity rejection')
            require(recorded == error.result,'Recorded fidelity rejection differs from independent recalculation')
            rejections[name] = error.result
            continue
        require(recorded.get('status') == 'accepted','Recorded rejection/exclusion has no independently reproduced failure')
        if qualification_modes:
            result['kernel_qualification'] = dict(policy_version=2,modes=qualification_modes,
                scalar_negatives_preserved=True,c_codec_evidence='recorded and source checked; not reexecuted by this verifier',
                miniature_parity_is_full_model_parity=False)
        results[name] = result
    require(set(state['experiment_results']) == set(results)|set(skipped)|set(rejections),
            'Missing recorded experiment outcomes')
    require((state['status'] == 'completed_evidence_with_rejections') == bool(rejections),
            'Campaign terminal status misrepresents fidelity rejections')
    return dict(status='completed_evidence_with_rejections' if rejections else 'accepted_evidence',
                verified_file_count=count, experiments=results, exclusions=skipped,rejected_fidelity=rejections,
                checkpoint_sha256=config['expected_checkpoint_sha256'],
                authenticity_certified=False, physical_ssd_certified=False,
                full_recurrent_state_parity_validated=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    text = json.dumps(verify(a.campaign), indent=2, allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x') as f:
            f.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
