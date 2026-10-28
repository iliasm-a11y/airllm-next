#!/usr/bin/env python3
"""Opt-in targeted router captures from unmodified historical Qwen checkouts.

Captures stop after the selected router call, before its experts run. They are
numerical diagnostics, never complete generations or performance measurements.
The Python profile hook reads the original router's locals at return; it does
not replace its arithmetic, sort its outputs or change the fidelity verifier.
The diagnostic adds host copies/synchronization and must not be timed as a bench.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DISK_RESERVE = 5 << 30
MAX_CAPTURE_BYTES = 16 << 20
ARRAY_NAMES = ('x', 'routing_input', 'weight', 'logits', 'probabilities',
               'order', 'selected', 'scores', 'next_score', 'tied',
               'reference_scores', 'reference_logits', 'reference_probabilities')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def checkpoint_digest(path):
    """Hash all checkpoint bytes with Darwin fd caching disabled; fail closed.

    The separately opened read-only descriptor keeps the same 1 MiB hash buffer.
    F_NOCACHE is a per-descriptor policy, never a cache purge or cold-NAND proof.
    Other evidence hashes intentionally keep the ordinary digest() path.
    """
    from airllm.nextgen.physical_io import set_fd_nocache
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        set_fd_nocache(stream.fileno())
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _call_with_locals(function, args, kwargs, *, locals_function=None):
    """Observe one Python return without altering the original call or result."""
    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError('Target router capture requires an unused Python profile hook')
    captured = {}
    code = (function if locals_function is None else locals_function).__code__

    def profile(frame, event, argument):
        if frame.f_code is code and event == 'return':
            captured.update(frame.f_locals)

    sys.setprofile(profile)
    try:
        result = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    if not captured:
        raise RuntimeError('Original router returned without captured locals')
    return result, captured


def _save(path, value):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary = stream.name
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(folder, name, value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.array(value, copy=True)
    if array.dtype.hasobject or array.nbytes > MAX_CAPTURE_BYTES:
        raise ValueError('Router diagnostic array is unbounded or has object dtype')
    if array.dtype.kind in 'fc' and not np.isfinite(array).all():
        raise FloatingPointError('Non-finite router diagnostic array')
    target = folder / (name + '.npy')
    with target.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(file=target.name, sha256=digest(target), bytes=target.stat().st_size,
                shape=list(array.shape), dtype=str(array.dtype))


def _load_artifact(folder, descriptor):
    import numpy as np
    relative = Path(descriptor['file'])
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError('Router artifact must be a direct relative file')
    path = folder / relative
    if (path.is_symlink() or path.stat().st_size != descriptor['bytes']
            or digest(path) != descriptor['sha256']):
        raise ValueError('Router diagnostic artifact changed')
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != descriptor['shape'] or str(array.dtype) != descriptor['dtype']:
        raise ValueError('Router artifact shape/dtype differs from its descriptor')
    if array.nbytes > MAX_CAPTURE_BYTES or (array.dtype.kind in 'fc' and not np.isfinite(array).all()):
        raise ValueError('Router artifact is unbounded or non-finite')
    return array


class TargetCaptured(Exception):
    """Intentional stop after raw routing values; not a generation success."""


def capture_worker(args):
    import numpy as np
    from benchmark_research51 import engine_manifest

    reference = json.loads(args.reference_report.read_text())
    if reference.get('backend_options',{}).get('router_reference_chunk',0):
        raise NotImplementedError('Target return-local capture supports the original router only; canonical router validation requires the complete RuntimeObserver ordered-route and full-logit evidence')
    expected = dict(git_head=reference['git_head'], source_sha256=reference['source_sha256'])
    if engine_manifest(args.engine_root) != expected:
        raise ValueError('Diagnostic engine differs from the original recorded source tree')
    turn = reference['turns'][args.turn]
    inputs = turn['full_input_ids']
    if not 0 <= args.position < len(inputs):
        raise ValueError('Target position is outside the original consumed prompt')
    sys.path.insert(0, str(args.engine_root / 'air_llm'))
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    from hyperstream2_observer import RuntimeObserver

    class CaptureObserver(RuntimeObserver):
        def _route_wrapper(self, original):
            locals_bank = {}

            @wraps(original)
            def router(*a, **kw):
                state = self._state()
                x = a[1] if len(a) > 1 else kw['x']
                layer = state.get('mlp_layer')
                start = state['positions'].get(layer, -1)
                targeted = (layer == args.layer and start <= args.position < start + int(x.shape[1]))
                if not targeted:
                    return original(*a, **kw)
                arithmetic = original.__globals__.get('_original_qwen_route',original)
                result, local_values = _call_with_locals(original, a, kw,locals_function=arithmetic)
                locals_bank.update(local_values)
                return result

            observed = super()._route_wrapper(router)

            @wraps(original)
            def observe(*a, **kw):
                result = observed(*a, **kw)
                if locals_bank:
                    state = self._state()
                    start = state['positions'][args.layer]
                    arrays = {name: _artifact(args.output_dir, name, locals_bank[name])
                              for name in ARRAY_NAMES
                              if name in locals_bank and locals_bank[name] is not None}
                    local_values = locals_bank
                    row = args.position - start
                    probabilities = np.asarray(local_values['probabilities'])[row]
                    ids = np.asarray(result[0])[row].astype(int)
                    weights = np.asarray(result[1])[row]
                    order = np.asarray(local_values['order'])[row].astype(int)
                    k = int(local_values['top_k'])
                    ranked = probabilities[order]
                    adjacent_ties = [int(i) for i in range(len(ranked)-1)
                                     if ranked[i] == ranked[i+1]]
                    report.update(status='target_captured_stopped_before_experts', arrays=arrays,
                        target=dict(layer=args.layer, absolute_position=args.position,
                                    chunk_start=start, chunk_tokens=int(local_values['flat'].shape[0]),
                                    row=row, selected_ids=ids.tolist(),
                                    normalized_weights=weights.tolist(),
                                    original_argsort_ids=order[:k].tolist(),
                                    top_k_boundary_tie=bool(ranked[k-1] == ranked[k])
                                    if k < len(ranked) else False,
                                    equal_probability_adjacent_ranks_zero_based=adjacent_ties),
                        router=dict(tie_policy=local_values.get('tie_policy'),
                                    top_k=k, experts=int(local_values['experts']),
                                    original_source_file_sha256=digest(Path(original.__code__.co_filename)),
                                    top_k_method='argsort(-softmax(float32(logits))) then first k; CPU Torch only on exact boundary ties',
                                    normalization='selected probabilities / sum(selected probabilities)',
                                    original_expert_weight_pairing_preserved=True,
                                    torch_boundary_fallback_used=local_values.get('reference_scores') is not None))
                    _save(args.output_dir / 'capture.json', report)
                    raise TargetCaptured()
                return result
            return observe

    report = dict(schema='airllm-target-router-capture-v1', status='started',
        reference_report_sha256=digest(args.reference_report), original_engine_manifest=expected,
        diagnostic_sha256=digest(__file__), model=dict(path=str(args.model), bytes=args.model.stat().st_size),
        source_report_qualifies_this_run=False, complete_generation=False,
        performance_measurement=False, fidelity_contract_changed=False,
        instrumentation='Target-only Python return-local capture; additional host copies and synchronization. No numerical replacement or post-hoc route sorting.',
        lazy_expression_note='Captured MLX intermediates are materialized after the original router IDs are observed. Replay can identify sensitivity; it does not prove internal fused-kernel intermediates were materialized in the uninstrumented run.',
        target_request=dict(layer=args.layer, absolute_position=args.position),
        original_options=reference['options'], original_backend_options=reference['backend_options'],
        full_input_ids=inputs)
    report['checkpoint_hash_cache_policy'] = dict(
        requested='Darwin F_NOCACHE on a separately opened read-only checkpoint descriptor',
        applied_before=False, applied_after=False, buffer_bytes=1 << 20,
        failure_policy='abort; no cached fallback', cache_purged=False,
        cold_cache_claimed=False, physical_ssd_bandwidth_certified=False)
    _save(args.output_dir / 'capture.json', report)
    before = checkpoint_digest(args.model)
    if before != args.expected_sha256:
        raise ValueError('Complete checkpoint hash differs before diagnostic inference')
    report['checkpoint_sha256_before'] = before
    report['checkpoint_hash_cache_policy']['applied_before'] = True
    _save(args.output_dir / 'capture.json', report)
    try:
        with NativeGGUFQwenModel(args.model, NextConfig(**reference['options']),
                                 **reference['backend_options']) as model:
            with CaptureObserver(model):
                model.forward(np.array([inputs], dtype=np.int32), logits_to_keep=1)
    except TargetCaptured:
        pass
    else:
        raise ValueError('Target route was not reached')
    if engine_manifest(args.engine_root) != expected:
        raise ValueError('Original engine sources changed during the diagnostic')
    after = checkpoint_digest(args.model)
    if after != before:
        raise ValueError('Complete checkpoint hash changed during diagnostic inference')
    report['checkpoint_sha256_after'] = after
    report['checkpoint_hash_cache_policy']['applied_after'] = True
    report['integrity_checked_before_after'] = True
    _save(args.output_dir / 'capture.json', report)
    return report


def pair_analysis(reference, candidate):
    """Read-only comparisons of captured facts; classification stays conditional."""
    import math
    import numpy as np
    left, right = [], []
    reports = []
    for path, bank in ((reference, left), (candidate, right)):
        report = json.loads(Path(path).read_text())
        if (report.get('status') != 'target_captured_stopped_before_experts'
                or not report.get('integrity_checked_before_after')
                or report.get('checkpoint_sha256_before') != report.get('checkpoint_sha256_after')):
            raise ValueError('Only complete integrity-checked targeted captures are comparable')
        arrays = {name: _load_artifact(Path(path).parent, desc)
                  for name, desc in report['arrays'].items()}
        bank.extend((report, arrays))
        reports.append(dict(path=str(path), sha256=digest(path)))
    a, x = left
    b, y = right
    if (a['checkpoint_sha256_before'] != b['checkpoint_sha256_before']
            or a['target_request'] != b['target_request']
            or a['full_input_ids'] != b['full_input_ids']):
        raise ValueError('Target/model/consumed prompt identities differ')
    if (x['weight'].shape != y['weight'].shape or x['weight'].dtype != y['weight'].dtype
            or not np.array_equal(x['weight'].view(np.uint8), y['weight'].view(np.uint8))):
        raise ValueError('Captured router weights differ')
    i, j = a['target']['row'], b['target']['row']
    ax, bx = x['routing_input'][i], y['routing_input'][j]
    al, bl = x['logits'][i], y['logits'][j]
    ap, bp = x['probabilities'][i], y['probabilities'][j]
    ids_a, ids_b = a['target']['selected_ids'], b['target']['selected_ids']
    affected = sorted({expert for rank, expert in enumerate(ids_a)
                       if rank >= len(ids_b) or ids_b[rank] != expert}
                      | {expert for rank, expert in enumerate(ids_b)
                         if rank >= len(ids_a) or ids_a[rank] != expert})
    exact = []
    for expert in affected:
        # Products of two FP32 inputs are exactly representable as FP64;
        # fsum supplies a high-accuracy diagnostic of the reduction, not a new router.
        exact.append(dict(expert=expert,
            reference_fsum_fp64=math.fsum(float(u)*float(v) for u,v in zip(ax,x['weight'][expert])),
            candidate_fsum_fp64=math.fsum(float(u)*float(v) for u,v in zip(bx,y['weight'][expert])),
            reference_logit=float(al[expert]), candidate_logit=float(bl[expert]),
            reference_probability=float(ap[expert]), candidate_probability=float(bp[expert])))
    return dict(schema='airllm-router-comparison-v1', captures=reports,
        strict_ordered_route_pass=ids_a == ids_b, selected_set_equal=set(ids_a) == set(ids_b),
        reference_ids=ids_a, candidate_ids=ids_b, affected_experts=exact,
        activation_bitwise_equal=np.array_equal(ax.view(np.uint8),bx.view(np.uint8)),
        maximum_activation_absolute_difference=float(np.max(np.abs(ax.astype('float64')-bx))),
        maximum_router_logit_absolute_difference=float(np.max(np.abs(al.astype('float64')-bl))),
        maximum_router_probability_absolute_difference=float(np.max(np.abs(ap.astype('float64')-bp))),
        captured_router_weights_bitwise_equal=True,
        reference_boundary_tie=a['target']['top_k_boundary_tie'],
        candidate_boundary_tie=b['target']['top_k_boundary_tie'],
        original_expert_weight_pairing_preserved=True,
        performance_gain_qualified=False, complete_generation=False,
        cause='See measured activation/logit/probability differences and controlled replays; rank permutation alone does not establish its cause.')


def replay_worker(args):
    """Isolate router GEMM/softmax shape using the very same captured inputs."""
    import importlib.metadata
    import numpy as np
    import mlx.core as mx
    record = json.loads(args.capture.read_text())
    if (record.get('status') != 'target_captured_stopped_before_experts'
            or not record.get('integrity_checked_before_after')
            or record.get('checkpoint_sha256_before') != record.get('checkpoint_sha256_after')):
        raise ValueError('Replay requires an integrity-checked original capture')
    bank = {name:_load_artifact(args.capture.parent,record['arrays'][name])
            for name in ('routing_input','weight','logits','probabilities','order')}
    if any(bank[name].dtype != np.dtype('float32') for name in ('routing_input','weight','logits','probabilities')):
        raise ValueError('This controlled replay implements the captured FP32 router only')
    row = record['target']['row']
    arrays = {}
    summary = {}
    with mx.stream(mx.gpu):
        x = mx.array(bank['routing_input'])
        w = mx.array(bank['weight'])
        for name,values in (('grouped',x),('singleton',x[row:row+1])):
            logits = values @ w.T
            probability = mx.softmax(logits.astype(mx.float32),axis=-1)
            order = mx.argsort(-probability,axis=-1)
            mx.eval(logits,probability,order)
            arrays.update({name+'_'+kind:_artifact(args.output_dir,name+'_'+kind,value)
                           for kind,value in (('logits',logits),('probabilities',probability),('order',order))})
            index = row if name == 'grouped' else 0
            host_logits,host_prob,host_order = np.array(logits)[index],np.array(probability)[index],np.array(order)[index]
            summary[name] = dict(topk_ids=host_order[:record['router']['top_k']].astype(int).tolist(),
                logits_bitwise_equal_original=host_logits.tobytes() == bank['logits'][row].tobytes(),
                probabilities_bitwise_equal_original=host_prob.tobytes() == bank['probabilities'][row].tobytes(),
                max_abs_logit_difference_original=float(np.max(np.abs(host_logits.astype('float64')-bank['logits'][row]))))
    _save(args.output_dir/'replay.json',dict(schema='airllm-router-shape-replay-v1',
        capture_sha256=digest(args.capture),mlx=importlib.metadata.version('mlx'),device='gpu',
        same_input_and_weight_values_for_both_shapes=True,
        shapes=dict(grouped=list(bank['routing_input'].shape),singleton=[1,bank['routing_input'].shape[1]]),
        results=summary,arrays=arrays,performance_gain_qualified=False,
        scope='Router GEMM + FP32 softmax + argsort replay on identical captured values. No expert or full-model fidelity claim; no original fused-intermediate equivalence is assumed.'))


def _guard_command(command, output_dir):
    sys.path.insert(0, str(ROOT/'air_llm'))
    from airllm.nextgen.experiment_guard import ExperimentLimits, supervise
    target = output_dir/'controller.json'
    try:
        guard = supervise(command,target,limits=ExperimentLimits(timeout_seconds=900.),
            env=dict(os.environ, MLX_ENABLE_TF32='0', PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    except MemoryError:
        raise SystemExit(75)
    except RuntimeError:
        saved = json.loads(target.with_suffix('.guard.json').read_text())
        temporary = saved.get('violation') in ('system_reserve','system_swap_growth','macos_memory_pressure')
        raise SystemExit(75 if temporary else 1)
    print(json.dumps(dict(exit_code=guard['exit_code'],violation=guard['violation'],output=str(output_dir))))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest='mode', required=True)
    capture = sub.add_parser('capture', allow_abbrev=False)
    capture.add_argument('--reference-report', type=Path, required=True)
    capture.add_argument('--engine-root', type=Path, required=True)
    capture.add_argument('--model', type=Path, required=True)
    capture.add_argument('--expected-sha256', required=True)
    capture.add_argument('--output-dir', type=Path, required=True)
    capture.add_argument('--layer', type=int, default=30)
    capture.add_argument('--position', type=int, default=154)
    capture.add_argument('--turn', type=int, default=0)
    capture.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    compare = sub.add_parser('compare', allow_abbrev=False)
    compare.add_argument('reference', type=Path)
    compare.add_argument('candidate', type=Path)
    compare.add_argument('--output', type=Path, required=True)
    replay = sub.add_parser('replay', allow_abbrev=False)
    replay.add_argument('--capture',type=Path,required=True)
    replay.add_argument('--output-dir',type=Path,required=True)
    replay.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.mode == 'compare':
        result = pair_analysis(args.reference, args.candidate)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(output=str(args.output), strict_ordered_route_pass=result['strict_ordered_route_pass'])))
        return
    if args.mode == 'replay':
        args.capture = args.capture.expanduser().resolve(strict=True)
        args.output_dir = args.output_dir.expanduser().absolute()
        if args.worker:
            replay_worker(args)
            return
        args.output_dir.mkdir(parents=True,exist_ok=False)
        if shutil.disk_usage(args.output_dir).free < DISK_RESERVE:
            raise SystemExit(75)
        _guard_command([sys.executable,str(Path(__file__).resolve()),'replay','--worker',
                        '--capture',str(args.capture),'--output-dir',str(args.output_dir)],args.output_dir)
        return
    for name in ('reference_report', 'engine_root', 'model'):
        setattr(args,name,getattr(args,name).expanduser().resolve(strict=True))
    args.output_dir = args.output_dir.expanduser().absolute()
    if args.worker:
        capture_worker(args)
        return
    if len(args.expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_sha256):
        p.error('--expected-sha256 must be the complete lowercase SHA-256')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if min(shutil.disk_usage(path).free for path in (args.output_dir,args.model.parent)) < DISK_RESERVE:
        _save(args.output_dir/'admission.json',dict(status='deferred_disk_reserve',disk_reserve_bytes=DISK_RESERVE))
        raise SystemExit(75)
    command = [sys.executable,str(Path(__file__).resolve()),'capture','--worker',
        '--reference-report',str(args.reference_report),'--engine-root',str(args.engine_root),
        '--model',str(args.model),'--expected-sha256',args.expected_sha256,
        '--output-dir',str(args.output_dir),'--layer',str(args.layer),
        '--position',str(args.position),'--turn',str(args.turn)]
    _guard_command(command,args.output_dir)


if __name__ == '__main__':
    main()
