#!/usr/bin/env python3
"""Extra expert-load instrumentation; not a primary speed comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import benchmark_research51 as benchmark

WRAPPER = Path(__file__).resolve()
BASE = Path(benchmark.__file__).resolve()
EXPERT = re.compile(r"(?:^|\.)ffn_(?:gate|up|down)_exps\.weight$")


def instrument_packed_tile(original):
    def measured(self, name, *args, **kwargs):
        if not EXPERT.search(name):
            return original(self, name, *args, **kwargs)
        loads, begin = self.stats.get('loads', 0), perf_counter()
        try:
            result = original(self, name, *args, **kwargs)
        except BaseException:
            self.stats['expert_packed_failed_calls'] = self.stats.get('expert_packed_failed_calls', 0) + 1
            self.stats['expert_packed_failed_seconds'] = self.stats.get('expert_packed_failed_seconds', 0.) + perf_counter() - begin
            raise
        elapsed = perf_counter() - begin
        kind = 'miss_load' if self.stats.get('loads', 0) > loads else 'cache_hit'
        for key, value in (('calls', 1), ('seconds', elapsed), ('bytes', int(result.nbytes))):
            metric = 'expert_packed_' + kind + '_' + key
            self.stats[metric] = self.stats.get(metric, 0) + value
        return result
    return measured


def profiled_worker(args):
    # Import the selected engine only inside the supervised worker.
    if not benchmark.protocol_unchanged(json.loads(args.expected_protocol_json)):
        raise RuntimeError('Research profiling protocol changed before engine import')
    engine = benchmark.engine_manifest(args.engine_root)
    if (args.expected_engine_head is not None and engine['git_head'] != args.expected_engine_head
        or args.expected_engine_sources_json is not None and engine['source_sha256'] != json.loads(args.expected_engine_sources_json)):
        raise RuntimeError('Engine differs from its frozen campaign manifest')
    sys.path.insert(0, str(args.engine_root / 'air_llm'))
    from airllm.nextgen.gguf_qwen import GGUFWeights
    original, save = GGUFWeights._packed_tile, benchmark.save
    source_raw = WRAPPER.read_bytes()
    metadata = dict(extra_instrumentation=True, primary_speed_comparison=False,
        wrapper_path=str(WRAPPER), wrapper_source=source_raw.decode(), wrapper_sha256=hashlib.sha256(source_raw).hexdigest(),
        base_harness_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        scope='Only ffn_gate_exps.weight, ffn_up_exps.weight and ffn_down_exps.weight packed selections.',
        semantics='Successful _packed_tile calls classified by delta reader.stats.loads; seconds cover the full helper including cache lookup, read, admission and any synchronization. Bytes are selected encoded bytes, not physical storage bytes. Failed calls are separate. Instrumentation adds overhead.')
    def profiled_save(path, report):
        report['expert_load_profiling'] = metadata
        save(path, report)
    GGUFWeights._packed_tile, benchmark.save = instrument_packed_tile(original), profiled_save
    try:
        return benchmark.worker_original(args)
    finally:
        GGUFWeights._packed_tile, benchmark.save = original, save


def main():
    # Reuse the complete baseline parser, its protected outputs and supervisor.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--expected-protocol-json')
    known, _ = parser.parse_known_args()
    if known.tasks is not None:
        protocol = (json.loads(known.expected_protocol_json) if known.expected_protocol_json
                    else benchmark.protocol_manifest([BASE, known.tasks.expanduser().resolve()]))
        for source in (BASE, WRAPPER):
            protocol.setdefault(str(source), hashlib.sha256(source.read_bytes()).hexdigest())
        sys.argv.extend(['--expected-protocol-json', json.dumps(protocol)])
    if '--worker' in sys.argv[1:]:
        benchmark.worker_original, benchmark.worker = benchmark.worker, profiled_worker
    else:
        sys.path.insert(0, str(benchmark.ROOT / 'air_llm'))
        from airllm.nextgen import experiment_guard
        supervise = experiment_guard.supervise
        def profiled_supervise(command, *args, **kwargs):
            command = [*command]
            command[1] = str(WRAPPER)
            return supervise(command, *args, **kwargs)
        experiment_guard.supervise = profiled_supervise
    try:
        benchmark.main()
    finally:
        if '--worker' not in sys.argv[1:]:
            experiment_guard.supervise = supervise


if __name__ == '__main__':
    main()
