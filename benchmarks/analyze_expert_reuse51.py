#!/usr/bin/env python3
"""Analyze recorded routes only: no model imports, weight reads, or generation.

Outputs count expert requests and ideal union reuse. Counts are not SSD bytes,
load times, accepted draft tokens, concurrent inference, or model throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
import unittest

MAX_MEMBER_BYTES = 8 << 20
MAX_ARCHIVE_MEMBERS = 4096


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def historical_route_hash(routes):
    # The runtime hashes integer expert keys. JSON reload turns them into strings;
    # restore that type before sorting, otherwise expert 10 precedes expert 2.
    original_types = {str(layer): {int(expert): count for expert, count in counts.items()}
                      for layer, counts in routes.items()}
    return digest(json.dumps(original_types, sort_keys=True).encode())


def parse_routes(routes):
    if not isinstance(routes, dict) or not routes:
        raise ValueError("missing route identities")
    parsed = {}
    for layer, counts in routes.items():
        layer = int(layer)
        if layer < 0 or not isinstance(counts, dict) or not counts:
            raise ValueError("invalid layer route histogram")
        selected = set()
        for expert, count in counts.items():
            expert = int(expert)
            if expert < 0 or type(count) is not int or count != 1:
                raise ValueError("decode requires one selection per expert; aggregated prefill is excluded")
            selected.add(expert)
        if len(selected) != len(counts):
            raise ValueError("duplicate normalized expert id")
        parsed[layer] = frozenset(selected)
    if len(parsed) != len(routes):
        raise ValueError("duplicate normalized layer id")
    return parsed


def extract_archive(path):
    groups = {}
    with tarfile.open(path, "r:gz") as archive:
        for number, member in enumerate(archive):
            if number >= MAX_ARCHIVE_MEMBERS:
                raise ValueError("archive member count exceeds analysis bound")
            if not member.isfile() or not member.name.endswith(".json") or member.name.endswith(".guard.json"):
                continue
            if member.size > MAX_MEMBER_BYTES:
                raise ValueError("JSON member exceeds analysis bound")
            raw = archive.extractfile(member).read(MAX_MEMBER_BYTES + 1)
            document = json.loads(raw)
            if not isinstance(document, dict) or not document.get("capture_enabled"):
                continue
            if document.get("status") != "completed":
                continue
            for turn in document.get("turns", []):
                captures = turn.get("captures", [])
                decode = [capture for capture in captures if capture["step"] > 0]
                if not decode:
                    continue
                steps = [capture["step"] for capture in decode]
                positions = [capture["position"] for capture in decode]
                if steps != list(range(1, len(decode) + 1)) or positions != list(range(positions[0], positions[0] + len(decode))):
                    raise ValueError("incomplete/noncausal decode capture")
                frames = [parse_routes(capture["routes"]) for capture in decode]
                layers = set(frames[0])
                if any(set(frame) != layers for frame in frames):
                    raise ValueError("decode layer set changed")
                widths = {len(experts) for frame in frames for experts in frame.values()}
                if len(widths) != 1:
                    raise ValueError("variable top-k requires a different analysis schema")
                checks = {check["step"]: check for check in turn.get("step_checks", [])}
                for capture in decode:
                    check = checks.get(capture["step"])
                    if check and check["routes_sha256"] != historical_route_hash(capture["routes"]):
                        raise ValueError("capture route hash differs from timed check")
                identity = digest(canonical({
                    "model_file_bytes": document.get("model_file_bytes"),
                    "input_ids": turn.get("full_input_ids"),
                    "generated_ids": turn.get("generated_ids"),
                    "decode": [{key: value for key, value in c.items() if key in ("step", "position", "routes", "logits_sha256")} for c in decode],
                }))
                source = {"archive_member": member.name, "member_sha256": digest(raw),
                          "historical_git_head": document.get("git_head"), "task_id": turn.get("id"),
                          "variant": document.get("variant"), "capture_count": len(captures),
                          "decode_hashes_verified": sum(c["step"] in checks for c in decode)}
                if identity in groups:
                    groups[identity]["sources"].append(source)
                else:
                    groups[identity] = {"frames": frames, "sources": [source], "trace_sha256": identity,
                                        "positions": positions, "top_k": widths.pop()}
    if not groups:
        raise ValueError("no recorded decode route identities in archive")
    return list(groups.values())


def frequency_summary(frames):
    counts = Counter((layer, expert) for frame in frames for layer, selected in frame.items() for expert in selected)
    return {"requests": sum(counts.values()), "unique_layer_expert_pairs": len(counts),
            "by_layer": [{"layer": layer, "unique_experts": len({expert for (candidate, expert) in counts if candidate == layer}),
                          "top_frequencies": [{"expert": expert, "count": count} for (candidate, expert), count in
                                               sorted(counts.items(), key=lambda item: (-item[1], item[0])) if candidate == layer][:12]}
                         for layer in sorted(frames[0])]}


def temporal_unions(frames, depths=(1, 2, 4)):
    result = []
    for depth in depths:
        if depth > len(frames):
            continue
        requested = unique = windows = 0
        for start in range(len(frames) - depth + 1):
            for layer in sorted(frames[0]):
                selected = [frames[start + offset][layer] for offset in range(depth)]
                requested += sum(map(len, selected))
                unique += len(set().union(*selected))
                windows += 1
        result.append({"depth": depth, "layer_windows": windows, "ungrouped_expert_requests": requested,
                       "union_expert_requests": unique, "ideal_request_reduction_factor": requested / unique,
                       "saved_requests": requested - unique})
    return result


def cache_replay(frames, capacity, policy):
    """Equal-unit global cache; a layer's current demands cannot evict each other.

    Histograms do not preserve the router's top-k order. Stable expert-id order
    is therefore used for admission and disclosed; no byte/time claim follows.
    """
    cache, frequencies = {}, Counter()
    hits = misses = bypasses = clock = 0
    for frame in frames:
        for layer in sorted(frame):
            clock += 1
            wants = {(layer, expert) for expert in frame[layer]}
            frequencies.update(wants)
            missing = sorted(wants - cache.keys())
            hits += len(wants) - len(missing)
            misses += len(missing)
            for key in wants & cache.keys():
                cache[key] = clock
            for key in missing:
                if capacity == 0:
                    bypasses += 1
                    continue
                if len(cache) >= capacity:
                    eligible = cache.keys() - wants
                    if not eligible:
                        bypasses += 1
                        continue
                    if policy == "lru":
                        victim = min(eligible, key=lambda item: (cache[item], item))
                    elif policy in ("lfu", "lfu_admit"):
                        victim = min(eligible, key=lambda item: (frequencies[item], cache[item], item))
                    else:
                        raise ValueError("unknown cache policy")
                    if policy == "lfu_admit" and frequencies[key] < frequencies[victim]:
                        bypasses += 1
                        continue
                    del cache[victim]
                cache[key] = clock
            assert len(cache) <= capacity
    return {"capacity_expert_units": capacity, "policy": policy, "hits": hits, "misses": misses,
            "bypassed_admissions": bypasses, "requests": hits + misses, "final_resident_units": len(cache)}


def causal_predictors(frames):
    """Predict before learning the current target; experts remain actual demands."""
    frequency, transitions = defaultdict(Counter), defaultdict(Counter)
    totals = {name: Counter() for name in ("previous_token", "causal_frequency", "causal_transition")}
    previous = None
    for frame in frames:
        if previous is not None:
            for layer, actual in frame.items():
                k = len(actual)
                scores = Counter()
                for expert in previous[layer]:
                    scores.update(transitions[(layer, expert)])
                candidates = set(scores) | set(frequency[layer])
                predictions = {
                    "previous_token": previous[layer],
                    "causal_frequency": set(sorted(frequency[layer], key=lambda expert: (-frequency[layer][expert], expert))[:k]),
                    "causal_transition": set(sorted(candidates, key=lambda expert: (-scores[expert], -frequency[layer][expert], expert))[:k]),
                }
                for name, predicted in predictions.items():
                    totals[name].update({"proposed_experts": len(predicted), "actual_experts": len(actual),
                                         "correct_experts": len(predicted & actual), "wrong_experts": len(predicted - actual)})
        # Current observations become available only after the scoring above.
        for layer, actual in frame.items():
            frequency[layer].update(actual)
            if previous is not None:
                for expert in previous[layer]:
                    transitions[(layer, expert)].update(actual)
        previous = frame
    return [{"predictor": name, **dict(counts), "recall": counts["correct_experts"] / counts["actual_experts"] if counts["actual_experts"] else None}
            for name, counts in totals.items()]


def olmoe_replay(path, start, positions):
    # NumPy loads a stored route-id trace only; no inference framework is imported.
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids_long"]
        categories = archive["cats_long"].tolist()
        experts = int(archive["N"])
    if ids.ndim != 4 or ids.dtype.kind not in "iu":
        raise ValueError("OLMoE identities must be a four-dimensional integer array")
    layers, streams, total_positions, k = ids.shape
    if not 0 <= start < total_positions or positions < 1 or start + positions > total_positions:
        raise ValueError("requested teacher-forced range is outside captured trace")
    ids = ids[:, :, start:start + positions, :]
    if np.any(ids < 0) or np.any(ids >= experts) or np.any(np.diff(np.sort(ids, axis=-1), axis=-1) <= 0):
        raise ValueError("invalid or repeated expert identity")
    rows = []
    for width in (1, 2, 4, 8):
        if width > streams:
            continue
        requests = unique = groups = 0
        # Every category combination avoids claiming one selected pair is typical.
        for combo in itertools.combinations(range(streams), width):
            selected = ids[:, combo, :, :].transpose(0, 2, 1, 3).reshape(layers, positions, width * k)
            ordered = np.sort(selected, axis=-1)
            unique += int((1 + np.sum(ordered[..., 1:] != ordered[..., :-1], axis=-1)).sum())
            requests += layers * positions * width * k
            groups += 1
        rows.append({"streams": width, "category_combinations": groups, "positions_per_stream": positions,
                     "ungrouped_expert_requests": requests, "union_expert_requests": unique,
                     "ideal_request_reduction_factor": requests / unique, "saved_requests": requests - unique})
    frequencies = []
    for layer in range(layers):
        counts = np.bincount(ids[layer].reshape(-1), minlength=experts)
        order = np.lexsort((np.arange(experts), -counts))[:12]
        frequencies.append({"layer": layer, "unique_experts": int(np.count_nonzero(counts)),
                            "top_frequencies": [{"expert": int(expert), "count": int(counts[expert])} for expert in order]})
    return {"evidence": "new offline reanalysis of inherited teacher-forced route identities",
            "trace_sha256": digest(path.read_bytes()), "full_trace_shape": [layers, streams, total_positions, k],
            "categories": categories, "position_range_zero_based": [start, start + positions],
            "independent_free_generations": False, "counterfactual_aligned_category_scheduling": True,
            "requests": int(ids.size), "by_layer": frequencies, "multi_sequence_union_replay": rows}


def analyze(args):
    traces = extract_archive(args.archive)
    summaries = []
    for trace in traces:
        frames = trace.pop("frames")
        summaries.append({**trace, "decode_steps": len(frames), "layers": len(frames[0]),
                          "frequency": frequency_summary(frames), "temporal_unions": temporal_unions(frames),
                          "causal_prediction": causal_predictors(frames),
                          "cold_global_cache_replays": [cache_replay(frames, capacity, policy)
                                                        for capacity in (0, 64, 128, 256, 512)
                                                        for policy in ("lru", "lfu", "lfu_admit")]})
    result = {"kind": "offline_route_reanalysis_51", "archive_sha256": digest(args.archive.read_bytes()),
              "script_sha256": digest(Path(__file__).read_bytes()), "new_model_forwards": 0,
              "historical_inference_timings_rebenchmarked": False, "generation_speedup_measured": False,
              "physical_io_measured": False, "expert_bytes_measured": None, "expert_load_seconds_measured": None,
              "unique_free_decode_traces": len(traces), "captured_source_reports": sum(len(trace["sources"]) for trace in traces),
              "decode_traces": summaries,
              "limits": ["Prefill aggregate histograms are excluded from chronological decode analysis.",
                         "Identical parity variants are deduplicated, never counted as independent sequences.",
                         "Expert histogram ordering is lost; equal-unit cache admissions use stable expert-id order.",
                         "No per-expert bytes, costs, KV memory or concurrent execution are recorded here.",
                         "Temporal union counts are not routes under rejected drafts or draft acceptance.",
                         "Cache replay and union factors are request counts, not measured speedups."]}
    if args.olmoe_trace is not None:
        result["olmoe_teacher_forced"] = olmoe_replay(args.olmoe_trace, args.start, args.positions)
    return result


class AnalysisTests(unittest.TestCase):
    def _archive_fixture(self, target, corrupt=False):
        import io
        routes = {"0": {"2": 1, "10": 1}}
        document = {"capture_enabled": True, "status": "completed", "model_file_bytes": 123,
                    "turns": [{"id": "fixture", "full_input_ids": [1], "generated_ids": [2],
                               "captures": [{"step": 0, "position": 1, "routes": {"0": {"2": 3}}},
                                            {"step": 1, "position": 2, "routes": routes, "logits_sha256": "fixture"}],
                               "step_checks": [{"step": 1, "routes_sha256": "bad" if corrupt else historical_route_hash(routes)}]}]}
        raw = json.dumps(document).encode()
        with tarfile.open(target, "w:gz") as archive:
            for name in ("reference.json", "candidate.json"):
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))

    def test_archive_deduplicates_variants_and_excludes_prefill(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "captures.tar.gz"
            self._archive_fixture(path)
            traces = extract_archive(path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0]["sources"]), 2)
        self.assertEqual(traces[0]["frames"], [{0: frozenset((2, 10))}])
        self.assertTrue(all(row["recall"] is None for row in causal_predictors(traces[0]["frames"])))

    def test_archive_refuses_a_tampered_route_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "captures.tar.gz"
            self._archive_fixture(path, corrupt=True)
            with self.assertRaisesRegex(ValueError, "route hash differs"):
                extract_archive(path)
    def test_historical_hash_restores_integer_sorting(self):
        original = {"0": {2: 1, 10: 1}}
        loaded = json.loads(json.dumps(original))
        self.assertEqual(historical_route_hash(loaded), digest(json.dumps(original, sort_keys=True).encode()))

    def test_prefill_histogram_is_not_decode(self):
        with self.assertRaises(ValueError):
            parse_routes({"0": {"1": 2}})

    def test_union_counts_overlap_without_future_invention(self):
        frames = [{0: {1, 2}}, {0: {2, 3}}]
        row = temporal_unions(frames, (2,))[0]
        self.assertEqual((row["ungrouped_expert_requests"], row["union_expert_requests"]), (4, 3))

    def test_predictor_does_not_learn_target_before_prediction(self):
        rows = causal_predictors([{0: {1}}, {0: {2}}, {0: {3}}])
        self.assertTrue(all(row["correct_experts"] == 0 for row in rows))
        self.assertTrue(all(row["actual_experts"] == 2 for row in rows))

    def test_cache_budget_and_current_demands(self):
        frames = [{0: {1, 2}}, {0: {1, 3}}, {0: {1, 2}}]
        for policy in ("lru", "lfu", "lfu_admit"):
            row = cache_replay(frames, 2, policy)
            self.assertEqual((row["hits"], row["misses"], row["final_resident_units"]), (2, 4, 2))
        self.assertEqual(cache_replay(frames, 0, "lru")["misses"], 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("evidence-throughput/raw-json-logs.tar.gz"))
    parser.add_argument("--olmoe-trace", type=Path)
    parser.add_argument("--start", type=int, default=512)
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("research/research51/results/expert-reuse.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"kind": result["kind"], "unique_free_decode_traces": result["unique_free_decode_traces"],
                      "captured_source_reports": result["captured_source_reports"], "output": str(args.output),
                      "new_model_forwards": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
