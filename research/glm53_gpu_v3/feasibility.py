"""Quantitative diagnostics and explicit conditional storage bounds.

This helper reads saved operation metadata only. It neither reads model weights
nor launches inference. Logical byte sums are never silently promoted to unique
required bytes, physical storage transfers, NAND reads or cache hits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

GIB = 1024**3
DEFAULT_BUDGETS = (0, 1*GIB, 2*GIB, 3*GIB, 4*GIB, 6*GIB, 8*GIB)


def _nonnegative(value, label, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {label}")
    if integer and type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def io_coverage_bound(unique_required_bytes, retained_coverage_bytes, bandwidth_bytes_per_second):
    """Necessary interface-service bound under the stated coverage assumptions.

    U counts distinct original bytes the implementation must consume for one
    token. C counts how many of those bytes are already covered by every cache
    tier before that token, expressed in original-byte coverage (not RSS).
    With no alternative source, at least max(U-C,0) bytes cross the storage
    interface. This says nothing about NAND traffic or actual compute overlap.
    A caller must independently establish U, C and the service-rate cap B.
    """
    u = _nonnegative(unique_required_bytes, "unique_required_bytes", integer=True)
    c = _nonnegative(retained_coverage_bytes, "retained_coverage_bytes", integer=True)
    b = _nonnegative(bandwidth_bytes_per_second, "bandwidth_bytes_per_second", positive=True)
    transferred = max(0, u-c)
    return {
        "unique_required_bytes": u,
        "retained_original_byte_coverage": c,
        "interface_rate_bytes_per_second": b,
        "minimum_interface_bytes_per_token": transferred,
        "minimum_storage_service_seconds_per_token": transferred/b,
        "maximum_io_only_tokens_per_second": b/transferred if transferred else None,
        "bound_nonbinding_when_transfer_zero": transferred == 0,
        "scope": "conditional necessary interface bound; excludes compute/latency/overhead; not NAND traffic",
    }


def required_coverage(unique_required_bytes, bandwidth_bytes_per_second, tokens_per_second):
    u = _nonnegative(unique_required_bytes, "unique_required_bytes", integer=True)
    b = _nonnegative(bandwidth_bytes_per_second, "bandwidth_bytes_per_second", positive=True)
    r = _nonnegative(tokens_per_second, "tokens_per_second", positive=True)
    # Ceiling preserves a necessary whole-byte coverage requirement.
    minimum = max(0, math.ceil(u-b/r))
    return {"tokens_per_second": r, "minimum_retained_original_byte_coverage": minimum,
            "storage_byte_budget_per_token": b/r,
            "scope": "necessary under supplied U and B, before any compute or safety headroom"}


def critical_path_reduction(speedup_factor):
    s = _nonnegative(speedup_factor, "speedup_factor", positive=True)
    if s < 1:
        raise ValueError("a requested speedup must be at least one")
    return 1-1/s


def _new_group():
    return {"operations": 0, "op_wall_seconds": 0., "inspection_seconds_excluded": 0.,
            "cpu_seconds_known_sum": 0., "cpu_counter_operations": 0,
            "os_read_bytes_known_sum": 0, "io_counter_operations": 0,
            "pageins_known_sum": 0, "pagein_counter_operations": 0,
            "logical_weight_bytes_known_sum": 0, "known_weight_operations": 0}


def _add(group, row, logical):
    group["operations"] += 1
    group["op_wall_seconds"] += _nonnegative(row["seconds"], "seconds")
    pages = row.get("weight_pages_before") or {}
    group["inspection_seconds_excluded"] += _nonnegative(pages.get("inspection_seconds", 0), "inspection_seconds")
    for key, count in (("cpu_seconds", "cpu_counter_operations"), ("os_read_bytes", "io_counter_operations"),
                       ("pageins", "pagein_counter_operations")):
        value = row.get(key)
        if value is not None:
            group[key+"_known_sum"] += _nonnegative(value, key, integer=key != "cpu_seconds")
            group[count] += 1
    if logical is not None:
        group["logical_weight_bytes_known_sum"] += logical
        group["known_weight_operations"] += 1


def _finished(group, steps, total_wall):
    out = dict(group)
    for key, count in (("cpu_seconds", "cpu_counter_operations"), ("os_read_bytes", "io_counter_operations"),
                       ("pageins", "pagein_counter_operations")):
        # Preserve incomplete measurement as unknown, with the known sum/count
        # available separately. Missing counters are never replaced by zeros.
        out[key] = group[key+"_known_sum"] if group[count] == group["operations"] else None
    out["op_wall_fraction_of_serialized_decode"] = group["op_wall_seconds"]/total_wall if total_wall else None
    out["mean_per_decode_step"] = {
        key: out[key]/steps if out[key] is not None else None
        for key in ("op_wall_seconds", "inspection_seconds_excluded", "cpu_seconds", "os_read_bytes", "pageins")}
    out["mean_per_decode_step"]["logical_weight_bytes_known_sum"] = group["logical_weight_bytes_known_sum"]/steps
    return out


def summarize_operations(rows):
    """Partition saved serialized decode operations; do not infer pure compute."""
    total = _new_group()
    categories = defaultdict(_new_group)
    sites = defaultdict(_new_group)
    step_groups = defaultdict(_new_group)
    steps = set()
    unknown_weight_observations = 0
    for row in rows:
        step = row.get("step")
        if type(step) is not int or step < 0 or not isinstance(row.get("op"), str):
            raise ValueError("operation requires a nonnegative integer step and op name")
        if step == 0:
            continue
        steps.add(step)
        pages = row.get("weight_pages_before") or {}
        if not isinstance(pages, dict):
            raise ValueError("weight_pages_before must be an object")
        logical = None
        is_weight = (pages.get("status") == "ok" and row["op"] in ("MUL_MAT", "MUL_MAT_ID"))
        if is_weight:
            logical = _nonnegative(pages.get("logical_selected_weight_bytes"), "logical_selected_weight_bytes", integer=True)
            is_weight = logical > 0
        if is_weight:
            category = "expert_weight_matmul" if row["op"] == "MUL_MAT_ID" else "dense_weight_matmul"
        else:
            category = "other_or_unattributed_operations"
            if pages.get("status") in ("error", "unsupported", "not_inspected"):
                unknown_weight_observations += 1
        source = row.get("source_weight")
        source_name = source.get("name") if isinstance(source, dict) else None
        name = source_name or row.get("name", "unnamed")
        if not isinstance(name, str):
            raise ValueError("operation/source name must be a string")
        key = (category, row["op"], name)
        for group in (total, categories[category], sites[key], step_groups[step]):
            _add(group, row, logical if is_weight else None)
    if not steps or total["op_wall_seconds"] <= 0:
        raise ValueError("nonempty timed decode operations required")
    n = len(steps)
    wall = total["op_wall_seconds"]
    top = [{"category": key[0], "op": key[1], "site": key[2], **_finished(value, n, wall)}
           for key, value in sites.items()]
    top.sort(key=lambda x: (-x["op_wall_seconds"], x["site"]))
    return {"scope": "saved serialized diagnostic operations, not production throughput or pure-kernel timings",
            "decode_step_ids": sorted(steps), "decode_steps": n,
            "total": _finished(total, n, wall),
            "categories": {key: _finished(value, n, wall) for key, value in categories.items()},
            "by_step": {str(step): _finished(value, 1, wall) for step, value in sorted(step_groups.items())},
            "top_sites_by_wall": top[:25],
            "unknown_weight_observations": unknown_weight_observations,
            "unique_required_weight_bytes_per_token": None,
            "physical_ssd_bytes_per_token": None,
            "file_cache_hit_bytes_per_token": None,
            "pure_compute_seconds_per_token": None,
            "pure_io_wait_seconds_per_token": None,
            "interpretation": [
                "dense/expert groups are interleaved across layers; their wall times partition this serialized diagnostic only",
                "operation wall includes arithmetic, faults, memory, backend waits and scheduler overhead",
                "CPU seconds sum threads and must not be subtracted from wall time to infer IO",
                "known logical sums may count repeated ranges and are not unique bytes or physical reads",
                "missing counters remain null; residency snapshots are not global file-cache hit rates",
                "inspection is excluded from operation timers but may perturb execution",
            ]}


def conditional_scenarios(volume_bytes, *, bandwidth_bytes_per_second=1_000_000_000, budgets=DEFAULT_BUDGETS):
    """Sensitivity only: the caller must establish volume as distinct required U."""
    _nonnegative(volume_bytes, "volume_bytes", integer=True)
    return {"assumed_unique_bytes_per_token": volume_bytes,
            "assumption_status": "unverified sensitivity unless independent source-range union establishes this U",
            "bandwidth_scope": "supplied interface-service scenario, not measured or certified by this helper",
            "retention_scope": "original-byte coverage across ALL cache tiers, not interchangeable with RSS or a GPU allocation budget",
            "cache_scenarios": [io_coverage_bound(volume_bytes, c, bandwidth_bytes_per_second) for c in budgets],
            "target_requirements": [required_coverage(volume_bytes, bandwidth_bytes_per_second, r) for r in (.5, 1., 3., 5.)],
            "excluded": ["compute cost", "read-command latency", "copy/synchronization overhead", "thermal limits", "safety headroom"],
            "limits": [
                "retained compressed or auxiliary-cache data must count by the original byte coverage it supplies",
                "the storage interface can serve controller-cache data; this is not a NAND read bound",
                "an advertised or user-estimated bandwidth is not a measured hard bandwidth cap",
                "if all required bytes are covered, the IO bound is nonbinding rather than infinite measured throughput",
            ]}


def closed_source_bounds(proof, *, bandwidth_bytes_per_second=1_000_000_000):
    """Bind an independently generated closed source union, preserving scope."""
    if proof.get("all_steps_closed") is not True or proof.get("decode_steps_closed") is not True:
        raise ValueError("complete source-operand union proof required")
    rows = proof.get("per_step")
    if (not isinstance(rows, list) or len(rows) < 2 or
            [r.get("step") for r in rows] != list(range(len(rows)))):
        raise ValueError("source proof needs contiguous completed steps including decode")
    values = []
    for row in rows:
        if (row.get("all_expected_fixed_matrices_and_expert_banks_resolved") is not True or
                row.get("unknown_sources") or row.get("missing_dense_names") or row.get("missing_expert_banks") or
                row.get("incomplete_dense_payload_names") or row.get("unexpected_weight_sources")):
            raise ValueError("source proof contains unresolved or incomplete operands")
        dense = _nonnegative(row.get("distinct_dense_operand_bytes"), "distinct dense bytes", integer=True)
        expert = _nonnegative(row.get("distinct_selected_expert_operand_bytes"), "distinct expert bytes", integer=True)
        overlap = _nonnegative(row.get("dense_expert_overlap_bytes"), "overlap bytes", integer=True)
        total = _nonnegative(row.get("distinct_known_matmul_operand_bytes"), "unique operand bytes", integer=True, positive=True)
        if overlap > min(dense, expert) or total != dense+expert-overlap:
            raise ValueError("source-union byte accounting is inconsistent")
        if (row.get("distinct_dense_names") != proof.get("expected_dense_matrices") or
                row.get("distinct_expert_bank_names") != proof.get("expected_expert_banks")):
            raise ValueError("source-union name coverage is inconsistent")
        phase = row.get("phase", "prefill" if row["step"] == 0 else "decode")
        if phase not in ("prefill", "decode"):
            raise ValueError("unknown source-proof phase")
        if phase == "decode":
            values.append(total)
    if not values:
        raise ValueError("source proof has no completed decode step")
    if proof.get("decode_distinct_known_bytes_range") != [min(values), max(values)]:
        raise ValueError("source-union range does not match saved per-step values")
    model = conditional_scenarios(min(values), bandwidth_bytes_per_second=bandwidth_bytes_per_second)
    model["assumption_status"] = "U is the proved distinct declared operand volume in the supplied closed graph capture; B and C remain conditional"
    return {"scope": "supplied closed original MUL_MAT/MUL_MAT_ID operand union; not storage transfers or every possible algorithm",
            "decode_steps": len(values), "decode_unique_operand_bytes_range": [min(values), max(values)],
            "conditional_storage_model": model,
            "conservative_volume_choice": "minimum captured decode U; a necessary conditional bound for each captured decode step",
            "non_matmul_weights_included": False,
            "limitations": proof.get("limitations", [])}


def build_report(profile, *, bandwidth_bytes_per_second=1_000_000_000, source_union=None):
    logical = profile["total"]["mean_per_decode_step"]["logical_weight_bytes_known_sum"]
    if logical != int(logical):
        # Variable routes/shapes may give a fractional mean. Rounding down makes
        # this a weaker sensitivity assumption; exact per-step values remain saved.
        scenario_volume = math.floor(logical)
    else:
        scenario_volume = int(logical)
    report = {"schema_version": 1, "profile": profile,
              "observed_physical_feasibility": {
                  "unique_required_bytes_verified_by_this_helper": False,
                  "physical_ssd_or_nand_bytes_measured_by_this_helper": False,
                  "unconditional_minimum_ssd_bytes_from_these_inputs": 0,
                  "unconditional_throughput_ceiling_from_these_inputs": None,
                  "reason": "logical sums alone do not establish unique requirements, retained coverage or a storage-rate cap"},
              "logical_volume_sensitivity_only": conditional_scenarios(scenario_volume,
                  bandwidth_bytes_per_second=bandwidth_bytes_per_second),
              "twenty_percent_speedup": {"speedup_factor": 1.2,
                  "necessary_fraction_of_original_critical_path_removed": critical_path_reduction(1.2),
                  "diagnostic_seconds_per_token_to_remove": profile["total"]["mean_per_decode_step"]["op_wall_seconds"]*critical_path_reduction(1.2),
                  "scope": "arithmetic latency budget, not proof that a given cache removes this much time"},
              "q2_reference_correction": "PR19 0.551 ms is GPU serial32 command time; native CPU10 is 0.664 ms, not 0.551 ms"}
    if source_union is not None:
        report["closed_source_operand_proof"] = closed_source_bounds(source_union,
            bandwidth_bytes_per_second=bandwidth_bytes_per_second)
        report["observed_physical_feasibility"]["distinct_declared_operands_verified_by_supplied_closed_proof"] = True
    return report


def notes_markdown(report):
    p = report["profile"]
    total = p["total"]["mean_per_decode_step"]["op_wall_seconds"]
    rows = []
    labels = {"dense_weight_matmul": "Produits de poids denses", "expert_weight_matmul": "Produits des experts sélectionnés",
              "other_or_unattributed_operations": "Autres opérations / non attribuées"}
    for key, label in labels.items():
        group = p["categories"].get(key)
        if group:
            m = group["mean_per_decode_step"]
            rows.append(f"| {label} | {m['op_wall_seconds']:.6f} | {100*group['op_wall_fraction_of_serialized_decode']:.2f} % | {m['logical_weight_bytes_known_sum']/1e9:.6f} |")
    budget_rows = []
    closed = report.get("closed_source_operand_proof")
    model = closed["conditional_storage_model"] if closed else report["logical_volume_sensitivity_only"]
    for row in model["cache_scenarios"]:
        bound = row["maximum_io_only_tokens_per_second"]
        budget_rows.append(f"| {row['retained_original_byte_coverage']/GIB:g} | {row['minimum_interface_bytes_per_token']/1e9:.6f} | {bound:.6f} |" if bound is not None else
                           f"| {row['retained_original_byte_coverage']/GIB:g} | 0 | Non contraignant |")
    contributions = {}
    for key in labels:
        g = p["categories"].get(key)
        contributions[key] = (f"{g['mean_per_decode_step']['op_wall_seconds']:.3f} s/étape ; "
                              f"{100*g['op_wall_fraction_of_serialized_decode']:.2f} %") if g else "Non présent"
    rate = model["cache_scenarios"][0]["interface_rate_bytes_per_second"]/1e9
    target_rows = [f"| {row['tokens_per_second']:g} | {row['minimum_retained_original_byte_coverage']/1e9:.6f} | "
                   f"{row['minimum_retained_original_byte_coverage']/GIB:.6f} |" for row in model["target_requirements"]]
    premise = (f"Le plan source d'une capture complète distincte établit entre {closed['decode_unique_operand_bytes_range'][0]:,} et "
               f"{closed['decode_unique_operand_bytes_range'][1]:,} octets de poids distincts déclarés par étape de décodage. "
               "L'union valide les noms, codecs, formes, strides, limites et routes canoniques du graphe capturé. Elle n'établit ni des transferts SSD ni un minimum pour tous les algorithmes possibles. "
               "Le tableau utilise le minimum U de ces étapes. Le profil temporel ci-dessus reste une observation séparée.") if closed else (
               "Les nombres suivants supposent que le volume unique requis U égale la somme logique du diagnostic. Cette hypothèse doit être vérifiée par union des plages sources ; elle ne constitue pas une mesure de lectures physiques.")
    return "\n".join([
        "# Notes quantitatives de faisabilité — métadonnées sauvegardées", "",
        "Ce document est généré par `feasibility.py`, sans accès aux poids ni inférence. Les données d'entrée sont des diagnostics sérialisés et ne constituent pas un benchmark de production.", "",
        f"{p['decode_steps']} étapes de décodage ; {total:.6f} s de temps mur d'opérations par étape, hors inspection.", "",
        "| Groupe intercalé dans les couches | Secondes/étape | Part du diagnostic | Go logiques connus/étape |",
        "|---|---:|---:|---:|", *rows, "",
        "```mermaid", "flowchart LR",
        '  T["Token précédent et états"] --> L["45 couches séquentielles\nProduits denses et experts intercalés"]',
        f'  L --> O["Logits puis prochain token\nTotal diagnostic : {total:.3f} s/étape"]',
        f'  D["Poids denses\n{contributions["dense_weight_matmul"]}"] -. "contribution mesurée" .-> L',
        f'  M["Experts sélectionnés\n{contributions["expert_weight_matmul"]}"] -. "contribution mesurée" .-> L',
        f'  R["Autres opérations\n{contributions["other_or_unattributed_operations"]}"] -. "contribution mesurée" .-> L',
        "```", "",
        "Le tableau partitionne le temps mur de ce diagnostic ; il ne sépare pas l'I/O et le calcul. Les temps CPU additionnent les threads. Les soustraire du temps mur n'a pas de sens physique. Les opérations sont intercalées, et les captures peuvent perturber l'ordonnancement.", "",
        f"## Sensibilité conditionnelle pour une interface à {rate:g} Go/s", "",
        premise, "",
        "Si U octets distincts sont nécessaires pour chaque token et si toutes les strates de cache couvrent ensemble au plus C de ces octets au début du token, au moins max(U−C,0) octets doivent traverser l'interface de stockage. Le débit ne peut alors dépasser B/max(U−C,0), même avec calcul et chevauchement idéaux. B doit être un plafond de service établi séparément. C est une couverture en octets d'origine, incluant les données compressées et caches auxiliaires, pas un RSS interchangeable.", "",
        "| Couverture retenue C (Gio) | Minimum conditionnel à transférer (Go/token) | Plafond I/O seul conditionnel (tok/s) |",
        "|---|---:|---:|", *budget_rows, "",
        "| Cible de débit (tok/s) | Couverture retenue nécessaire au minimum (Go) | Même couverture (Gio) |",
        "|---|---:|---:|", *target_rows, "",
        "Ces couvertures sont nécessaires sous les hypothèses B/U indiquées, avant coût CPU/GPU, latences, synchronisations et réserve mémoire de sécurité. Elles ne constituent pas des allocations autorisées ou démontrées faisables.", "",
        "Aucune borne de débit inconditionnelle n'est établie par les seules sommes logiques. Les lectures OS, les transferts du contrôleur et les lectures NAND restent des quantités distinctes. Les budgets du tableau sont des scénarios ; leur faisabilité mémoire n'est pas présumée.", "",
        f"Un gain de 20 % exige de retirer au moins {100*critical_path_reduction(1.2):.3f} % du chemin critique initial, soit {report['twenty_percent_speedup']['diagnostic_seconds_per_token_to_remove']:.6f} s/étape dans ce diagnostic. Un cache ne retire ce temps que s'il évite des accès coûteux réellement situés sur ce chemin.", "",
        "Correction de la mission : 0,551 ms correspond au **GPU serial32**, pas au CPU série. La référence native CPU10 mesurée sur ce groupe est 0,664 ms ; les temps GPU command-buffer et mur MLX ont des périmètres différents.", "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--bandwidth-gb-s", type=float, default=1.)
    parser.add_argument("--source-union", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or (args.notes and args.notes.exists()):
        raise ValueError("refusing to overwrite prior evidence")
    b = _nonnegative(args.bandwidth_gb_s, "bandwidth-gb-s", positive=True)*1e9
    digest = hashlib.sha256()
    with args.operations.open("rb") as stream:
        def rows():
            for line in stream:
                digest.update(line)
                if line.strip():
                    yield json.loads(line)
        profile = summarize_operations(rows())
    source_union = None
    if args.source_union:
        if args.source_union.stat().st_size > 16*1024**2:
            raise ValueError("source-union report exceeds parsing budget")
        source_union = json.loads(args.source_union.read_text())
    report = build_report(profile, bandwidth_bytes_per_second=b, source_union=source_union)
    if args.source_union:
        report["closed_source_operand_proof"]["source_report"] = {"file": args.source_union.name,
            "sha256": hashlib.sha256(args.source_union.read_bytes()).hexdigest()}
    report["source"] = {"file": args.operations.name, "sha256": digest.hexdigest(), "bytes": args.operations.stat().st_size}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.notes:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(notes_markdown(report))
    print(json.dumps({"decode_steps": profile["decode_steps"], "diagnostic_seconds_per_step":
                      profile["total"]["mean_per_decode_step"]["op_wall_seconds"], "physical_bytes_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
