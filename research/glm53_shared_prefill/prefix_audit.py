"""Audit complete RONIN prompts with the companion tokenizer, without weights.

The LCP is computed from token IDs of complete rendered prompts, never by
tokenizing a character prefix in isolation. Native confirmation is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def common_prefix(sequences):
    """Return exact longest prefix; empty collections have no established LCP."""
    sequences = list(sequences)
    if not sequences:
        return []
    first = sequences[0]
    end = len(first)
    for current in sequences[1:]:
        end = min(end, len(current))
        for position in range(end):
            if first[position] != current[position]:
                end = position
                break
    return first[:end]


def token_summary(agents, *, microbatch=8):
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("positive microbatch required")
    ids = common_prefix(agent["token_ids"] for agent in agents)
    text = "".join(common_prefix(agent["rendered_prompt"] for agent in agents))
    systems = "".join(common_prefix(agent["messages"][0]["content"] for agent in agents))
    return {
        "agent_ids": [agent["agent_id"] for agent in agents],
        "common_token_count": len(ids), "common_token_ids": ids,
        "first_divergent_ids": {str(a["agent_id"]):
            a["token_ids"][len(ids):len(ids) + 4] for a in agents},
        "rendered_text_lcp_characters": len(text),
        "rendered_text_lcp_utf8_bytes": len(text.encode()),
        "rendered_text_lcp_sha256": sha256(text.encode()),
        "system_text_lcp_characters": len(systems),
        "system_text_lcp_utf8_bytes": len(systems.encode()),
        "system_text_lcp_sha256": sha256(systems.encode()),
        "microbatch_tokens": microbatch,
        "largest_nonzero_full_microbatch_boundary_inside_lcp":
            (len(ids) // microbatch) * microbatch or None,
        "batch_boundary_note": "Arithmetic candidate only; split/unsplit KDA numerical equivalence unqualified.",
    }


def render_agents(agents, tokenizer, template):
    for agent in agents:
        rendered = template.render(messages=agent["messages"], add_generation_prompt=True,
                                   reasoning_effort="low")
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
        agent.update(rendered_prompt=rendered, token_ids=encoding.ids,
                     rendered_utf8_sha256=sha256(rendered.encode()),
                     prompt_token_count=len(encoding.ids))
    return agents


def historical_messages(database):
    """Only explicit isolated probe DB, opened read-only with no journal writes."""
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT agent_id,mission_id,messages,context_sha256,state FROM "
            "glm_agent_decisions WHERE agent_id IN (1,600) ORDER BY sequence").fetchall()
    return [{"agent_id": row["agent_id"], "mission_id": row["mission_id"],
             "messages": json.loads(row["messages"]),
             "context_sha256": row["context_sha256"], "recorded_state": row["state"]}
            for row in rows]


def generated_messages(agent_ids, tasks):
    """Use actual RONIN construction, with a temporary store and no Engine init."""
    sys.path.insert(0, str(REPO / "integrations/ronin/runtime"))
    from ronin_runtime import Runtime, Store, IDENTITY, ROLES, redact
    from agent_context import AgentContexts
    agents = []
    with tempfile.TemporaryDirectory() as temporary:
        # Runtime.create uses the real Store, but no Runtime.__init__ / Engine.
        runtime = Runtime.__new__(Runtime)
        runtime.store = Store(temporary)
        contexts = AgentContexts(runtime.store, identity=IDENTITY, roles=ROLES, sanitize=redact)
        try:
            for agent_id in agent_ids:
                task = tasks.get(agent_id, {
                    "prompt": "Calcule 17 × 24, puis compare le résultat au stock de 420 pièces. Explique le calcul et rappelle le repère de ta mémoire.",
                    "marker": f"RONIN-AUDIT-{agent_id:03d}",
                })
                marker = task["marker"]
                memory = task.get("memory", f"Repère privé de cette mission : {marker}. Ce repère appartient uniquement à cette identité et à cette mission.")
                mission = runtime.create({"prompt": task["prompt"]})
                contexts.memory(agent_id, mission["id"], memory)
                decision = contexts.begin(agent_id, mission["id"], mission["prompt"],
                                          {"scope": "offline exact prompt audit; no generation"})
                agents.append({"agent_id": agent_id, "mission_id": mission["id"],
                               "messages": decision["messages"], "marker": marker,
                               "context_sha256": decision["context_sha256"]})
                contexts.finish(decision["id"], "failed",
                                provenance={"scope": "offline audit; no model output"},
                                error_code="OFFLINE_AUDIT_ONLY")
        finally:
            runtime.store.db.close()
    return agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-db", type=Path)
    parser.add_argument("--tasks", required=True, type=Path,
                        help="JSON array: agent_id, prompt, marker and optional memory")
    parser.add_argument("--all-agents", action="store_true")
    parser.add_argument("--microbatch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    import tokenizers
    import jinja2
    # An explicit task artifact aligns this audit with a chosen campaign without
    # constructing its Engine or importing execution code.
    task_data = json.loads(args.tasks.read_text())
    if (not isinstance(task_data, list) or any(not isinstance(task, dict) or
            type(task.get("agent_id")) is not int or not 1 <= task["agent_id"] <= 600 or
            not isinstance(task.get("prompt"), str) or not task["prompt"].strip() or
            not isinstance(task.get("marker"), str) or not task["marker"] for task in task_data)):
        raise ValueError("explicit structured tasks required")
    tasks = {task["agent_id"]: task for task in task_data}
    if len(tasks) != len(task_data):
        raise ValueError("duplicate agent tasks")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=["jinja2.ext.loopcontrols"])
    template = environment.from_string((args.model / "chat_template.jinja").read_text())
    generated = render_agents(generated_messages(range(1, 601) if args.all_agents else (1, 600),
                                               tasks), tokenizer, template)
    pair = [agent for agent in generated if agent["agent_id"] in (1, 600)]
    report = {
        "schema_version": 1,
        "scope": "Companion tokenizer/template on complete real RONIN-constructed messages; no weights, native server, prefill, cache or inference.",
        "native_confirmed": False,
        "options": {"reasoning_effort": "low", "add_generation_prompt": True,
                    "add_special_tokens": False, "tools": [], "history": "empty first turns; no trimming"},
        "versions": {"tokenizers": tokenizers.__version__, "jinja2": jinja2.__version__},
        "files": {name: sha256((args.model / name).read_bytes()) for name in
                  ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json")},
        "tasks_sha256": sha256(args.tasks.read_bytes()),
        "sources": {name: sha256((REPO / name).read_bytes()) for name in
                    ("integrations/ronin/runtime/agent_context.py", "integrations/ronin/runtime/ronin_runtime.py")},
        "agents": generated,
        "all_generated": token_summary(generated, microbatch=args.microbatch),
        "pair_1_600": token_summary(pair, microbatch=args.microbatch),
    }
    if args.probe_db:
        historical = render_agents(historical_messages(args.probe_db), tokenizer, template)
        report["historical_probe"] = {"scope": "saved decisions from explicitly supplied isolated probe opened read-only",
                                      "agents": historical,
                                      "summary": token_summary(historical, microbatch=args.microbatch)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = report["all_generated"]
    print(json.dumps({"output": str(args.output), "agents": len(generated),
                      "common_tokens": summary["common_token_count"],
                      "system_prefix_utf8_bytes": summary["system_text_lcp_utf8_bytes"],
                      "batch_boundary": summary["largest_nonzero_full_microbatch_boundary_inside_lcp"],
                      "native_confirmed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
