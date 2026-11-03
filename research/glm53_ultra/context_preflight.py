"""Offline context sizing using the checkpoint's companion tokenizer/template.

The native /apply-template and /tokenize checks remain authoritative. This
tool loads no weights and makes no inference or equivalence claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "air_llm"), str(REPO / "integrations/ronin/runtime"),
                str(REPO / "integrations/ronin/engine")]


def main():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    from ronin_runtime import Runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols"])
    template = env.from_string((args.model / "chat_template.jinja").read_text())
    report = {"scope": "offline companion tokenizer/template preflight; no native server or inference",
              "files": {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                        for name in ("tokenizer.json", "chat_template.jinja")},
              "raw_reference_tokens": tokenizer.encode("La capitale de la France est", add_special_tokens=False).ids,
              "agents": []}
    with tempfile.TemporaryDirectory() as temporary:
        runtime = Runtime(temporary)
        try:
            for agent in (1, 600):
                mission = runtime.create({"prompt": "Réponds uniquement par OK."})
                runtime.agent_contexts.memory(agent, mission["id"], "Repère de travail.")
                decision = runtime.agent_contexts.begin(agent, mission["id"], mission["prompt"],
                                                       {"scope": "offline preflight"})
                rendered = template.render(messages=decision["messages"], add_generation_prompt=True,
                                           reasoning_effort="low")
                ids = tokenizer.encode(rendered, add_special_tokens=False).ids
                report["agents"].append({"agent_id": agent,
                    "messages_bytes": len(json.dumps(decision["messages"], ensure_ascii=False).encode()),
                    "rendered_prompt_tokens": len(ids),
                    "fits_context512_output32": len(ids) + 32 <= 512,
                    "fits_context1024_output32": len(ids) + 32 <= 1024})
                runtime.agent_contexts.finish(decision["id"], "failed", provenance={"scope": "offline only"},
                                              error_code="PREFLIGHT_ONLY")
        finally:
            if runtime.engine is not None:
                runtime.engine.close()
            runtime.store.db.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
