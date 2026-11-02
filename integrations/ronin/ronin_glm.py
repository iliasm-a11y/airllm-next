#!/usr/bin/env python3
"""Select and run the real local GLM from the RONIN runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "air_llm"))
sys.path.insert(0, str(ROOT / "runtime"))

from ronin_runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "generate", "stdio"])
    parser.add_argument("--config", type=Path, help="JSON with model_path and native runtime options")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default=None,
                        help="Explicit GGUF chat-template conditioning; omitted preserves the template default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reasoning_effort is not None and args.action != "generate":
        parser.error("--reasoning-effort applies to generate only")
    for value, env in ((args.config, "RONIN_GLM_CONFIG"), (args.model, "RONIN_GLM_MODEL"),
                       (args.runtime, "RONIN_GLM_RUNTIME"), (args.verification, "RONIN_GLM_VERIFICATION"),
                       (args.home, "RONIN_DATA_HOME")):
        if value is not None:
            os.environ[env] = str(value.expanduser().resolve())
    if args.action == "stdio":
        from ronin_runtime import main as serve
        return serve()
    runtime = Runtime(args.home or os.environ.get("RONIN_DATA_HOME"))
    result = {}
    code = 0
    try:
        if args.model:
            runtime.dispatch("settings", {"model_path": str(args.model.expanduser().resolve())})
        if args.action == "status":
            result = runtime.dispatch("status", {})
        else:
            if not args.prompt:
                parser.error("generate requires --prompt")
            mission = runtime.dispatch("create", {"prompt": args.prompt})
            params = {"id": mission["id"], "max_new_tokens": args.max_tokens}
            if args.reasoning_effort is not None: params["reasoning_effort"] = args.reasoning_effort
            result = runtime.dispatch("generate", params)
            runtime.state(mission["id"], "completed")
            result["mission_id"] = mission["id"]
            result["transport"] = "RONIN Runtime.dispatch -> Engine.stream -> NativeGlmModel.stream_chat"
    except Exception as error:
        result = {"error": {"code": getattr(error, "code", type(error).__name__), "message": str(error)}}
        code = 2
    finally:
        runtime.dispatch("shutdown", {})
        runtime.store.db.close()
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
