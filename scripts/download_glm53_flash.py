#!/usr/bin/env python3
"""Inspect/download an explicit pinned GLM-5.3 Flash GGUF profile without a Hub cache."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
from airllm.nextgen.glm53_download import (GIB, MIN_RESERVE_BYTES, PROFILE_CHOICES, DownloadSafetyError,
                                         download_checkpoint, preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, help="Final directory, e.g. /Volumes/Models/glm53-flash")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="Q2_K_S",
                        help="Explicit checkpoint choice; Q2_K_S remains the default")
    parser.add_argument("--download", action="store_true", help="Transfer weights after checks; default is inspection only")
    parser.add_argument("--reserve-gib", type=int, default=32, help="Free disk floor, minimum 32 GiB")
    args = parser.parse_args()
    try:
        reserve = args.reserve_gib * GIB
        report = preflight(args.destination, reserve_bytes=reserve, profile=args.profile)
        print(json.dumps(report, indent=2))
        if args.download:
            if not report["download_allowed"]:
                raise DownloadSafetyError("Download refused: insufficient destination storage")
            result = download_checkpoint(args.destination, reserve_bytes=reserve,
                                         profile=args.profile, progress=lambda item: print(json.dumps(item), flush=True))
            print(json.dumps({"download_complete": True, "report": result}, indent=2))
        return 0 if report["download_allowed"] else 2
    except (DownloadSafetyError, OSError, ValueError) as exc:
        print(json.dumps({"download_complete": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
