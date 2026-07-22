from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confidemd.cases import load_cases
from confidemd.manifest import build_generation_manifest, load_model_config


METHODS = ("naive", "privacy-aware", "ci-reasoning", "generate-then-redact")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provider-neutral ConfideMD generation requests.")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "synthetic" / "cases.json")
    parser.add_argument("--models", type=Path, default=ROOT / "data" / "models.json")
    parser.add_argument("--model", action="append", dest="selected_models")
    parser.add_argument("--method", action="append", choices=METHODS, dest="selected_methods")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_model_config(args.models)
    models = args.selected_models or config["generation_models"]
    methods = args.selected_methods or ["naive"]
    manifest = build_generation_manifest(load_cases(args.cases), models, methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(request, separators=(",", ":")) + "\n" for request in manifest))
    print(f"wrote {len(manifest)} requests to {args.output}")


if __name__ == "__main__":
    main()
