from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confidemd.cases import load_cases
from confidemd.manifest import CONDITIONS, build_gate_plan, build_generation_manifest, load_model_config


METHODS = ("naive", "privacy-aware", "ci-reasoning", "generate-then-redact", "gate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provider-neutral ConfideMD generation requests.")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "synthetic" / "cases.json")
    parser.add_argument("--models", type=Path, default=ROOT / "data" / "models.json")
    parser.add_argument("--model", action="append", dest="selected_models")
    parser.add_argument("--method", action="append", choices=METHODS, dest="selected_methods")
    parser.add_argument("--condition", action="append", choices=CONDITIONS, dest="conditions")
    parser.add_argument(
        "--document",
        action="append",
        dest="document_labels",
        help="keep documents with this type:disposition label, for example referral:required",
    )
    parser.add_argument("--trace", action="store_true", help="workflow only: store every stage")
    parser.add_argument("--output-floor", type=int, default=0, help="workflow only: minimum output limit per stage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_model_config(args.models)
    models = args.selected_models or config["generation_models"]
    methods = args.selected_methods or ["naive"]
    labels = set(args.document_labels) if args.document_labels else None
    cases = load_cases(args.cases)
    if "gate" in methods:
        if len(methods) > 1:
            parser.error("--method gate builds workflow bundles and cannot be combined with other methods")
        records = build_gate_plan(
            cases,
            args.selected_models or config["gate_models"],
            conditions=args.conditions,
            document_labels=labels,
            trace=args.trace,
            output_floor=args.output_floor,
        )
        unit = "workflow bundles"
    else:
        records = build_generation_manifest(
            cases, models, methods, conditions=args.conditions, document_labels=labels
        )
        unit = "requests"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
    print(f"wrote {len(records)} {unit} to {args.output}")


if __name__ == "__main__":
    main()
