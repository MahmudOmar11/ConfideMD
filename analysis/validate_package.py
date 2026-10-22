from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"

FILES = {
    DATA / "synthetic_scored.csv": (["case_id", "document", "condition", "model"], 7270),
    DATA / "patient_authored_derived_scored.csv": (["case_id", "document", "condition", "model"], 3150),
    DATA / "gate_scored.csv": (["case_id", "document", "condition", "model", "method"], 10389),
    DATA / "october_runs_scored.csv": (["run", "case_id", "document", "condition", "model", "method"], 2208),
    DATA / "workflow_stage_records.csv": (["run", "case_id", "document", "condition", "model"], 480),
    DATA / "unscored_outputs.csv": (["table", "case_id", "document", "condition", "model", "method"], 21),
    DATA / "clinician_weights.csv": (["cell", "group"], 4),
    ROOT / "data" / "withholding_requests.csv": (["case_id"], 165),
    ROOT / "data" / "model_versions.csv": (["model"], 11),
}
FORBIDDEN_COLUMNS = {"quote", "source", "source_row", "patient_text", "transcript", "secret", "withhold_line"}


def main() -> None:
    manifest = json.loads((DATA / "data_manifest.json").read_text())
    for path, (keys, expected_rows) in FILES.items():
        name = path.relative_to(ROOT)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(frame) != expected_rows:
            raise SystemExit(f"{name}: expected {expected_rows} rows, found {len(frame)}")
        duplicates = int(frame.duplicated(keys).sum())
        if duplicates:
            raise SystemExit(f"{name}: {duplicates} duplicate analytical keys")
        forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
        if forbidden:
            raise SystemExit(f"{name}: forbidden columns present: {sorted(forbidden)}")
        personal_path_pattern = "/" + "Users" + "/|/" + "home" + "/"
        if frame.astype(str).apply(lambda column: column.str.contains(personal_path_pattern, regex=True).any()).any():
            raise SystemExit(f"{name}: personal path detected")
    if manifest["missing_scored_rows"] != {"synthetic": 10, "patient_authored": 0, "gate": 11}:
        raise SystemExit("manifest missingness does not match the frozen dataset")
    october = manifest["october_runs"]
    if october["scored_rows"] != october["expected_rows"] or any(october["missing_scored_rows"].values()):
        raise SystemExit("manifest October-run counts do not match the frozen dataset")
    scored = pd.read_csv(DATA / "october_runs_scored.csv")
    if scored.run.value_counts().to_dict() != october["scored_rows"]:
        raise SystemExit("October-run rows do not match the manifest")
    print("replication package validation: PASS")


if __name__ == "__main__":
    main()
