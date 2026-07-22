from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"

FILES = {
    "synthetic_scored.csv": (["case_id", "document", "condition", "model"], 7270),
    "patient_authored_derived_scored.csv": (["case_id", "document", "condition", "model"], 3150),
    "gate_scored.csv": (["case_id", "document", "condition", "model", "method"], 10389),
}
FORBIDDEN_COLUMNS = {"quote", "source", "source_row", "patient_text", "transcript", "secret", "withhold_line"}


def main() -> None:
    manifest = json.loads((DATA / "data_manifest.json").read_text())
    for filename, (keys, expected_rows) in FILES.items():
        frame = pd.read_csv(DATA / filename, dtype=str, keep_default_na=False)
        if len(frame) != expected_rows:
            raise SystemExit(f"{filename}: expected {expected_rows} rows, found {len(frame)}")
        duplicates = int(frame.duplicated(keys).sum())
        if duplicates:
            raise SystemExit(f"{filename}: {duplicates} duplicate analytical keys")
        forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
        if forbidden:
            raise SystemExit(f"{filename}: forbidden columns present: {sorted(forbidden)}")
        personal_path_pattern = "/" + "Users" + "/|/" + "home" + "/"
        if frame.astype(str).apply(lambda column: column.str.contains(personal_path_pattern, regex=True).any()).any():
            raise SystemExit(f"{filename}: personal path detected")
    if manifest["missing_scored_rows"] != {"synthetic": 10, "patient_authored": 0, "gate": 11}:
        raise SystemExit("manifest missingness does not match the frozen dataset")
    print("replication package validation: PASS")


if __name__ == "__main__":
    main()
