from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
BOOTSTRAP_REPS = 10_000


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def load_table(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / name)
    for column in (
        "raw_leak",
        "harmful_omission",
        "term_positive",
        "unambiguous_forbidden",
        "explicit_summary_forbidden",
        "conservative_forbidden",
        "legacy_clinical_forbidden",
    ):
        frame[column] = frame[column].astype(str).str.lower().isin({"true", "1", "1.0"}).astype(float)
    return frame


def bootstrap_mean_by_cluster(frame: pd.DataFrame, outcome: str, cluster: str, label: str) -> tuple[float, float]:
    aggregate = frame.groupby(cluster)[outcome].agg(["sum", "count"])
    rng = np.random.default_rng(stable_seed(label))
    draw = rng.integers(0, len(aggregate), size=(BOOTSTRAP_REPS, len(aggregate)))
    numerator = aggregate["sum"].to_numpy(float)[draw].sum(axis=1)
    denominator = aggregate["count"].to_numpy(float)[draw].sum(axis=1)
    values = numerator / denominator
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def rate(frame: pd.DataFrame, outcome: str, cluster: str, label: str, metadata: dict) -> dict:
    lower, upper = bootstrap_mean_by_cluster(frame, outcome, cluster, label)
    return {
        **metadata,
        "numerator": int(frame[outcome].sum()),
        "n": len(frame),
        "clusters": frame[cluster].nunique(),
        "estimate": float(frame[outcome].mean()),
        "lower": lower,
        "upper": upper,
    }


def endpoint_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "primary_unambiguous": frame.unambiguous_forbidden == 1,
        "conservative_explicit_recipient": frame.conservative_forbidden == 1,
        "summary_sensitive_fact_inclusion_all": frame.surface == "patient_or_proxy_summary",
        "summary_explicit_recipient_leak": frame.explicit_summary_forbidden == 1,
        "all_legacy_forbidden": frame.disposition == "forbidden",
    }


def endpoint_rates(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in datasets.items():
        cluster = "template_id" if dataset == "synthetic" else "case_id"
        seed_dataset = "synthetic" if dataset == "synthetic" else "real"
        for endpoint, mask in endpoint_masks(frame).items():
            for condition in ("request_off", "request_on"):
                subset = frame[mask & (frame.condition == condition)]
                rows.append(rate(subset, "raw_leak", cluster, f"endpoint-{seed_dataset}-{endpoint}-{condition}", {
                    "dataset": dataset, "endpoint": endpoint, "condition": condition,
                }))
    return pd.DataFrame(rows)


def paired_condition(frame: pd.DataFrame, dataset: str, endpoint: str) -> dict:
    cluster = "template_id" if dataset == "synthetic" else "case_id"
    seed_dataset = "synthetic" if dataset == "synthetic" else "real"
    subset = frame[endpoint_masks(frame)[endpoint]]
    paired = subset.pivot_table(
        index=["case_id", "template_id", "document", "model"],
        columns="condition",
        values="raw_leak",
        aggfunc="first",
    ).dropna().reset_index()
    paired["difference"] = paired.request_on - paired.request_off
    lower, upper = bootstrap_mean_by_cluster(paired, "difference", cluster, f"condition-{seed_dataset}-{endpoint}")
    return {
        "dataset": dataset,
        "endpoint": endpoint,
        "request_off": paired.request_off.mean(),
        "request_on": paired.request_on.mean(),
        "risk_difference": paired.difference.mean(),
        "lower": lower,
        "upper": upper,
        "pairs": len(paired),
        "clusters": paired[cluster].nunique(),
    }


def model_rates(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in datasets.items():
        cluster = "template_id" if dataset == "synthetic" else "case_id"
        seed_dataset = "synthetic" if dataset == "synthetic" else "real"
        base = frame[(frame.unambiguous_forbidden == 1) & (frame.condition == "request_on")]
        for model in sorted(base.model.unique()):
            subset = base[base.model == model]
            rows.append(rate(subset, "raw_leak", cluster, f"model-{seed_dataset}-{model}", {"dataset": dataset, "model": model}))
    return pd.DataFrame(rows)


def category_rates(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in datasets.items():
        cluster = "template_id" if dataset == "synthetic" else "case_id"
        seed_dataset = "synthetic" if dataset == "synthetic" else "real"
        base = frame[(frame.unambiguous_forbidden == 1) & (frame.condition == "request_on")]
        for category in sorted(base.category.unique()):
            subset = base[base.category == category]
            rows.append(rate(subset, "raw_leak", cluster, f"category-{seed_dataset}-{category}", {"dataset": dataset, "category": category}))
    return pd.DataFrame(rows)


def omission_rates(stage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = stage[stage.disposition == "required"]
    for condition in ("request_off", "request_on"):
        subset = required[required.condition == condition]
        rows.append(rate(subset, "harmful_omission", "template_id", f"omission-{condition}-all-all-models", {"condition": condition}))
    return pd.DataFrame(rows)


def gate_rates(gate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(gate.model.unique()):
        for method in sorted(gate.method.unique()):
            for condition in ("request_off", "request_on"):
                cell = gate[(gate.model == model) & (gate.method == method) & (gate.condition == condition)]
                for endpoint, mask, outcome in (
                    ("primary_unambiguous", cell.unambiguous_forbidden == 1, "raw_leak"),
                    ("all_legacy_forbidden", cell.disposition == "forbidden", "raw_leak"),
                    ("required_documents", cell.disposition == "required", "harmful_omission"),
                ):
                    subset = cell[mask]
                    rows.append(rate(subset, outcome, "template_id", f"gate-rate-{model}-{method}-{condition}-{endpoint if endpoint != 'required_documents' else 'omission'}", {
                        "model": model, "method": method, "condition": condition, "endpoint": endpoint,
                    }))
    return pd.DataFrame(rows)


def gate_paired(gate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(gate.model.unique()):
        model_frame = gate[gate.model == model]
        for method in sorted(set(model_frame.method) - {"naive"}):
            for endpoint, mask, outcome in (
                ("primary_unambiguous", model_frame.unambiguous_forbidden == 1, "raw_leak"),
                ("all_legacy_forbidden", model_frame.disposition == "forbidden", "raw_leak"),
                ("required_documents", model_frame.disposition == "required", "harmful_omission"),
            ):
                subset = model_frame[mask]
                wide = subset.pivot_table(
                    index=["case_id", "template_id", "document", "condition", "model"],
                    columns="method",
                    values=outcome,
                    aggfunc="first",
                ).dropna(subset=["naive", method]).reset_index()
                wide["difference"] = wide[method] - wide.naive
                lower, upper = bootstrap_mean_by_cluster(wide, "difference", "template_id", f"gate-paired-{model}-{method}-{endpoint}")
                rows.append({
                    "model": model,
                    "method": method,
                    "endpoint": endpoint,
                    "naive": wide.naive.mean(),
                    "method_rate": wide[method].mean(),
                    "risk_difference": wide.difference.mean(),
                    "lower": lower,
                    "upper": upper,
                    "pairs": len(wide),
                    "clusters": wide.template_id.nunique(),
                    "improved": int(((wide.naive == 1) & (wide[method] == 0)).sum()),
                    "worsened": int(((wide.naive == 0) & (wide[method] == 1)).sum()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce ConfideMD analysis tables.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = load_table("synthetic_scored.csv")
    patient_authored = load_table("patient_authored_derived_scored.csv")
    gate = load_table("gate_scored.csv")
    datasets = {"synthetic": synthetic, "patient_authored": patient_authored}

    outputs = {
        "endpoint_rates.csv": endpoint_rates(datasets),
        "condition_effects.csv": pd.DataFrame([
            paired_condition(frame, dataset, endpoint)
            for dataset, frame in datasets.items()
            for endpoint in endpoint_masks(frame)
        ]),
        "model_rates.csv": model_rates(datasets),
        "category_rates.csv": category_rates(datasets),
        "omission_rates.csv": omission_rates(synthetic),
        "gate_rates.csv": gate_rates(gate),
        "gate_paired_vs_naive.csv": gate_paired(gate),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    print(f"wrote {len(outputs)} result tables to {output_dir}")


if __name__ == "__main__":
    main()
