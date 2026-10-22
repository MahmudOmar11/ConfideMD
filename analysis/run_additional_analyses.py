"""Reproduce the additional analyses in the Supplementary Information (Tables S12, S13, and S18 to S22).

Inputs are the sanitized tables in data/; outputs go to results/additional/. Bootstrap labels match
the original analysis so that every interval reproduces the supplement exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived"
RESULTS = ROOT / "results" / "additional"
BOOTSTRAP_REPS = 10_000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from confidemd.stages import STAGE_LABELS, STAGES  # noqa: E402
import run_analysis as main_analysis  # noqa: E402

GATE_MODELS = ["deepseek-deepseek-v4-flash", "meta-llama-llama-3.3-70b-instruct"]
FLAGS = (
    "raw_leak",
    "harmful_omission",
    "term_positive",
    "unambiguous_forbidden",
    "explicit_summary_forbidden",
    "conservative_forbidden",
    "legacy_clinical_forbidden",
)
# The October runs keep the names they had in the original analysis inside the bootstrap labels.
RUN_LABELS = {"B": "gate_trace", "C": "gate_m3", "D": "gate_fix"}
COMPARISONS = (
    "identical runs (no request, run 1 vs run 2)",
    "request effect (no request vs request, run 1)",
)


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, label: str) -> tuple[float, float]:
    """Percentile interval for a ratio of cluster sums, resampling clusters with replacement."""
    if len(numerator) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed(label))
    draw = rng.integers(0, len(numerator), size=(BOOTSTRAP_REPS, len(numerator)))
    boot_numerator = numerator[draw].sum(axis=1)
    boot_denominator = denominator[draw].sum(axis=1)
    values = np.divide(
        boot_numerator, boot_denominator, out=np.full(BOOTSTRAP_REPS, np.nan), where=boot_denominator != 0
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def clustered_rate(frame: pd.DataFrame, outcome: str, cluster: str, label: str) -> dict:
    if frame.empty:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan, "n": 0, "clusters": 0}
    aggregate = frame.groupby(cluster, sort=True)[outcome].agg(["sum", "count"])
    lower, upper = bootstrap_ratio(aggregate["sum"].to_numpy(float), aggregate["count"].to_numpy(float), label)
    return {
        "estimate": float(frame[outcome].sum() / len(frame)),
        "lower": lower,
        "upper": upper,
        "n": int(len(frame)),
        "clusters": int(len(aggregate)),
    }


def paired_difference(first: pd.Series, second: pd.Series, clusters: pd.Series, label: str) -> tuple[float, float]:
    """Interval for the mean of second minus first, clustered."""
    aggregate = pd.DataFrame({"d": second.to_numpy(float) - first.to_numpy(float), "c": clusters.to_numpy()})
    aggregate = aggregate.groupby("c", sort=True).d.agg(["sum", "count"])
    return bootstrap_ratio(aggregate["sum"].to_numpy(float), aggregate["count"].to_numpy(float), label)


def paired_condition_difference(frame: pd.DataFrame, outcome: str, cluster: str, label: str) -> dict:
    keys = ["case_id", "document", "model"]
    off = frame[frame.condition == "request_off"][list(dict.fromkeys(keys + [cluster, outcome]))]
    on = frame[frame.condition == "request_on"][keys + [outcome]]
    paired = off.rename(columns={outcome: "off"}).merge(
        on.rename(columns={outcome: "on"}), on=keys, how="inner", validate="one_to_one"
    )
    lower, upper = paired_difference(paired.off, paired.on, paired[cluster], label)
    return {
        "off": float(paired.off.mean()),
        "on": float(paired.on.mean()),
        "risk_difference": float((paired.on - paired.off).mean()),
        "lower": lower,
        "upper": upper,
        "pairs": int(len(paired)),
        "clusters": int(paired[cluster].nunique()),
        "improved": int(((paired.off == 1) & (paired.on == 0)).sum()),
        "worsened": int(((paired.off == 0) & (paired.on == 1)).sum()),
    }


def paired_workflow_difference(frame: pd.DataFrame, outcome: str, label: str) -> dict:
    keys = ["case_id", "document", "condition", "model"]
    naive = frame[frame.method == "naive"][keys + ["template_id", outcome]].rename(columns={outcome: "naive"})
    workflow = frame[frame.method == "gate"][keys + [outcome]].rename(columns={outcome: "workflow"})
    paired = naive.merge(workflow, on=keys, how="inner", validate="one_to_one")
    lower, upper = paired_difference(paired.naive, paired.workflow, paired.template_id, label)
    return {
        "naive": float(paired.naive.mean()),
        "method_rate": float(paired.workflow.mean()),
        "risk_difference": float((paired.workflow - paired.naive).mean()),
        "lower": lower,
        "upper": upper,
        "pairs": int(len(paired)),
        "clusters": int(paired.template_id.nunique()),
    }


def load_scored(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in FLAGS:
        frame[column] = frame[column].astype(str).str.lower().isin({"true", "1", "1.0"}).astype(float)
    return frame


def attach_clinician_weight(frame: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Score each row by the physicians' consensus in its scorer-agreement cell.

    Rows both scorers call present score 1 and rows both call absent score 0. Discordant rows take
    the share of physician-confirmed disclosures among consensus rows of the same cell, separately
    for intimate partner violence and other domains.
    """
    share = {
        (row.cell, row.group): row.clinician_positive / row.consensus_rows for row in weights.itertuples()
    }
    group = np.where(frame.category == "ipv", "ipv", "non_ipv")
    judge = frame.raw_leak == 1
    term = frame.term_positive == 1
    value = np.zeros(len(frame))
    value[judge.to_numpy() & term.to_numpy()] = 1.0
    for cell, mask in (
        ("judge_positive_term_negative", (judge & ~term).to_numpy()),
        ("judge_negative_term_positive", (~judge & term).to_numpy()),
    ):
        for name in ("ipv", "non_ipv"):
            selected = mask & (group == name)
            value[selected] = share[(cell, name)]
    out = frame.copy()
    out["clinician_weighted"] = value
    return out


def scorer_sensitivity(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Table S13: disclosure under the adjudicator, the lexical detector, and the clinician weights."""
    rows = []
    for dataset, frame in datasets.items():
        cluster = "template_id" if dataset == "synthetic" else "case_id"
        seed_dataset = "synthetic" if dataset == "synthetic" else "real"
        masks = {
            "primary_unambiguous": frame.unambiguous_forbidden == 1,
            "summary_sensitive_fact_inclusion_all": frame.surface == "patient_or_proxy_summary",
            "all_legacy_forbidden": frame.disposition == "forbidden",
        }
        for endpoint, mask in masks.items():
            for condition in ("request_off", "request_on"):
                subset = frame[mask & (frame.condition == condition)]
                for outcome, scorer in (
                    ("raw_leak", "raw_judge_primary"),
                    ("term_positive", "lexical_sensitivity"),
                    ("clinician_weighted", "clinician_weighted_sensitivity"),
                ):
                    result = clustered_rate(
                        subset, outcome, cluster, f"sensitivity-{seed_dataset}-{endpoint}-{condition}-{scorer}"
                    )
                    rows.append({
                        "dataset": dataset, "endpoint": endpoint, "condition": condition, "scorer": scorer,
                        "numerator": float(subset[outcome].sum()), **result,
                    })
    return pd.DataFrame(rows)


def scorer_paired_differences(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Primary endpoint under each scorer, with the paired request effect."""
    rows = []
    for scorer, outcome in (
        ("adjudicator", "raw_leak"),
        ("clinician_weighted", "clinician_weighted"),
        ("lexical", "term_positive"),
    ):
        label_outcome = "clinician_weighted_sensitivity" if outcome == "clinician_weighted" else outcome
        for dataset, label_set in (("patient_authored", "patient-authored"), ("synthetic", "synthetic")):
            frame = datasets[dataset]
            cluster = "template_id" if dataset == "synthetic" else "case_id"
            primary = frame[frame.unambiguous_forbidden == 1]
            off = clustered_rate(primary[primary.condition == "request_off"], outcome, cluster,
                                 f"rev-scorer-{label_set}-{label_outcome}-off")
            on = clustered_rate(primary[primary.condition == "request_on"], outcome, cluster,
                                f"rev-scorer-{label_set}-{label_outcome}-on")
            effect = paired_condition_difference(primary, outcome, cluster, f"rev-scorer-{label_set}-{label_outcome}-rd")
            rows.append({
                "dataset": dataset, "scorer": scorer,
                "off": off["estimate"], "off_lower": off["lower"], "off_upper": off["upper"],
                "on": on["estimate"], "on_lower": on["lower"], "on_upper": on["upper"],
                "risk_difference": effect["risk_difference"], "rd_lower": effect["lower"], "rd_upper": effect["upper"],
                "n_off": off["n"], "n_on": on["n"], "pairs": effect["pairs"],
            })
    return pd.DataFrame(rows)


def request_positions(requests: pd.DataFrame) -> pd.DataFrame:
    """Table S18: where the withholding requests appear."""
    rows = []
    for case_set in ("synthetic", "patient_authored"):
        group = requests[requests.case_set == case_set]
        rows.append({
            "case_set": case_set,
            "cases": len(group),
            "request_is_final_line": int((group.position == "final line").sum()),
            "clinician_replies": int(group.clinician_replies.astype(str).eq("True").sum()),
            "median_relative_position": float(group.relative_position.median()),
            "median_dialogue_lines": float(group.dialogue_lines.median()),
            "median_line_words": float(group.line_words.median()),
        })
    return pd.DataFrame(rows)


def rerun_variation(synthetic: pd.DataFrame, rerun: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Table S19: identical request-absent runs compared with the request effect (primary endpoint)."""
    keys = ["case_id", "document", "model"]
    kept = rerun[~rerun.empty_output.astype(str).eq("True")]
    rows = []
    models = sorted(kept.model.unique())
    for model in models + ["both"]:
        selected = synthetic.model.isin(models) if model == "both" else synthetic.model == model
        original = synthetic[(synthetic.unambiguous_forbidden == 1) & selected]
        repeat = kept[(kept.unambiguous_forbidden == 1) & (kept.model.isin(models) if model == "both" else kept.model == model)]
        first = original[original.condition == "request_off"][keys + ["template_id", "raw_leak"]]
        pairs = {
            COMPARISONS[0]: first.merge(repeat[keys + ["raw_leak"]], on=keys, suffixes=("_1", "_2")),
            COMPARISONS[1]: first.merge(
                original[original.condition == "request_on"][keys + ["raw_leak"]], on=keys, suffixes=("_1", "_2")
            ),
        }
        for comparison, paired in pairs.items():
            discordant = (paired.raw_leak_1 != paired.raw_leak_2).astype(float)
            aggregate = pd.DataFrame({"f": discordant.to_numpy(), "c": paired.template_id.to_numpy()})
            aggregate = aggregate.groupby("c").f.agg(["sum", "count"])
            disc_lower, disc_upper = bootstrap_ratio(
                aggregate["sum"].to_numpy(float), aggregate["count"].to_numpy(float), f"runA-disc-{model}-{comparison}"
            )
            net_lower, net_upper = paired_difference(
                paired.raw_leak_1, paired.raw_leak_2, paired.template_id, f"runA-net-{model}-{comparison}"
            )
            rows.append({
                "model": model,
                "comparison": comparison,
                "pairs": len(paired),
                "discordant": int(discordant.sum()),
                "discordance": float(discordant.mean()),
                "discordance_lower": disc_lower,
                "discordance_upper": disc_upper,
                "one_to_zero": int(((paired.raw_leak_1 == 1) & (paired.raw_leak_2 == 0)).sum()),
                "zero_to_one": int(((paired.raw_leak_1 == 0) & (paired.raw_leak_2 == 1)).sum()),
                "net_change": float((paired.raw_leak_2 - paired.raw_leak_1).mean()),
                "net_lower": net_lower,
                "net_upper": net_upper,
                "rate_1": float(paired.raw_leak_1.mean()),
                "rate_2": float(paired.raw_leak_2.mean()),
            })

    empty = rerun[rerun.empty_output.astype(str).eq("True")]
    original = synthetic[(synthetic.condition == "request_off") & synthetic.model.isin(models)]
    both = original[keys + ["disposition", "raw_leak", "harmful_omission"]].merge(
        kept[keys + ["raw_leak", "harmful_omission"]], on=keys, suffixes=("_1", "_2")
    )
    forbidden = both[both.disposition == "forbidden"]
    required = both[both.disposition == "required"]
    counts = {
        "rerun_documents": len(rerun),
        "rerun_textually_identical_documents": int(rerun.identical_to_original.astype(str).eq("True").sum()),
        **{f"rerun_empty_documents_excluded:{m}": int((empty.model == m).sum()) for m in models},
        **{f"rerun_empty_primary_documents_excluded:{m}": int(((empty.model == m) & (empty.unambiguous_forbidden == 1)).sum())
           for m in models},
        "rerun_all_forbidden_pairs": len(forbidden),
        "rerun_all_forbidden_discordant": int((forbidden.raw_leak_1 != forbidden.raw_leak_2).sum()),
        "rerun_required_pairs": len(required),
        "rerun_required_omissions_run_1": int(required.harmful_omission_1.sum()),
        "rerun_required_omissions_run_2": int(required.harmful_omission_2.sum()),
        "rerun_required_discordant": int((required.harmful_omission_1 != required.harmful_omission_2).sum()),
    }
    return pd.DataFrame(rows), counts


def workflow_stage_trace(stages: pd.DataFrame, gate: pd.DataFrame, october: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Table S20: first stage at which a required fact was lost, with harmful omissions per stage."""
    keys = ["case_id", "document", "condition", "model"]
    june = gate[(gate.method == "gate") & (gate.disposition == "required")][keys + ["harmful_omission"]].assign(run="main")
    later = october[(october.run != "A") & (october.disposition == "required")][keys + ["run", "harmful_omission"]]
    outcomes = pd.concat([june, later], ignore_index=True)
    merged = stages.merge(outcomes, on=keys + ["run"], how="left", validate="one_to_one")
    if merged.harmful_omission.isna().any():
        raise SystemExit("a stage record has no scored outcome")
    rows = []
    for (run, model), group in merged.groupby(["run", "model"], sort=True):
        for stage in STAGES + ("all",):
            subset = group if stage == "all" else group[group.first_lost_stage == stage]
            rows.append({
                "run": run,
                "model": model,
                "first_lost_stage": stage,
                "description": "All required referrals" if stage == "all" else STAGE_LABELS[stage],
                "harmful_omissions": int(subset.harmful_omission.sum()),
                "required_referrals": len(subset),
            })
    agreement = june.merge(later[later.run == "B"], on=keys, suffixes=("_june", "_october"))
    counts = {
        f"workflow_omission_agreement_june_vs_october:{model}": int(
            (group.harmful_omission_june == group.harmful_omission_october).sum()
        )
        for model, group in agreement.groupby("model")
    }
    counts.update({f"workflow_omission_pairs_june_vs_october:{model}": len(group) for model, group in agreement.groupby("model")})
    return pd.DataFrame(rows), counts


def workflow_runs(synthetic: pd.DataFrame, october: pd.DataFrame, gate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S21 and per-condition rates: workflow runs against the unguarded main-evaluation documents."""
    paired_rows = []
    main_run = main_analysis.gate_paired(gate)
    for model in GATE_MODELS:
        for endpoint, outcome in (("primary_unambiguous", "disclosure"), ("required_documents", "omission")):
            row = main_run[(main_run.model == model) & (main_run.method == "gate") & (main_run.endpoint == endpoint)].iloc[0]
            paired_rows.append({
                "run": "main", "model": model, "outcome": outcome, "unguarded": row.naive, "workflow": row.method_rate,
                "workflow_events": int(round(row.method_rate * row.pairs)),
                "risk_difference": row.risk_difference, "lower": row.lower, "upper": row.upper,
                "pairs": int(row.pairs), "templates": int(row.clusters),
            })
    rate_rows = []
    keys = ["case_id", "document", "condition"]
    for run, label in RUN_LABELS.items():
        frame = october[october.run == run]
        for model in sorted(frame.model.unique()):
            workflow = frame[frame.model == model]
            for condition in ("request_off", "request_on", "both"):
                subset = workflow if condition == "both" else workflow[workflow.condition == condition]
                disclosure = subset[subset.unambiguous_forbidden == 1]
                required = subset[subset.disposition == "required"]
                a = clustered_rate(disclosure, "raw_leak", "template_id", f"{label}-{model}-{condition}-disc")
                b = clustered_rate(required, "harmful_omission", "template_id", f"{label}-{model}-{condition}-om")
                rate_rows.append({
                    "run": run, "model": model, "condition": condition,
                    "disclosures": int(disclosure.raw_leak.sum()), "disclosure_documents": a["n"],
                    "disclosure": a["estimate"], "disclosure_lower": a["lower"], "disclosure_upper": a["upper"],
                    "omissions": int(required.harmful_omission.sum()), "required_referrals": b["n"],
                    "omission": b["estimate"], "omission_lower": b["lower"], "omission_upper": b["upper"],
                })
            unguarded = synthetic[synthetic.model == model]
            for outcome, mask_workflow, mask_unguarded, name in (
                ("raw_leak", workflow.unambiguous_forbidden == 1, unguarded.unambiguous_forbidden == 1, "disclosure"),
                ("harmful_omission", workflow.disposition == "required", unguarded.disposition == "required", "omission"),
            ):
                paired = unguarded[mask_unguarded][keys + ["template_id", outcome]].merge(
                    workflow[mask_workflow][keys + [outcome]], on=keys, suffixes=("_u", "_w")
                )
                lower, upper = paired_difference(
                    paired[f"{outcome}_u"], paired[f"{outcome}_w"], paired.template_id, f"{label}-{model}-paired-{name}"
                )
                paired_rows.append({
                    "run": run, "model": model, "outcome": name,
                    "unguarded": paired[f"{outcome}_u"].mean() if len(paired) else np.nan,
                    "workflow": paired[f"{outcome}_w"].mean() if len(paired) else np.nan,
                    "workflow_events": int(paired[f"{outcome}_w"].sum()),
                    "risk_difference": (paired[f"{outcome}_w"] - paired[f"{outcome}_u"]).mean() if len(paired) else np.nan,
                    "lower": lower, "upper": upper, "pairs": len(paired), "templates": paired.template_id.nunique(),
                })
    return pd.DataFrame(paired_rows), pd.DataFrame(rate_rows)


def workflow_heldout(gate: pd.DataFrame, cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S22: the 36 piloting cases (first six per domain) against the 84 held-out cases."""
    by_domain: dict[str, list[str]] = {}
    for case in cases:
        by_domain.setdefault(case["category"], []).append(case["id"])
    pilot = {case_id for domain in sorted(by_domain) for case_id in by_domain[domain][:6]}
    frame = gate.assign(split=np.where(gate.case_id.isin(pilot), "piloting", "held_out"))
    rows = []
    for model in GATE_MODELS:
        for split in ("piloting", "held_out"):
            for method in ("naive", "gate"):
                subset = frame[(frame.model == model) & (frame.split == split) & (frame.method == method)]
                primary = subset[subset.unambiguous_forbidden == 1]
                required = subset[subset.disposition == "required"]
                rows.append({
                    "model": model, "cases": split, "method": method,
                    "disclosures": int(primary.raw_leak.sum()), "disclosure_documents": len(primary),
                    "omissions": int(required.harmful_omission.sum()), "required_referrals": len(required),
                })
    paired_rows = []
    for model in GATE_MODELS:
        heldout = frame[(frame.model == model) & (frame.split == "held_out")]
        for outcome, mask, name in (
            ("raw_leak", heldout.unambiguous_forbidden == 1, "disclosure"),
            ("harmful_omission", heldout.disposition == "required", "omission"),
        ):
            result = paired_workflow_difference(heldout[mask], outcome, f"rev-heldout-{model}-{name}")
            paired_rows.append({"model": model, "outcome": name, **result})
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def deepseek_exclusion(synthetic: pd.DataFrame) -> pd.DataFrame:
    """Table S12, recomputed with the bootstrap seed of Table S5."""
    primary = synthetic[synthetic.unambiguous_forbidden == 1]
    rows = []
    for models, frame in (
        ("all_seven", primary),
        ("excluding_deepseek", primary[primary.model != "deepseek-deepseek-v4-flash"]),
    ):
        result = paired_condition_difference(frame, "raw_leak", "template_id", "condition-synthetic-primary_unambiguous")
        rows.append({"models": models, **result})
    return pd.DataFrame(rows)


def omission_request_differences(synthetic: pd.DataFrame, cases: list[dict]) -> pd.DataFrame:
    """Harmful omission in required referrals: paired request effect by safety-tension stratum."""
    tension = {case["id"]: case.get("safety_tension") or "none" for case in cases}
    required = synthetic[synthetic.disposition == "required"].assign(safety_tension=lambda f: f.case_id.map(tension))
    rows = []
    for stratum in ("all", "resolved", "unresolved"):
        subset = required if stratum == "all" else required[required.safety_tension == stratum]
        result = paired_condition_difference(subset, "harmful_omission", "template_id", f"rev-omission-{stratum}")
        rows.append({"stratum": stratum, **result})
    return pd.DataFrame(rows)


def summary_counts(synthetic, patient, gate, unscored, rerun_counts, stage_counts) -> pd.DataFrame:
    scored = len(synthetic) + len(patient) + len(gate)
    reused = int((gate.method == "naive").sum())
    counts = {
        "scored_outputs": scored,
        "workflow_arm_unguarded_rows_scored": reused,
        "distinct_scored_documents": scored - reused,
        "unscored_outputs": len(unscored),
        **{
            f"unscored:{table}:{reason}": len(group)
            for (table, reason), group in unscored.groupby(["table", "reason"], sort=True)
        },
        **rerun_counts,
        **stage_counts,
    }
    return pd.DataFrame({"quantity": list(counts), "value": list(counts.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the additional analyses.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((DATA / "synthetic" / "cases.json").read_text())["cases"]
    weights = pd.read_csv(DERIVED / "clinician_weights.csv")
    synthetic = attach_clinician_weight(load_scored(DERIVED / "synthetic_scored.csv"), weights)
    patient = attach_clinician_weight(load_scored(DERIVED / "patient_authored_derived_scored.csv"), weights)
    gate = load_scored(DERIVED / "gate_scored.csv")
    october = load_scored(DERIVED / "october_runs_scored.csv")
    stages = pd.read_csv(DERIVED / "workflow_stage_records.csv")
    requests = pd.read_csv(DATA / "withholding_requests.csv", keep_default_na=False)
    unscored = pd.read_csv(DERIVED / "unscored_outputs.csv")
    datasets = {"synthetic": synthetic, "patient_authored": patient}

    variation, rerun_counts = rerun_variation(synthetic, october[october.run == "A"])
    trace, stage_counts = workflow_stage_trace(stages, gate, october)
    base_models, run_rates = workflow_runs(synthetic, october, gate)
    heldout, heldout_paired = workflow_heldout(gate, cases)
    outputs = {
        "scorer_sensitivity.csv": scorer_sensitivity(datasets),
        "scorer_paired_differences.csv": scorer_paired_differences(datasets),
        "request_positions.csv": request_positions(requests),
        "rerun_variation.csv": variation,
        "workflow_stage_trace.csv": trace,
        "workflow_base_models.csv": base_models,
        "workflow_run_rates.csv": run_rates,
        "workflow_heldout.csv": heldout,
        "workflow_heldout_paired.csv": heldout_paired,
        "deepseek_exclusion.csv": deepseek_exclusion(synthetic),
        "omission_request_differences.csv": omission_request_differences(synthetic, cases),
        "summary_counts.csv": summary_counts(synthetic, patient, gate, unscored, rerun_counts, stage_counts),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    print(f"wrote {len(outputs)} additional-analysis tables to {args.output_dir}")


if __name__ == "__main__":
    main()
