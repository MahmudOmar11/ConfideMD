from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from confidemd.cases import build_conditions, load_cases
from confidemd.manifest import build_gate_plan, build_generation_manifest, load_model_config


ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL = ROOT / "results" / "additional"


def pct(value: float) -> float:
    return round(100 * float(value), 1)


class AdditionalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = {path.name: pd.read_csv(path) for path in sorted(ADDITIONAL.glob("*.csv"))}

    def test_reference_outputs_match_fresh_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "run_additional_analyses.py"), "--output-dir", directory],
                check=True,
            )
            observed_names = sorted(path.name for path in Path(directory).glob("*.csv"))
            self.assertEqual(observed_names, sorted(self.tables))
            for name, expected in self.tables.items():
                observed = pd.read_csv(Path(directory) / name)
                assert_frame_equal(observed, expected, check_exact=True)

    def test_table_s18_request_positions(self) -> None:
        frame = self.tables["request_positions.csv"].set_index("case_set")
        self.assertEqual(frame.loc["synthetic"].tolist(), [120, 0, 120, 0.74, 11, 24])
        self.assertEqual(frame.loc["patient_authored"].tolist(), [45, 45, 0, 1.0, 3, 19])

    def test_table_s19_rerun_variation(self) -> None:
        frame = self.tables["rerun_variation.csv"]
        expected = {
            ("meta-llama-llama-3.3-70b-instruct", 0): (212, 30, 14.2, 14, 16, 25.9, 26.9, 0.9, -4.1, 6.1),
            ("meta-llama-llama-3.3-70b-instruct", 1): (212, 33, 15.6, 25, 8, 25.9, 17.9, -8.0, -15.7, -0.5),
            ("deepseek-deepseek-v4-flash", 0): (204, 39, 19.1, 32, 7, 40.7, 28.4, -12.3, -19.6, -5.0),
            ("deepseek-deepseek-v4-flash", 1): (212, 59, 27.8, 33, 26, 42.0, 38.7, -3.3, -10.4, 3.8),
            ("both", 0): (416, 69, 16.6, 46, 23, 33.2, 27.6, -5.5, -10.0, -1.2),
            ("both", 1): (424, 92, 21.7, 58, 34, 34.0, 28.3, -5.7, -11.5, -0.2),
        }
        comparisons = sorted(frame.comparison.unique())
        for (model, index), values in expected.items():
            row = frame[(frame.model == model) & (frame.comparison == comparisons[index])].iloc[0]
            observed = (
                int(row.pairs), int(row.discordant), pct(row.discordance), int(row.one_to_zero), int(row.zero_to_one),
                pct(row.rate_1), pct(row.rate_2), pct(row.net_change), pct(row.net_lower), pct(row.net_upper),
            )
            self.assertEqual(observed, values, (model, comparisons[index]))

    def test_table_s20_stage_trace(self) -> None:
        frame = self.tables["workflow_stage_trace.csv"]
        columns = {
            ("main", "meta-llama-llama-3.3-70b-instruct"): ["2/2", "0/0", "2/2", "2/2", "1/1", "4/4", "0/69", "11/80"],
            ("main", "deepseek-deepseek-v4-flash"): ["26/26", "0/0", "22/22", "1/1", "4/4", "0/0", "0/27", "53/80"],
            ("C", "openai-gpt-5.4-mini"): ["18/18", "2/2", "6/6", "8/8", "4/4", "0/0", "0/42", "38/80"],
            ("B", "meta-llama-llama-3.3-70b-instruct"): ["0/0", "2/2", "0/0", "2/2", "2/2", "5/5", "1/69", "12/80"],
            ("B", "deepseek-deepseek-v4-flash"): ["58/58", "2/2", "9/9", "1/1", "0/0", "3/3", "0/7", "73/80"],
            ("D", "deepseek-deepseek-v4-flash"): ["8/8", "2/2", "3/3", "3/3", "2/2", "0/0", "0/62", "18/80"],
        }
        for (run, model), cells in columns.items():
            column = frame[(frame.run == run) & (frame.model == model)]
            observed = [f"{row.harmful_omissions}/{row.required_referrals}" for row in column.itertuples()]
            self.assertEqual(observed, cells, (run, model))
        counts = dict(zip(self.tables["summary_counts.csv"].quantity, self.tables["summary_counts.csv"].value))
        self.assertEqual(counts["workflow_omission_agreement_june_vs_october:meta-llama-llama-3.3-70b-instruct"], 75)
        self.assertEqual(counts["workflow_omission_agreement_june_vs_october:deepseek-deepseek-v4-flash"], 50)

    def test_table_s21_workflow_base_models(self) -> None:
        frame = self.tables["workflow_base_models.csv"]
        expected = {
            ("main", "meta-llama-llama-3.3-70b-instruct", "disclosure"): (21.7, 1.2, -20.5, -27.2, -13.8, 424),
            ("main", "meta-llama-llama-3.3-70b-instruct", "omission"): (13.9, 12.7, -1.3, -8.8, 6.3, 79),
            ("main", "deepseek-deepseek-v4-flash", "disclosure"): (40.3, 1.7, -38.7, -45.3, -31.4, 424),
            ("main", "deepseek-deepseek-v4-flash", "omission"): (6.2, 66.2, 60.0, 42.5, 77.5, 80),
            ("C", "openai-gpt-5.4-mini", "disclosure"): (25.8, 0.0, -25.8, -32.1, -18.9, 423),
            ("C", "openai-gpt-5.4-mini", "omission"): (3.8, 47.5, 43.8, 21.2, 67.5, 80),
            ("D", "deepseek-deepseek-v4-flash", "disclosure"): (40.3, 0.7, -39.6, -46.3, -32.4, 424),
            ("D", "deepseek-deepseek-v4-flash", "omission"): (6.2, 22.5, 16.2, 5.0, 28.7, 80),
        }
        for (run, model, outcome), values in expected.items():
            row = frame[(frame.run == run) & (frame.model == model) & (frame.outcome == outcome)].iloc[0]
            observed = (
                pct(row.unguarded), pct(row.workflow), pct(row.risk_difference), pct(row.lower), pct(row.upper),
                int(row.pairs),
            )
            self.assertEqual(observed, values, (run, model, outcome))

    def test_table_s22_heldout_cases(self) -> None:
        frame = self.tables["workflow_heldout.csv"]
        expected = {
            ("meta-llama-llama-3.3-70b-instruct", "piloting"): ((35, 140, 1, 27), (1, 140, 3, 28)),
            ("meta-llama-llama-3.3-70b-instruct", "held_out"): ((57, 284, 10, 52), (4, 284, 8, 52)),
            ("deepseek-deepseek-v4-flash", "piloting"): ((61, 140, 1, 28), (3, 140, 15, 28)),
            ("deepseek-deepseek-v4-flash", "held_out"): ((110, 284, 4, 52), (4, 284, 38, 52)),
        }
        for (model, cases), values in expected.items():
            for method, counts in zip(("naive", "gate"), values):
                row = frame[(frame.model == model) & (frame.cases == cases) & (frame.method == method)].iloc[0]
                observed = (row.disclosures, row.disclosure_documents, row.omissions, row.required_referrals)
                self.assertEqual(tuple(int(value) for value in observed), counts, (model, cases, method))
        paired = self.tables["workflow_heldout_paired.csv"]
        effects = {
            ("meta-llama-llama-3.3-70b-instruct", "disclosure"): (-18.7, -26.4, -10.8),
            ("meta-llama-llama-3.3-70b-instruct", "omission"): (-3.8, -12.5, 4.2),
            ("deepseek-deepseek-v4-flash", "disclosure"): (-37.3, -45.3, -28.9),
            ("deepseek-deepseek-v4-flash", "omission"): (65.4, 42.3, 85.4),
        }
        for (model, outcome), values in effects.items():
            row = paired[(paired.model == model) & (paired.outcome == outcome)].iloc[0]
            self.assertEqual((pct(row.risk_difference), pct(row.lower), pct(row.upper)), values)

    def test_table_s12_deepseek_exclusion(self) -> None:
        frame = self.tables["deepseek_exclusion.csv"].set_index("models")
        for models, values in {
            "all_seven": (37.7, 34.2, -3.4, -7.0, 0.3, 1481, 153, 102),
            "excluding_deepseek": (37.0, 33.5, -3.5, -7.1, 0.2, 1269, 120, 76),
        }.items():
            row = frame.loc[models]
            observed = (
                pct(row.off), pct(row.on), pct(row.risk_difference), pct(row.lower), pct(row.upper),
                int(row.pairs), int(row.improved), int(row.worsened),
            )
            self.assertEqual(observed, values)

    def test_table_s13_scorer_sensitivity(self) -> None:
        frame = self.tables["scorer_sensitivity.csv"]

        def cell(dataset: str, endpoint: str, condition: str, scorer: str) -> str:
            row = frame[
                (frame.dataset == dataset) & (frame.endpoint == endpoint)
                & (frame.condition == condition) & (frame.scorer == scorer)
            ].iloc[0]
            return f"{pct(row.estimate)} ({pct(row.lower)}-{pct(row.upper)})"

        primary = "primary_unambiguous"
        self.assertEqual(cell("synthetic", primary, "request_off", "raw_judge_primary"), "37.7 (29.7-45.1)")
        self.assertEqual(cell("synthetic", primary, "request_on", "lexical_sensitivity"), "32.7 (25.5-39.9)")
        self.assertEqual(cell("synthetic", primary, "request_on", "clinician_weighted_sensitivity"), "33.5 (26.5-40.1)")
        self.assertEqual(cell("patient_authored", primary, "request_off", "raw_judge_primary"), "67.6 (61.7-73.2)")
        self.assertEqual(cell("patient_authored", primary, "request_off", "lexical_sensitivity"), "44.8 (36.7-52.7)")
        self.assertEqual(
            cell("patient_authored", primary, "request_on", "clinician_weighted_sensitivity"), "26.1 (21.1-31.5)"
        )
        self.assertEqual(
            cell("patient_authored", "summary_sensitive_fact_inclusion_all", "request_off", "clinician_weighted_sensitivity"),
            "85.8 (79.7-91.3)",
        )
        self.assertEqual(
            cell("patient_authored", "all_legacy_forbidden", "request_on", "lexical_sensitivity"), "29.6 (22.9-36.7)"
        )
        differences = self.tables["scorer_paired_differences.csv"].set_index(["dataset", "scorer"])
        self.assertEqual(pct(differences.loc[("patient_authored", "adjudicator")].risk_difference), -33.8)
        self.assertEqual(pct(differences.loc[("patient_authored", "lexical")].risk_difference), -24.9)
        self.assertEqual(pct(differences.loc[("synthetic", "lexical")].risk_difference), -4.3)

    def test_scored_output_counts(self) -> None:
        counts = dict(zip(self.tables["summary_counts.csv"].quantity, self.tables["summary_counts.csv"].value))
        self.assertEqual(counts["scored_outputs"], 20809)
        self.assertEqual(counts["distinct_scored_documents"], 18732)
        self.assertEqual(counts["unscored_outputs"], 21)
        self.assertEqual(counts["rerun_textually_identical_documents"], 2)
        self.assertEqual(counts["rerun_empty_primary_documents_excluded:deepseek-deepseek-v4-flash"], 8)
        self.assertEqual((counts["rerun_required_omissions_run_1"], counts["rerun_required_omissions_run_2"]), (4, 4))


class AdditionalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(ROOT / "data" / "synthetic" / "cases.json")
        cls.requests = pd.read_csv(ROOT / "data" / "withholding_requests.csv", keep_default_na=False)
        cls.october = pd.read_csv(ROOT / "data" / "derived" / "october_runs_scored.csv")

    def test_requests_cover_both_case_sets(self) -> None:
        self.assertEqual(self.requests.case_set.value_counts().to_dict(), {"synthetic": 120, "patient_authored": 45})
        self.assertEqual(self.requests.case_id.nunique(), 165)

    def test_synthetic_request_lines_and_positions_come_from_the_transcripts(self) -> None:
        rows = self.requests[self.requests.case_set == "synthetic"].set_index("case_id")
        for case in self.cases:
            row = rows.loc[case["id"]]
            line = f"{row.speaker_label}: {row.request_text}"
            self.assertEqual(line, case["withhold_line"])
            lines = [value for value in case["transcript_on"].split("\n") if value.strip()]
            self.assertEqual(lines.count(line), 1)
            index = lines.index(line)
            self.assertEqual((row.line_index, row.dialogue_lines), (index + 1, len(lines)))
            reply = lines[index + 1][4:].strip() if lines[index + 1].startswith("Dr:") else ""
            self.assertEqual(row.clinician_reply, reply)
            off_lines = build_conditions(case)["request_off"].splitlines()
            self.assertNotIn(line, off_lines)

    def test_patient_authored_rows_carry_only_the_request_and_its_position(self) -> None:
        rows = self.requests[self.requests.case_set == "patient_authored"]
        scored = pd.read_csv(ROOT / "data" / "derived" / "patient_authored_derived_scored.csv")
        self.assertEqual(set(rows.case_id), set(scored.case_id))
        self.assertEqual(dict(zip(rows.case_id, rows.category)), dict(zip(scored.case_id, scored.category)))
        self.assertTrue(rows.position.eq("final line").all())
        self.assertTrue(rows.clinician_reply.eq("").all())
        self.assertTrue(rows.speaker_label.eq("Patient request").all())
        self.assertTrue(rows.dialogue_lines.eq(3).all())
        self.assertFalse(rows.request_text.str.contains("\n").any())
        self.assertLessEqual(int(rows.request_text.str.len().max()), 300)

    def test_model_versions_cover_every_model_in_the_code_and_data(self) -> None:
        versions = pd.read_csv(ROOT / "data" / "model_versions.csv", keep_default_na=False)
        identifiers = set(versions.request_identifier) - {""}
        config = load_model_config(ROOT / "data" / "models.json")
        self.assertLessEqual(set(config["generation_models"]) | {config["adjudication_model"]}, identifiers)
        runs = json.loads((ROOT / "data" / "additional_runs.json").read_text())
        self.assertLessEqual({model for run in runs["runs"] for model in run["models"]}, identifiers)
        self.assertIn("anthropic/claude-haiku-4.5", identifiers)
        self.assertIn("openai/gpt-4o-mini", identifiers)
        evaluated = versions[versions.roles.str.contains("Evaluated model")]
        self.assertEqual(len(evaluated), 7)
        self.assertEqual(evaluated.reasoning_control.value_counts().to_dict(), {"yes": 4, "no": 2, "unknown": 1})
        self.assertTrue(evaluated.access_dates.str.startswith("2026-06-18 to 2026-06-22").all())
        screening = versions[versions.roles.str.contains("Patient-authored screening")]
        self.assertEqual(len(screening), 2)
        self.assertTrue(screening.access_dates.eq("2026-06-21 (screening of patient-authored cases)").all())

    def test_run_specifications_reproduce_the_october_documents(self) -> None:
        runs = json.loads((ROOT / "data" / "additional_runs.json").read_text())["runs"]
        for run in runs:
            labels = set(run["document_labels"]) if run["document_labels"] else None
            if run["method"] == "naive":
                planned = {
                    (request["case_id"], request["document"], request["condition"], request["model"].replace("/", "-"))
                    for request in build_generation_manifest(
                        self.cases, run["models"], ["naive"], conditions=run["conditions"], document_labels=labels
                    )
                }
            else:
                planned = {
                    (bundle["case_id"], document["key"], bundle["condition"], bundle["model"].replace("/", "-"))
                    for bundle in build_gate_plan(
                        self.cases, run["models"], conditions=run["conditions"], document_labels=labels,
                        trace=run["trace"], output_floor=run["output_floor"],
                    )
                    for document in bundle["case_view"]["documents"]
                }
            scored = self.october[self.october.run == run["run"]]
            observed = set(zip(scored.case_id, scored.document, scored.condition, scored.model))
            self.assertEqual(len(planned), run["documents"], run["run"])
            self.assertEqual(observed, planned, run["run"])

    def test_unscored_outputs_are_exactly_the_missing_scored_rows(self) -> None:
        unscored = pd.read_csv(ROOT / "data" / "derived" / "unscored_outputs.csv")
        config = load_model_config(ROOT / "data" / "models.json")
        designs = {
            "synthetic_scored": (config["generation_models"], ["naive"]),
            "gate_scored": (config["gate_models"], ["naive", "privacy-aware", "ci-reasoning", "generate-then-redact", "gate"]),
        }
        for table, (models, methods) in designs.items():
            expected = {
                (case["id"], document["key"], condition, model.replace("/", "-"), method)
                for case in self.cases
                for document in case["documents"]
                for condition in ("request_off", "request_on")
                for model in models
                for method in methods
            }
            frame = pd.read_csv(ROOT / "data" / "derived" / f"{table}.csv")
            scored_methods = frame.method if "method" in frame else ["naive"] * len(frame)
            scored = set(zip(frame.case_id, frame.document, frame.condition, frame.model, scored_methods))
            listed = unscored[unscored.table == table]
            missing = set(zip(listed.case_id, listed.document, listed.condition, listed.model, listed.method))
            self.assertEqual(expected - scored, missing, table)

    def test_stage_records_cover_every_required_referral_of_each_run(self) -> None:
        stages = pd.read_csv(ROOT / "data" / "derived" / "workflow_stage_records.csv")
        required = {
            (case["id"], document["key"])
            for case in self.cases
            for document in case["documents"]
            if case["disposition"][document["key"]]["secret_expected"] == "required"
        }
        self.assertTrue(set(zip(stages.case_id, stages.document)) <= required)
        self.assertEqual(stages.run.value_counts().to_dict(), {"main": 160, "B": 160, "C": 80, "D": 80})
        self.assertEqual(int(stages.duplicated(["run", "case_id", "document", "condition", "model"]).sum()), 0)


if __name__ == "__main__":
    unittest.main()
