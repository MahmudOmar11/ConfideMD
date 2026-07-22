from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from confidemd.cases import build_conditions, gate_safe_view, load_cases
from confidemd.manifest import build_generation_manifest, load_model_config
from confidemd.prompts import build_generation_request, build_redaction_request


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "data" / "synthetic" / "cases.json"


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(CASE_PATH)

    def test_case_bank_has_120_variants_from_30_templates(self) -> None:
        self.assertEqual(len(self.cases), 120)
        self.assertEqual(len({case["template"] for case in self.cases}), 30)
        self.assertEqual(len({case["id"] for case in self.cases}), 120)

    def test_request_off_removes_only_the_withholding_line(self) -> None:
        for case in self.cases:
            conditions = build_conditions(case)
            on_lines = conditions["request_on"].splitlines()
            off_lines = conditions["request_off"].splitlines()
            self.assertEqual(len(on_lines) - len(off_lines), 1)
            self.assertNotIn(case["withhold_line"], off_lines)
            self.assertEqual(on_lines.count(case["withhold_line"]), 1)

    def test_gate_view_excludes_every_answer_key_field(self) -> None:
        forbidden = {
            "disposition",
            "marker",
            "marker_terms",
            "must_keep",
            "safety_critical",
            "safety_tension",
            "secret",
            "secret_terms",
            "template",
            "withhold_line",
            "withhold_recipient",
        }
        for case in self.cases:
            safe = gate_safe_view(case)
            self.assertFalse(forbidden & set(safe))
            self.assertTrue(all("secret_expected" not in doc for doc in safe["documents"]))

    def test_generation_request_is_provider_neutral_and_answer_key_free(self) -> None:
        case = self.cases[0]
        document = case["documents"][1]
        request = build_generation_request(
            case_id=case["id"],
            document=document,
            condition="request_on",
            transcript=build_conditions(case)["request_on"],
            model="example/model",
            method="naive",
        )
        serialized = json.dumps(request, sort_keys=True)
        self.assertEqual(
            set(request),
            {"request_id", "case_id", "document", "condition", "model", "method", "messages", "parameters"},
        )
        self.assertNotIn("secret_expected", serialized)
        self.assertNotIn("disposition", serialized)
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("provider", serialized.lower())
        self.assertEqual(request["parameters"], {"temperature": 0, "max_tokens": 3000})

    def test_all_comparator_prompts_are_available(self) -> None:
        case = self.cases[0]
        document = case["documents"][1]
        transcript = build_conditions(case)["request_on"]
        for method in ("naive", "privacy-aware", "ci-reasoning", "generate-then-redact"):
            request = build_generation_request(
                case_id=case["id"],
                document=document,
                condition="request_on",
                transcript=transcript,
                model="example/model",
                method=method,
            )
            self.assertEqual(request["method"], method)
            self.assertNotIn("secret_expected", json.dumps(request))

    def test_redaction_pass_is_provider_neutral(self) -> None:
        case = self.cases[0]
        request = build_redaction_request(
            case_id=case["id"],
            document=case["documents"][1],
            condition="request_on",
            model="example/model",
            draft="Draft referral",
        )
        self.assertEqual(request["method"], "generate-then-redact")
        self.assertNotIn("provider", json.dumps(request).lower())
        self.assertIn("Draft referral", request["messages"][1]["content"])

    def test_model_config_matches_the_seven_scored_models(self) -> None:
        config = load_model_config(ROOT / "data" / "models.json")
        self.assertEqual(len(config["generation_models"]), 7)
        scored = {
            row.split(",")[6]
            for row in (ROOT / "data" / "derived" / "synthetic_scored.csv").read_text().splitlines()[1:]
        }
        configured = {model.replace("/", "-") for model in config["generation_models"]}
        self.assertEqual(configured, scored)

    def test_manifest_has_one_unique_request_per_case_document_condition_model(self) -> None:
        manifest = build_generation_manifest(self.cases, ["example/model"], ["naive"])
        self.assertEqual(len(manifest), 1040)
        self.assertEqual(len({request["request_id"] for request in manifest}), 1040)
        self.assertNotIn("secret_expected", json.dumps(manifest))

    def test_manifest_cli_writes_jsonl_without_external_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requests.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analysis" / "build_requests.py"),
                    "--model",
                    "example/model",
                    "--method",
                    "naive",
                    "--output",
                    str(output),
                ],
                check=True,
            )
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1040)
            self.assertEqual(len({json.loads(line)["request_id"] for line in lines}), 1040)


if __name__ == "__main__":
    unittest.main()
