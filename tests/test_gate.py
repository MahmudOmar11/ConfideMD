from __future__ import annotations

import unittest

from confidemd.gate import OutputLimitFloor, gate_bundle, per_document_constraints, route_document, safety_judgment
from confidemd.refusal import fallback_template
from confidemd.stages import first_lost_stage_logged, first_lost_stage_saved


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict:
        self.calls.append(("json", system, user))
        if "clinical privacy analyst" in system:
            return {
                "secret_label": "example condition",
                "secret_surface_forms": ["example condition"],
                "marker_items": [{"name": "ExampleDrug", "implies": "example condition"}],
                "withhold_request": {"present": True, "from_whom": "family"},
                "visit_purpose": "planned procedure",
                "secret_is_visit_purpose": False,
                "clinical_facts_safe": ["planned procedure", "ExampleDrug affects anaesthesia"],
            }
        if "clinical-privacy gatekeeper" in system:
            return {
                "manages_intervention": True,
                "diagnosis": "KEEP",
                "drug": "KEEP",
                "mechanism": "drug interaction",
            }
        if "clinical safety auditor" in system:
            return {"missing": []}
        if "clinical-privacy auditor" in system:
            return {"leak": False, "what": "", "span": ""}
        raise AssertionError("unexpected JSON stage")

    def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str:
        self.calls.append(("text", system, user))
        return "Recipient-specific document"


class GateTests(unittest.TestCase):
    def test_route_document_uses_recipient_type(self) -> None:
        self.assertEqual(route_document({"type": "note"}), "INCLUDE")
        self.assertEqual(route_document({"type": "patient_summary"}), "SUPPRESS")
        self.assertEqual(route_document({"type": "employer_letter"}), "SUPPRESS")
        self.assertEqual(route_document({"type": "referral"}), "JUDGE")

    def test_nonmanaging_specialty_forces_suppression(self) -> None:
        class NonManagingClient(RecordingClient):
            def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict:
                if "clinical-privacy gatekeeper" in system:
                    return {
                        "manages_intervention": False,
                        "diagnosis": "KEEP",
                        "drug": "KEEP",
                        "mechanism": "none",
                    }
                return super().json(system, user, reasoning=reasoning, max_tokens=max_tokens)

        decision = safety_judgment(
            NonManagingClient(),
            "Dermatology",
            {
                "secret_label": "example condition",
                "marker_items": [{"name": "ExampleDrug"}],
                "visit_purpose": "skin lesion",
            },
        )
        self.assertEqual((decision["diagnosis"], decision["drug"]), ("SUPPRESS", "SUPPRESS"))

    def test_suppression_constraints_remove_secret_entailing_facts(self) -> None:
        mode, keep_facts, keep_terms, suppress_terms, contentless = per_document_constraints(
            {"type": "employer_letter"},
            "SUPPRESS",
            {
                "secret_label": "example condition",
                "secret_surface_forms": ["example condition"],
                "marker_items": [{"name": "ExampleDrug"}],
                "clinical_facts_safe": ["fit for work", "example condition follow-up"],
                "secret_is_visit_purpose": False,
            },
            {},
        )
        self.assertEqual(mode, "SUPPRESS")
        self.assertEqual(keep_facts, ["fit for work"])
        self.assertEqual(keep_terms, [])
        self.assertEqual(suppress_terms, ["example condition", "ExampleDrug"])
        self.assertFalse(contentless)

    def test_gate_bundle_extracts_once_and_never_receives_answer_keys(self) -> None:
        client = RecordingClient()
        case_view = {
            "id": "case-001",
            "documents": [
                {"key": "note", "type": "note"},
                {"key": "referral-anesthesiology", "type": "referral", "specialty": "Anesthesiology"},
                {"key": "referral-dermatology", "type": "referral", "specialty": "Dermatology"},
                {"key": "patient-summary", "type": "patient_summary"},
                {"key": "employer-letter", "type": "employer_letter"},
            ],
        }
        result = gate_bundle(case_view, "request_on", "Synthetic transcript", client)
        self.assertEqual(set(result), {doc["key"] for doc in case_view["documents"]})
        extraction_calls = [call for call in client.calls if "clinical privacy analyst" in call[1]]
        self.assertEqual(len(extraction_calls), 1)
        serialized_calls = repr(client.calls).lower()
        self.assertNotIn("secret_expected", serialized_calls)
        self.assertNotIn("disposition", serialized_calls)

    def test_gate_reuses_a_specialty_judgment_within_a_bundle(self) -> None:
        client = RecordingClient()
        case_view = {
            "id": "case-001",
            "documents": [
                {"key": "referral-a", "type": "referral", "specialty": "Anesthesiology"},
                {"key": "referral-b", "type": "referral", "specialty": "Anesthesiology"},
            ],
        }
        gate_bundle(case_view, "request_on", "Synthetic transcript", client)
        judgments = [call for call in client.calls if "clinical-privacy gatekeeper" in call[1]]
        self.assertEqual(len(judgments), 1)

    def test_empty_replies_fall_back_to_suppression_without_raising(self) -> None:
        class EmptyClient(RecordingClient):
            def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str:
                self.calls.append(("json", system, user))
                return ""

        case_view = {
            "id": "case-001",
            "documents": [
                {"key": "referral-anesthesiology", "type": "referral", "specialty": "Anesthesiology"},
                {"key": "employer-letter", "type": "employer_letter"},
            ],
        }
        result = gate_bundle(case_view, "request_on", "Synthetic transcript", EmptyClient(), trace=True)
        text, record = result["referral-anesthesiology"]
        self.assertEqual(record["mode"], "SUPPRESS")
        self.assertEqual((record["judgment"]["diagnosis"], record["judgment"]["drug"]), ("SUPPRESS", "SUPPRESS"))
        self.assertEqual(record["trace"]["judgment_raw"], {})
        self.assertEqual(first_lost_stage_logged(record), "extraction_no_output")
        self.assertEqual(first_lost_stage_saved(record, result["employer-letter"][1]), "extraction_no_output")

    def test_refused_draft_is_retried_once_then_replaced(self) -> None:
        class RefusingClient(RecordingClient):
            def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str:
                self.calls.append(("text", system, user))
                return "I can't help with that."

        client = RefusingClient()
        document = {"key": "employer-letter", "type": "employer_letter"}
        result = gate_bundle({"id": "case-001", "documents": [document]}, "request_off", "Synthetic transcript", client)
        text, record = result["employer-letter"]
        self.assertEqual(text, fallback_template(document))
        self.assertTrue(record["refused"] and record["fallback"])
        self.assertEqual(sum(1 for call in client.calls if call[0] == "text"), 2)
        self.assertFalse(any("clinical-privacy auditor" in call[1] for call in client.calls))

    def test_trace_stores_every_stage_and_records_the_kept_fact(self) -> None:
        case_view = {"id": "case-001", "documents": [
            {"key": "referral-anesthesiology", "type": "referral", "specialty": "Anesthesiology"}
        ]}
        untraced = gate_bundle(case_view, "request_on", "Synthetic transcript", RecordingClient())
        self.assertNotIn("trace", untraced["referral-anesthesiology"][1])
        record = gate_bundle(case_view, "request_on", "Synthetic transcript", RecordingClient(), trace=True)[
            "referral-anesthesiology"
        ][1]
        self.assertEqual(set(record["trace"]), {"ledger", "judgment_raw", "keep_facts", "draft", "verify", "repaired"})
        self.assertEqual(record["trace"]["verify"], {"kind": "keep", "missing": []})
        self.assertNotIn("_raw", record["judgment"])
        self.assertEqual(first_lost_stage_logged(record), "passed_to_drafting")

    def test_output_limit_floor_raises_every_stage_limit(self) -> None:
        class LimitClient(RecordingClient):
            def __init__(self) -> None:
                super().__init__()
                self.limits: list[int] = []

            def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict:
                self.limits.append(max_tokens)
                return super().json(system, user, reasoning=reasoning, max_tokens=max_tokens)

            def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str:
                self.limits.append(max_tokens)
                return super().text(system, user, reasoning=reasoning, max_tokens=max_tokens)

        case_view = {"id": "case-001", "documents": [
            {"key": "referral-anesthesiology", "type": "referral", "specialty": "Anesthesiology"},
            {"key": "employer-letter", "type": "employer_letter"},
        ]}
        specified = LimitClient()
        gate_bundle(case_view, "request_on", "Synthetic transcript", specified)
        self.assertEqual(sorted(set(specified.limits)), [250, 300, 700, 800])
        raised = LimitClient()
        gate_bundle(case_view, "request_on", "Synthetic transcript", OutputLimitFloor(raised, 4000))
        self.assertEqual(set(raised.limits), {4000})

    def test_saved_record_stages_follow_the_june_rules(self) -> None:
        employer = {"suppress_terms": ["example condition"]}
        judged = {"diagnosis": "SUPPRESS", "drug": "SUPPRESS", "mechanism": "none", "manages_intervention": False}
        self.assertEqual(first_lost_stage_saved({"mode": "SUPPRESS", "judgment": judged}, employer), "not_managing")
        empty = dict(judged, mechanism="")
        self.assertEqual(first_lost_stage_saved({"mode": "SUPPRESS", "judgment": empty}, employer), "recipient_need_no_output")
        kept = dict(judged, manages_intervention=True)
        self.assertEqual(first_lost_stage_saved({"mode": "SUPPRESS", "judgment": kept}, employer), "recipient_need_suppress")
        included = {"mode": "INCLUDE", "judgment": kept, "fallback": True}
        self.assertEqual(first_lost_stage_saved(included, employer), "drafting_fallback")
        self.assertEqual(first_lost_stage_saved({**included, "fallback": False}, employer), "passed_to_drafting")
        self.assertEqual(first_lost_stage_saved(included, {"suppress_terms": []}), "extraction_no_output")


if __name__ == "__main__":
    unittest.main()
