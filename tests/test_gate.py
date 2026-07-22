from __future__ import annotations

import unittest

from confidemd.gate import gate_bundle, per_document_constraints, route_document, safety_judgment


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


if __name__ == "__main__":
    unittest.main()
