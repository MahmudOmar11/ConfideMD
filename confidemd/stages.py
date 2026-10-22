"""First workflow stage at which a required fact was not carried forward (Supplementary Table S20).

Two record types exist. The June 2026 experiment saved one record per document without stage
outputs, so an empty extraction reply cannot be told apart from a ledger that named no fact, and
an empty recipient-need reply is recognized by its missing mechanism text. The October 2026 runs
stored every stage (`gate_bundle(..., trace=True)`).
"""

from __future__ import annotations

from typing import Any


STAGES = (
    "extraction_no_output",
    "extraction_no_fact",
    "recipient_need_no_output",
    "not_managing",
    "recipient_need_suppress",
    "drafting_fallback",
    "passed_to_drafting",
)

STAGE_LABELS = {
    "extraction_no_output": "Extraction returned no output",
    "extraction_no_fact": "Extraction named no sensitive fact",
    "recipient_need_no_output": "Recipient-need step returned no output",
    "not_managing": "Specialty judged not to manage the intervention (hard rule)",
    "recipient_need_suppress": "Recipient-need step chose suppress",
    "drafting_fallback": "Drafting failed twice (generic template)",
    "passed_to_drafting": "Fact passed to drafting step",
}


def first_lost_stage_saved(record: dict[str, Any], employer_letter_record: dict[str, Any]) -> str:
    """Classify a required referral from the June saved records.

    `employer_letter_record` is the saved record of the employer letter written in the same bundle;
    its suppress list is empty only when the bundle's ledger held no sensitive term or marker.
    """
    judgment = record["judgment"]
    if not employer_letter_record["suppress_terms"]:
        return "extraction_no_output"
    if record["mode"] == "SUPPRESS" and judgment.get("mechanism", "") == "":
        return "recipient_need_no_output"
    if record["mode"] == "SUPPRESS" and not judgment["manages_intervention"]:
        return "not_managing"
    if record["mode"] == "SUPPRESS":
        return "recipient_need_suppress"
    if record.get("fallback"):
        return "drafting_fallback"
    return "passed_to_drafting"


def first_lost_stage_logged(record: dict[str, Any]) -> str:
    """Classify a required referral from a record written with `trace=True`."""
    stage_log = record.get("trace", {})
    ledger = stage_log.get("ledger", {})
    verdict = stage_log.get("judgment_raw")
    if not ledger.get("secret_label") and not ledger.get("visit_purpose") and not ledger.get("clinical_facts_safe"):
        return "extraction_no_output"
    if not ledger.get("secret_label"):
        return "extraction_no_fact"
    if not verdict:
        return "recipient_need_no_output"
    if record["mode"] == "SUPPRESS" and verdict.get("manages_intervention") is False:
        return "not_managing"
    if record["mode"] == "SUPPRESS":
        return "recipient_need_suppress"
    if record["fallback"]:
        return "drafting_fallback"
    return "passed_to_drafting"
