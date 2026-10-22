from __future__ import annotations

import re
from typing import Any


REFUSAL_PATTERN = re.compile(
    r"\b(i (?:can(?:'|no)?t|am unable to|won'?t)|i'?m (?:not able|unable)|"
    r"i (?:must|have to) (?:decline|refuse)|as an ai|i cannot create|i cannot provide|"
    r"i'?m sorry,? but i (?:can'?t|cannot))\b",
    re.I,
)

FALLBACK_DOCUMENTS = {
    "note": (
        "Clinical Encounter Note\n\nThe patient was seen today for evaluation. "
        "History, examination, assessment, and plan were documented and discussed "
        "with the patient. Follow-up arranged as clinically indicated."
    ),
    "referral": (
        "Referral Letter\n\nDear Colleague,\n\nThank you for seeing this patient, "
        "who is referred to your service for further evaluation and management as "
        "discussed. Relevant clinical details are available in the medical record. "
        "Please contact the referring clinic with any questions.\n\nKind regards."
    ),
    "patient_summary": (
        "After-Visit Summary\n\nYou were seen in clinic today. We discussed "
        "your care plan together. A detailed note is in your medical record. "
        "Please contact the clinic if you have any questions or need follow-up details."
    ),
    "employer_letter": (
        "Medical Certificate\n\nTo Whom It May Concern,\n\nThis confirms the "
        "above patient attended a medical appointment on the stated date. Any "
        "necessary work adjustments or time off have been discussed with the "
        "patient. No further clinical information is disclosed.\n\nYours faithfully."
    ),
}


def is_refusal(text: str | None) -> bool:
    """Flag an empty output, or a short output that matches a refusal phrase."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    return len(stripped) < 400 and bool(REFUSAL_PATTERN.search(stripped))


def fallback_template(document: dict[str, Any]) -> str:
    """Neutral document used in place of a refusal by the comparators and the workflow."""
    return FALLBACK_DOCUMENTS.get(document.get("type", ""), FALLBACK_DOCUMENTS["note"])
