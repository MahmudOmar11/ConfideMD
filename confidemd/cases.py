from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ANSWER_KEY_FIELDS = {
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


def _normalize_line(value: str) -> str:
    value = value.replace("“", '"').replace("”", '"').replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().lower()


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    cases = payload.get("cases")
    if not isinstance(cases, list) or payload.get("n") != len(cases):
        raise ValueError("case file count does not match its cases array")
    identifiers = [case.get("id") for case in cases]
    if None in identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("case identifiers must be present and unique")
    return cases


def build_conditions(case: dict[str, Any]) -> dict[str, str]:
    transcript = str(case["transcript_on"])
    target = _normalize_line(str(case["withhold_line"]))
    lines = transcript.splitlines()
    matches = [index for index, line in enumerate(lines) if _normalize_line(line) == target]
    if len(matches) != 1:
        raise ValueError(f"expected one withholding line, found {len(matches)}")
    request_off = "\n".join(line for index, line in enumerate(lines) if index != matches[0])
    return {"request_on": transcript, "request_off": request_off}


def document_label(case: dict[str, Any], document: dict[str, Any]) -> str:
    """Label a document by type and expected disposition, for example `referral:required`."""
    return f"{document['type']}:{case['disposition'][document['key']]['secret_expected']}"


def select_documents(case: dict[str, Any], labels: set[str] | None = None) -> list[dict[str, Any]]:
    """Return the case documents whose label is in `labels`, or every document when no labels are given."""
    if not labels:
        return list(case["documents"])
    return [document for document in case["documents"] if document_label(case, document) in labels]


def gate_safe_view(case: dict[str, Any], labels: set[str] | None = None) -> dict[str, Any]:
    """Strip the answer key before the workflow sees a case.

    `labels` restricts the documents, as in the October 2026 runs; the selection happens here,
    outside the workflow, which never receives a disposition.
    """
    documents = [
        {key: document[key] for key in ("key", "type", "specialty") if key in document}
        for document in select_documents(case, labels)
    ]
    safe = {"id": case["id"], "documents": documents}
    if ANSWER_KEY_FIELDS & set(safe):
        raise ValueError("answer-key field reached the gate view")
    return safe
