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


def gate_safe_view(case: dict[str, Any]) -> dict[str, Any]:
    documents = [
        {key: document[key] for key in ("key", "type", "specialty") if key in document}
        for document in case["documents"]
    ]
    safe = {"id": case["id"], "documents": documents}
    if ANSWER_KEY_FIELDS & set(safe):
        raise ValueError("answer-key field reached the gate view")
    return safe
