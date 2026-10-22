from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .cases import build_conditions, gate_safe_view, select_documents
from .prompts import _slug, build_generation_request


CONDITIONS = ("request_off", "request_on")


def load_model_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    models = config.get("generation_models")
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("generation_models must be a non-empty unique list")
    return config


def _selected_conditions(conditions: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(conditions) if conditions else CONDITIONS
    unknown = set(selected) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unsupported conditions: {sorted(unknown)}")
    return selected


def build_generation_manifest(
    cases: Iterable[dict[str, Any]],
    models: Iterable[str],
    methods: Iterable[str],
    *,
    conditions: Iterable[str] | None = None,
    document_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build single-call generation requests.

    `conditions` and `document_labels` restrict the design as the October 2026 runs did; run A
    regenerated only the request-absent condition.
    """
    models, methods = list(models), list(methods)
    selected_conditions = _selected_conditions(conditions)
    requests: list[dict[str, Any]] = []
    for case in cases:
        for condition, transcript in build_conditions(case).items():
            if condition not in selected_conditions:
                continue
            for document in select_documents(case, document_labels):
                for model in models:
                    for method in methods:
                        requests.append(
                            build_generation_request(
                                case_id=case["id"],
                                document=document,
                                condition=condition,
                                transcript=transcript,
                                model=model,
                                method=method,
                            )
                        )
    identifiers = [request["request_id"] for request in requests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("generation request identifiers are not unique")
    return requests


def build_gate_plan(
    cases: Iterable[dict[str, Any]],
    models: Iterable[str],
    *,
    conditions: Iterable[str] | None = None,
    document_labels: set[str] | None = None,
    trace: bool = False,
    output_floor: int = 0,
) -> list[dict[str, Any]]:
    """List the workflow bundles of a run: one per case, condition, and base model.

    Each bundle carries the answer-key-free case view and transcript that `gate_bundle` receives,
    plus the run settings: stage logging and the output-limit floor (0 means as specified).
    """
    models = list(models)
    selected_conditions = _selected_conditions(conditions)
    bundles: list[dict[str, Any]] = []
    for case in cases:
        view = gate_safe_view(case, document_labels)
        if not view["documents"]:
            continue
        for condition, transcript in build_conditions(case).items():
            if condition not in selected_conditions:
                continue
            for model in models:
                bundles.append(
                    {
                        "bundle_id": "__".join((case["id"], condition, _slug(model), "gate")),
                        "case_id": case["id"],
                        "condition": condition,
                        "model": model,
                        "case_view": view,
                        "transcript": transcript,
                        "trace": trace,
                        "output_floor": output_floor,
                    }
                )
    return bundles
