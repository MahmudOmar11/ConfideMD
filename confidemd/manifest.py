from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .cases import build_conditions
from .prompts import build_generation_request


def load_model_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    models = config.get("generation_models")
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("generation_models must be a non-empty unique list")
    return config


def build_generation_manifest(
    cases: Iterable[dict[str, Any]],
    models: Iterable[str],
    methods: Iterable[str],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for case in cases:
        for condition, transcript in build_conditions(case).items():
            for document in case["documents"]:
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
