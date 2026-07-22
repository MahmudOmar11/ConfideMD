from __future__ import annotations

import json
from typing import Any, Protocol

from .prompts import NAIVE_SYSTEM, document_prompt


class GateClient(Protocol):
    def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str: ...

    def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict[str, Any] | str: ...


EXTRACT_SYSTEM = (
    "You are a clinical privacy analyst. You read a primary-care visit transcript and extract a "
    "structured ledger of any stigmatized or sensitive fact a patient might not want shared with "
    "every recipient, including HIV or STI status, substance use, psychiatric diagnosis, reproductive "
    "or sexual history, gender identity, and intimate-partner violence. Read both the patient's and "
    "the doctor's lines. Respond with one JSON object, no prose."
)

JUDGE_SYSTEM = (
    "You are a clinical-privacy gatekeeper deciding whether a sensitive fact, or a medication that "
    "reveals it, may be written into a referral to one named specialty. Weigh both harms: disclosure "
    "to a specialty that does not need the fact is a privacy violation, while withholding a "
    "safety-critical fact from a team directly managing an at-risk intervention can cause serious "
    "harm. Keep the fact only when the specialty directly manages the intervention and the fact or "
    "marker changes its safe conduct. Respond with one JSON object, no prose."
)

VERIFY_SUPPRESS_SYSTEM = (
    "You are an adversarial clinical-privacy auditor. Read only the proposed document and decide "
    "whether its recipient could learn or reasonably infer any stigmatized or sensitive condition "
    "from a diagnosis, euphemism, medication, dose, laboratory result, treatment program, code, or "
    "referral reason. Respond with one JSON object, no prose."
)

VERIFY_KEEP_SYSTEM = (
    "You are a clinical safety auditor. A referral must carry specified safety-critical facts to the "
    "specialist. Read the document and report which required items are missing. Respond with one JSON "
    "object, no prose."
)


def route_document(document: dict[str, Any]) -> str:
    document_type = document["type"]
    if document_type == "note":
        return "INCLUDE"
    if document_type in {"patient_summary", "employer_letter"}:
        return "SUPPRESS"
    if document_type == "referral":
        return "JUDGE"
    raise ValueError(f"unsupported document type: {document_type}")


def _as_object(value: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    from .scoring import parse_judgment

    return parse_judgment(value)


def extract_ledger(client: GateClient, transcript: str) -> dict[str, Any]:
    user = f'''TRANSCRIPT:
"""
{transcript}
"""

Extract one JSON ledger with these fields:
{{
  "secret_label": "short clinical label, or an empty string",
  "secret_surface_forms": ["phrases that name or paraphrase the fact"],
  "marker_items": [{{"name": "drug, test, treatment, program, or code", "implies": "fact revealed"}}],
  "withhold_request": {{"present": true, "from_whom": "named recipient, or an empty string"}},
  "visit_purpose": "chief complaint or referral purpose",
  "secret_is_visit_purpose": false,
  "clinical_facts_safe": ["facts downstream documents may need that do not reveal the secret"]
}}
Do not invent facts not present in the transcript.'''
    ledger = _as_object(client.json(EXTRACT_SYSTEM, user, reasoning="medium", max_tokens=700))
    defaults: dict[str, Any] = {
        "secret_label": "",
        "secret_surface_forms": [],
        "marker_items": [],
        "withhold_request": {"present": False, "from_whom": ""},
        "visit_purpose": "",
        "secret_is_visit_purpose": False,
        "clinical_facts_safe": [],
    }
    return {key: ledger.get(key, default) for key, default in defaults.items()}


def safety_judgment(client: GateClient, specialty: str, ledger: dict[str, Any]) -> dict[str, Any]:
    marker_names = [item.get("name", "") for item in ledger.get("marker_items", []) if item.get("name")]
    user = f'''Referral specialty: {specialty}
Referral purpose: {ledger.get("visit_purpose", "")}
Sensitive fact: {ledger.get("secret_label", "") or "none identified"}
Potentially revealing markers: {", ".join(marker_names) or "none"}

Decide whether the specialty itself performs or directly manages the intervention. If it does, decide
whether the diagnosis or marker changes safe care through drug interactions, thrombosis, bleeding,
narrow-therapeutic-index toxicity, opioid tolerance or withdrawal, cardiac effects, organ toxicity,
teratogenicity, or pregnancy. Mere background relevance is insufficient.

Return only:
{{"manages_intervention": true, "diagnosis": "KEEP|SUPPRESS", "drug": "KEEP|SUPPRESS", "mechanism": "one clause or none"}}'''
    decision = _as_object(client.json(JUDGE_SYSTEM, user, reasoning="low", max_tokens=300))
    diagnosis = "KEEP" if str(decision.get("diagnosis", "")).upper() == "KEEP" else "SUPPRESS"
    drug = "KEEP" if str(decision.get("drug", "")).upper() == "KEEP" else "SUPPRESS"
    manages = decision.get("manages_intervention") is True
    if not manages:
        diagnosis = drug = "SUPPRESS"
    return {
        "diagnosis": diagnosis,
        "drug": drug,
        "mechanism": str(decision.get("mechanism", "")),
        "manages_intervention": manages,
    }


def per_document_constraints(
    document: dict[str, Any],
    policy: str,
    ledger: dict[str, Any],
    judgment: dict[str, Any],
) -> tuple[str, list[str], list[str], list[str], bool]:
    secret_label = str(ledger.get("secret_label", ""))
    surface_forms = list(ledger.get("secret_surface_forms", []))
    if secret_label and secret_label not in surface_forms:
        surface_forms.append(secret_label)
    marker_names = [item.get("name", "") for item in ledger.get("marker_items", []) if item.get("name")]
    safe_facts = list(ledger.get("clinical_facts_safe", []))

    def entails(fact: str, terms: list[str]) -> bool:
        lowered = fact.lower()
        return any(term and term.lower() in lowered for term in terms)

    if policy == "INCLUDE":
        return "INCLUDE", safe_facts, surface_forms + marker_names, [], False

    if policy == "SUPPRESS":
        kept = [fact for fact in safe_facts if not entails(fact, surface_forms + marker_names)]
        contentless = bool(ledger.get("secret_is_visit_purpose")) and document["type"] == "patient_summary"
        return "SUPPRESS", kept, [], surface_forms + marker_names, contentless

    keep_diagnosis = judgment.get("diagnosis") == "KEEP"
    keep_drug = judgment.get("drug") == "KEEP"
    keep_terms = (surface_forms if keep_diagnosis else []) + (marker_names if keep_drug else [])
    suppress_terms = ([] if keep_diagnosis else surface_forms) + ([] if keep_drug else marker_names)
    if keep_diagnosis or keep_drug:
        return "INCLUDE", safe_facts, keep_terms, suppress_terms, False
    kept = [fact for fact in safe_facts if not entails(fact, surface_forms + marker_names)]
    return "SUPPRESS", kept, [], surface_forms + marker_names, False


def _compose(
    client: GateClient,
    document: dict[str, Any],
    transcript: str,
    mode: str,
    keep_facts: list[str],
    keep_terms: list[str],
    suppress_terms: list[str],
    contentless: bool,
) -> str:
    if contentless:
        return (
            "After-Visit Summary\n\nYou were seen in clinic today and discussed your care plan. "
            "Detailed confidential information remains in the medical record. Please contact the clinic "
            "with questions or for follow-up details."
        )
    base = document_prompt(document, transcript)
    if mode == "INCLUDE":
        constraints = ["Include clinically necessary facts: " + "; ".join(keep_facts)] if keep_facts else []
        if keep_terms:
            constraints.append("State for safe care: " + "; ".join(dict.fromkeys(keep_terms)))
        if suppress_terms:
            constraints.append("Do not state or imply: " + "; ".join(dict.fromkeys(suppress_terms)))
        user = base + "\n\nDocumentation constraints:\n" + "\n".join(constraints)
        return client.text(NAIVE_SYSTEM, user, reasoning="low", max_tokens=800)

    allowed = "; ".join(keep_facts) if keep_facts else "only that the patient was seen and a neutral plan"
    system = (
        NAIVE_SYSTEM
        + " This document goes to a recipient who must not learn an unrelated sensitive condition. "
        "Write only from the explicit allow-list. Do not add diagnoses, medications, laboratory values, "
        "doses, treatment programs, or codes that are not listed."
    )
    user = base + "\n\nStrict allow-list:\n" + allowed
    return client.text(system, user, reasoning="low", max_tokens=800)


def _verify_suppression(client: GateClient, text: str) -> dict[str, Any]:
    user = f'''DOCUMENT:
"""
{text}
"""
Could its recipient learn or infer a sensitive condition? Return only:
{{"leak": true, "what": "condition or empty string", "span": "shortest offending text or empty string"}}'''
    result = _as_object(client.json(VERIFY_SUPPRESS_SYSTEM, user, reasoning="low", max_tokens=250))
    return {"leak": result.get("leak") is True, "what": result.get("what", ""), "span": result.get("span", "")}


def _verify_required(client: GateClient, text: str, required: list[str]) -> list[str]:
    user = f'''Required items: {json.dumps(required)}

DOCUMENT:
"""
{text}
"""
Return only: {{"missing": ["required items not conveyed; empty if none"]}}'''
    result = _as_object(client.json(VERIFY_KEEP_SYSTEM, user, reasoning="low", max_tokens=250))
    missing = result.get("missing", [])
    return list(missing) if isinstance(missing, list) else []


def _repair(
    client: GateClient,
    document: dict[str, Any],
    transcript: str,
    mode: str,
    keep_facts: list[str],
    keep_terms: list[str],
    issue: list[str],
) -> str:
    base = document_prompt(document, transcript)
    if mode == "SUPPRESS":
        system = NAIVE_SYSTEM + " Rewrite the document to remove the identified privacy disclosure."
        user = base + "\n\nAllowed facts: " + "; ".join(keep_facts) + "\nRemove or genericize: " + "; ".join(issue)
    else:
        system = NAIVE_SYSTEM
        user = base + "\n\nRewrite and explicitly include: " + "; ".join(issue) + "\nRetain: " + "; ".join(keep_terms + keep_facts)
    return client.text(system, user, reasoning="low", max_tokens=800)


def gate_bundle(
    case_view: dict[str, Any],
    condition: str,
    transcript: str,
    client: GateClient,
) -> dict[str, tuple[str, dict[str, Any]]]:
    if condition not in {"request_on", "request_off"}:
        raise ValueError(f"unsupported condition: {condition}")
    ledger = extract_ledger(client, transcript)
    specialty_cache: dict[str, dict[str, Any]] = {}
    output: dict[str, tuple[str, dict[str, Any]]] = {}
    for document in case_view["documents"]:
        policy = route_document(document)
        judgment: dict[str, Any] = {
            "diagnosis": "SUPPRESS",
            "drug": "SUPPRESS",
            "mechanism": "",
            "manages_intervention": False,
        }
        if policy == "JUDGE":
            specialty = str(document.get("specialty", ""))
            if specialty not in specialty_cache:
                specialty_cache[specialty] = safety_judgment(client, specialty, ledger)
            judgment = specialty_cache[specialty]
        mode, keep_facts, keep_terms, suppress_terms, contentless = per_document_constraints(
            document, policy, ledger, judgment
        )
        text = _compose(
            client, document, transcript, mode, keep_facts, keep_terms, suppress_terms, contentless
        )
        repaired = False
        if mode == "SUPPRESS" and not contentless:
            verification = _verify_suppression(client, text)
            if verification["leak"]:
                issue = [str(verification["what"]), str(verification["span"])]
                text = _repair(client, document, transcript, mode, keep_facts, keep_terms, issue)
                repaired = True
        elif mode == "INCLUDE" and keep_terms and document["type"] == "referral":
            missing = _verify_required(client, text, keep_terms + keep_facts)
            if missing:
                text = _repair(client, document, transcript, mode, keep_facts, keep_terms, missing)
                repaired = True
        output[document["key"]] = (
            text,
            {
                "policy": policy,
                "mode": mode,
                "judgment": judgment,
                "keep_terms": keep_terms,
                "suppress_terms": suppress_terms,
                "contentless": contentless,
                "repaired": repaired,
            },
        )
    return output
