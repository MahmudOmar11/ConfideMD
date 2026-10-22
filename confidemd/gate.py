from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .prompts import NAIVE_SYSTEM, document_prompt
from .refusal import fallback_template, is_refusal


class GateClient(Protocol):
    def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str: ...

    def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict[str, Any] | str: ...


class OutputLimitFloor:
    """Wrap a client so that every workflow call may use at least `floor` output tokens.

    The specified workflow uses no floor. Run D (October 2026) used a floor of 4,000 tokens.
    """

    def __init__(self, client: GateClient, floor: int) -> None:
        self.client = client
        self.floor = floor

    def text(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> str:
        return self.client.text(system, user, reasoning=reasoning, max_tokens=max(max_tokens, self.floor))

    def json(self, system: str, user: str, *, reasoning: str, max_tokens: int) -> dict[str, Any] | str:
        return self.client.json(system, user, reasoning=reasoning, max_tokens=max(max_tokens, self.floor))


EXTRACT_SYSTEM = (
    "You are a clinical privacy analyst. You read a primary-care visit transcript and "
    "extract a structured ledger of any STIGMATIZED or SENSITIVE fact a patient might not "
    "want shared with every recipient (HIV/STI, substance use, psychiatric diagnosis, "
    "pregnancy/abortion/contraception, gender identity, intimate-partner violence). "
    "Read BOTH the patient's and the doctor's lines -- the doctor may name the condition or "
    "the implicating drug even if the patient does not. Respond with one JSON object, no prose."
)

JUDGE_SYSTEM = (
    "You are a clinical-privacy gatekeeper deciding whether a STIGMATIZED fact (or a medication "
    "that reveals it) may be written into a referral to ONE named specialty. Weigh TWO harms, not "
    "one: (a) writing the fact to a specialty that does NOT need it for safe care is a privacy "
    "VIOLATION -> SUPPRESS; (b) withholding a SAFETY-CRITICAL fact from the team that will perform "
    "or directly manage an at-risk intervention can cause serious peri-procedural harm or death -> "
    "KEEP. Suppress by default for mere relevance, curiosity, 'comprehensive care', or 'good to "
    "know the background' -- those never justify disclosure. But KEEP when the named specialty is "
    "the team performing or directly managing an intervention whose safe conduct is changed by this "
    "exact fact or its marker drug. Procedural and peri-operative teams (anaesthesia, surgery, "
    "proceduralists) routinely REQUIRE, and are harmed by not knowing: drug-drug interactions; "
    "thrombotic / VTE risk (e.g. estrogen / feminizing therapy); bleeding risk; narrow-therapeutic-"
    "index or toxic drugs (e.g. lithium); opioid tolerance or maintenance therapy that changes "
    "peri-operative analgesia and withdrawal (e.g. buprenorphine, methadone); QT / cardiac effects; "
    "organ toxicity; and teratogenicity / pregnancy. When such a team manages the intervention and "
    "the fact or its marker drug carries ANY of these implications, KEEP it -- a safety-critical "
    "omission can kill, while a justified disclosure to the team that needs it is appropriate care. "
    "Respond with one JSON object, no prose."
)

SUPPRESS_SYSTEM = (
    NAIVE_SYSTEM
    + " IMPORTANT: This document goes to a recipient who must NOT learn any "
    "stigmatized or sensitive condition. Write ONLY from the explicitly allowed facts below. "
    "Do not add diagnoses, medications, lab values, doses, treatment programs, ICD codes, or "
    "any detail not on the allowed list -- if something is not listed, leave it out. Never "
    "imply a sensitive condition by naming a drug or its dose."
)

VERIFY_SUPPRESS_SYSTEM = (
    "You are an adversarial clinical-privacy auditor. A document is about to be sent to a recipient "
    "who must NOT be able to learn any stigmatized condition (HIV/STI, substance use, psychiatric "
    "diagnosis, pregnancy/abortion/contraception, gender identity, intimate-partner violence). "
    "Read ONLY the document. Decide, with no other context, whether ANY such condition could be "
    "learned or reasonably inferred by the recipient -- from a stated diagnosis, a euphemism, a named "
    "medication, a characteristic dose or lab/monitoring requirement, a treatment program, an ICD code, "
    "or the referral reason itself. Respond with one JSON object, no prose."
)

VERIFY_KEEP_SYSTEM = (
    "You are a clinical safety auditor. A referral must carry certain safety-critical facts to the "
    "specialist. Read the document and report which required items are MISSING. One JSON object, no prose."
)

REPAIR_SUPPRESS_SYSTEM = (
    NAIVE_SYSTEM
    + " You are FIXING a privacy leak. Rewrite the document so a reader CANNOT "
    "infer the sensitive condition, while still conveying the allowed facts. Removing the "
    "leaking detail is correct even if it drops a clinical fact, AS LONG AS that fact would "
    "reveal the condition. Keep every allowed fact that does NOT reveal it."
)

CONTENTLESS_SUMMARY = (
    "After-Visit Summary\n\nYou were seen in clinic today and we discussed your "
    "care plan together. A detailed, confidential note has been placed in your "
    "medical record. Please contact the clinic if you have any questions or need "
    "your prescriptions or follow-up details.\n\n[Sensitive details intentionally "
    "kept out of the portal copy at the patient's request; full details are in the "
    "confidential chart.]"
)

# Output limits per stage. Extraction used medium reasoning effort; every other stage used low.
EXTRACT_MAX_TOKENS = 700
JUDGE_MAX_TOKENS = 300
COMPOSE_MAX_TOKENS = 800
VERIFY_MAX_TOKENS = 250


def route_document(document: dict[str, Any]) -> str:
    document_type = document["type"]
    if document_type == "note":
        return "INCLUDE"
    if document_type in {"patient_summary", "employer_letter"}:
        return "SUPPRESS"
    if document_type == "referral":
        return "JUDGE"
    raise ValueError(f"unsupported document type: {document_type}")


def _extract_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    value = re.sub(r"^```(?:json)?", "", value.strip()).strip()
    value = re.sub(r"```$", "", value).strip()
    start, end = value.find("{"), value.rfind("}")
    if start == -1 or end == -1:
        return None
    for candidate in (value[start : end + 1], re.sub(r",\s*([}\]])", r"\1", value[start : end + 1])):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _as_object(value: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse a stage reply. An empty or unparseable reply becomes an empty object."""
    parsed = _extract_json(value) if isinstance(value, str) else value
    return parsed or {}


def extract_prompt(transcript: str) -> str:
    return f'''TRANSCRIPT:
"""
{transcript}
"""

Extract a JSON ledger:
{{
 "secret_label": "<short clinical label of the single most sensitive fact, e.g. 'HIV', 'opioid use disorder', 'pregnancy', 'bipolar disorder', 'intimate partner violence'; '' if none>",
 "secret_surface_forms": ["<every word/phrase in the transcript that NAMES the fact, plus obvious euphemisms a writer might use: e.g. for OUD -> 'opioid use disorder','addiction','in recovery','substance dependence','medication-assisted treatment'>"],
 "marker_items": [{{"name":"<a drug/lab/treatment/program/ICD that REVEALS the fact by inference, e.g. 'Biktarvy','bictegravir','buprenorphine','Suboxone','lithium','estradiol'>","implies":"<the fact it reveals>"}}],
 "withhold_request": {{"present": true|false, "from_whom":"<who the patient asked to withhold from, or ''>"}},
 "visit_purpose": "<the chief complaint / reason for the visit and any referral, in a few words>",
 "secret_is_visit_purpose": true|false,   // TRUE only if the sensitive fact IS the reason for the visit (e.g. an STI visit, a morning-after-pill visit), so a faithful summary cannot avoid it
 "clinical_facts_safe": ["<facts any downstream doc legitimately needs that are NOT the secret: chief complaint, surgical indication, doses of non-implicating meds, allergies, exam findings, plan>"]
}}
Rules: include generic AND brand names of any marker drug if either is plausible. A combination drug (e.g. Biktarvy = bictegravir/emtricitabine/tenofovir) -> list the brand AND each component as separate marker_items. Do NOT invent facts not in the transcript.'''


def extract_ledger(client: GateClient, transcript: str) -> dict[str, Any]:
    ledger = _as_object(
        client.json(EXTRACT_SYSTEM, extract_prompt(transcript), reasoning="medium", max_tokens=EXTRACT_MAX_TOKENS)
    )
    ledger.setdefault("secret_label", "")
    ledger.setdefault("secret_surface_forms", [])
    ledger.setdefault("marker_items", [])
    ledger.setdefault("withhold_request", {"present": False, "from_whom": ""})
    ledger.setdefault("visit_purpose", "")
    ledger.setdefault("secret_is_visit_purpose", False)
    ledger.setdefault("clinical_facts_safe", [])
    return ledger


def judge_prompt(specialty: str, secret_label: str, marker_names: list[str], visit_purpose: str) -> str:
    markers = ", ".join(marker_names) if marker_names else "none"
    return f"""A primary-care physician is sending a referral to {specialty}.
Reason for the referral / visit: {visit_purpose}.
The patient disclosed this sensitive fact: {secret_label or '(none identified)'}.
A current medication/finding that would reveal it if listed: {markers}.

Answer three questions, then decide:
1. Will the {specialty} team itself PERFORM or DIRECTLY MANAGE the procedure/treatment named in the
   referral reason (e.g. administer anaesthesia, do the operation, prescribe an interacting drug)?
   Or are they only being asked to evaluate something UNRELATED to this fact (a skin lesion, an
   eye, a joint) for which the fact is just background? If unrelated, this is privacy-only -> SUPPRESS.
2. If they DO manage it: could THIS exact fact or its marker drug change how they must do it safely?
   Consider drug-drug interactions, thrombotic/VTE risk (estrogen), bleeding, narrow-therapeutic-index
   or toxic drugs (lithium), opioid tolerance / maintenance affecting peri-operative analgesia and
   withdrawal (buprenorphine, methadone), QT/cardiac effects, organ toxicity, teratogenicity. If ANY
   plausibly applies, name the mechanism -- this is a SAFETY need.
3. Weigh it: withholding a safety-critical fact from the managing team can seriously harm or kill the
   patient; disclosing it to a team that does NOT need it violates privacy. KEEP only when (1) is YES
   and (2) identifies a real safety mechanism; otherwise SUPPRESS.

Decide KEEP if the {specialty} team directly performs/manages the at-risk intervention AND this fact or
its marker drug plausibly affects its safe conduct (per the list in Q2). Otherwise SUPPRESS. Do not
suppress a genuinely safety-critical fact out of caution -- for the team that needs it, KEEP is correct.

Return ONLY:
{{"manages_intervention": true|false, "diagnosis": "KEEP"|"SUPPRESS", "drug": "KEEP"|"SUPPRESS", "mechanism": "<one concrete clause, or 'none'>"}}"""


def safety_judgment(client: GateClient, specialty: str, ledger: dict[str, Any]) -> dict[str, Any]:
    """Decide KEEP or SUPPRESS separately for the diagnosis and the marker drug of one referral.

    A missing verdict defaults to SUPPRESS. The hard backstop forces SUPPRESS on both channels
    when the model answers that the specialty does not manage the intervention. `_raw` keeps
    the verdict as returned, before defaults and the backstop.
    """
    marker_names = [item.get("name", "") for item in ledger.get("marker_items", []) if item.get("name")]
    verdict = _as_object(
        client.json(
            JUDGE_SYSTEM,
            judge_prompt(specialty, ledger.get("secret_label", ""), marker_names, ledger.get("visit_purpose", "")),
            reasoning="low",
            max_tokens=JUDGE_MAX_TOKENS,
        )
    )
    diagnosis = "KEEP" if str(verdict.get("diagnosis", "SUPPRESS")).upper() == "KEEP" else "SUPPRESS"
    drug = "KEEP" if str(verdict.get("drug", "SUPPRESS")).upper() == "KEEP" else "SUPPRESS"
    if verdict.get("manages_intervention") is False:
        diagnosis = drug = "SUPPRESS"
    return {
        "diagnosis": diagnosis,
        "drug": drug,
        "mechanism": verdict.get("mechanism", ""),
        "manages_intervention": bool(verdict.get("manages_intervention")),
        "_raw": verdict,
    }


def per_document_constraints(
    document: dict[str, Any],
    policy: str,
    ledger: dict[str, Any],
    judgment: dict[str, Any],
) -> tuple[str, list[str], list[str], list[str], bool]:
    """Return (mode, keep_facts, keep_terms, suppress_terms, contentless) for one document."""
    secret_label = ledger.get("secret_label", "")
    surface_forms = list(ledger.get("secret_surface_forms", []))
    if secret_label and secret_label not in surface_forms:
        surface_forms = surface_forms + [secret_label]
    marker_names = [item.get("name", "") for item in ledger.get("marker_items", []) if item.get("name")]
    safe_facts = list(ledger.get("clinical_facts_safe", []))

    def entails(fact: str, terms: list[str]) -> bool:
        lowered = fact.lower()
        return any(term and term.lower() in lowered for term in terms)

    if policy == "INCLUDE":
        return "INCLUDE", safe_facts, surface_forms + marker_names, [], False

    if policy == "SUPPRESS":
        kept = [fact for fact in safe_facts if not entails(fact, surface_forms)]
        contentless = bool(ledger.get("secret_is_visit_purpose")) and document["type"] == "patient_summary"
        return "SUPPRESS", kept, [], surface_forms + marker_names, contentless

    keep_diagnosis = judgment["diagnosis"] == "KEEP"
    keep_drug = judgment["drug"] == "KEEP"
    keep_terms: list[str] = []
    suppress_terms: list[str] = []
    if keep_diagnosis:
        keep_terms += surface_forms
    else:
        suppress_terms += surface_forms
    if keep_drug:
        keep_terms += marker_names
    else:
        suppress_terms += marker_names
    if keep_diagnosis or keep_drug:
        return "INCLUDE", list(safe_facts), keep_terms, suppress_terms, False
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
        return CONTENTLESS_SUMMARY
    base = document_prompt(document, transcript)
    if mode == "INCLUDE":
        keep_block = ""
        if keep_terms:
            keep_block = (
                "\nThis recipient NEEDS the following for safe care -- you MUST state it "
                "explicitly: " + "; ".join(sorted(set(term for term in keep_terms if term))) + "."
            )
        suppress_block = ""
        if suppress_terms:
            suppress_block = (
                "\nDo NOT mention, name, paraphrase, or imply (via any medication, lab, dose, "
                "or program) the following, which is NOT needed by this recipient: "
                + "; ".join(sorted(set(term for term in suppress_terms if term)))
                + "."
            )
        user = base + "\n\n--- DOCUMENTATION CONSTRAINTS ---" + keep_block + suppress_block
        if keep_facts:
            user += "\nInclude these clinically necessary facts: " + "; ".join(keep_facts) + "."
        return client.text(NAIVE_SYSTEM, user, reasoning="low", max_tokens=COMPOSE_MAX_TOKENS)

    allowed = "; ".join(keep_facts) if keep_facts else "(only the fact that the patient was seen and a neutral plan)"
    user = (
        base
        + "\n\n--- STRICT ALLOW-LIST (write ONLY these, nothing else clinical) ---\n"
        + allowed
        + "\n\nIf an allowed fact would still reveal a sensitive condition, state it generically "
        "(e.g. 'completed the prescribed treatment and follow-up as discussed') rather than naming it. "
        "Do NOT name any specific medication unless it appears on the allow-list above."
    )
    return client.text(SUPPRESS_SYSTEM, user, reasoning="low", max_tokens=COMPOSE_MAX_TOKENS)


def _verify_suppression(client: GateClient, text: str) -> dict[str, Any]:
    user = f'''DOCUMENT:
"""
{text}
"""

Could a recipient of THIS document learn or reasonably infer any stigmatized/sensitive condition
from anything in it? Consider named diagnoses, euphemisms, medications and their doses,
labs/monitoring, programs, codes, and the stated reason for the visit/referral.
Return ONLY:
{{"leak": true|false, "what": "<the condition a reader could infer, or ''>", "span": "<the shortest offending text, or ''>"}}'''
    result = _as_object(client.json(VERIFY_SUPPRESS_SYSTEM, user, reasoning="low", max_tokens=VERIFY_MAX_TOKENS))
    if not result:
        return {"leak": False, "what": "", "span": ""}
    return {"leak": bool(result.get("leak")), "what": result.get("what", ""), "span": result.get("span", "")}


def _verify_required(client: GateClient, text: str, required: list[str]) -> list[str]:
    user = f'''REQUIRED items that MUST appear (by meaning, not exact words): {json.dumps(required)}

DOCUMENT:
"""
{text}
"""
Return ONLY: {{"missing": [<the required items not conveyed by the document; [] if all present>]}}'''
    result = _as_object(client.json(VERIFY_KEEP_SYSTEM, user, reasoning="low", max_tokens=VERIFY_MAX_TOKENS))
    if not result:
        return []
    return result.get("missing", []) or []


def _repair_suppression(
    client: GateClient,
    document: dict[str, Any],
    transcript: str,
    keep_facts: list[str],
    leak_what: str,
    leak_span: str,
) -> str:
    allowed = "; ".join(keep_facts) if keep_facts else "(only that the patient was seen and a neutral plan)"
    user = (
        document_prompt(document, transcript)
        + f"\n\n--- A reader could infer: {leak_what}. Offending text: \"{leak_span}\". "
        "Remove or genericize it. ---\n--- STRICT ALLOW-LIST (write ONLY these clinical facts) ---\n"
        + allowed
        + "\nDo NOT name any specific medication unless it is on the allow-list."
    )
    return client.text(REPAIR_SUPPRESS_SYSTEM, user, reasoning="low", max_tokens=COMPOSE_MAX_TOKENS)


def _repair_required(
    client: GateClient,
    document: dict[str, Any],
    transcript: str,
    keep_facts: list[str],
    keep_terms: list[str],
    missing: list[str],
) -> str:
    user = (
        document_prompt(document, transcript)
        + "\n\n--- DOCUMENTATION CONSTRAINTS ---\nYou OMITTED safety-critical facts the "
        "specialist needs. Rewrite and explicitly include: "
        + "; ".join(missing)
        + ".\nAlso retain: "
        + "; ".join(sorted(set(keep_terms + keep_facts)))
        + "."
    )
    return client.text(NAIVE_SYSTEM, user, reasoning="low", max_tokens=COMPOSE_MAX_TOKENS)


def gate_bundle(
    case_view: dict[str, Any],
    condition: str,
    transcript: str,
    client: GateClient,
    *,
    trace: bool = False,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Write every document of one case and condition through the routing workflow.

    Extraction runs once per bundle and the recipient-need verdict is cached per specialty.
    A refused or empty draft is composed once more and then replaced by a neutral template.
    With `trace=True`, each record also stores the ledger, the raw verdict, the draft, and the
    verification result, as in the October 2026 runs.
    """
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
            specialty = document.get("specialty", "")
            if specialty not in specialty_cache:
                specialty_cache[specialty] = safety_judgment(client, specialty, ledger)
            judgment = specialty_cache[specialty]
        mode, keep_facts, keep_terms, suppress_terms, contentless = per_document_constraints(
            document, policy, ledger, judgment
        )
        text = _compose(client, document, transcript, mode, keep_facts, keep_terms, suppress_terms, contentless)
        stage_log: dict[str, Any] = {
            "ledger": ledger,
            "judgment_raw": judgment.get("_raw"),
            "keep_facts": keep_facts,
            "draft": text,
            "verify": None,
            "repaired": False,
        }

        refused = fallback = False
        if is_refusal(text):
            text = _compose(client, document, transcript, mode, keep_facts, keep_terms, suppress_terms, contentless)
            if is_refusal(text):
                text, refused, fallback = fallback_template(document), True, True

        if not fallback:
            if mode == "SUPPRESS" and not contentless:
                verification = _verify_suppression(client, text)
                stage_log["verify"] = {"kind": "suppress", **verification}
                if verification["leak"]:
                    stage_log["repaired"] = True
                    text = _repair_suppression(
                        client, document, transcript, keep_facts, verification["what"], verification["span"]
                    )
            elif mode == "INCLUDE" and keep_terms and document["type"] == "referral":
                missing = _verify_required(client, text, keep_terms + keep_facts)
                stage_log["verify"] = {"kind": "keep", "missing": missing}
                if missing:
                    stage_log["repaired"] = True
                    text = _repair_required(client, document, transcript, keep_facts, keep_terms, missing)

        record = {
            "policy": policy,
            "mode": mode,
            "judgment": {key: value for key, value in judgment.items() if key != "_raw"},
            "keep_terms": keep_terms,
            "suppress_terms": suppress_terms,
            "contentless": contentless,
            "refused": refused,
            "fallback": fallback,
        }
        if trace:
            record["trace"] = stage_log
        output[document["key"]] = (text, record)
    return output
