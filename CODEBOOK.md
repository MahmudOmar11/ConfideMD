# Data dictionary

## Scored tables

The files `synthetic_scored.csv`, `patient_authored_derived_scored.csv`, `gate_scored.csv`, and
`october_runs_scored.csv` share the following fields.

| Field | Meaning |
|---|---|
| `case_id` | Synthetic identifier or non-linkable public pseudonym |
| `template_id` | Synthetic source template; equals public case pseudonym for patient-authored data |
| `category` | Sensitive-information category |
| `document` | Intended document or recipient surface |
| `disposition` | Constructed routing label: forbidden, required, appropriate, or internal |
| `condition` | `request_off` or `request_on` |
| `model` | Evaluated model identifier |
| `method` | Generation method (`naive` or a workflow or comparator method); absent from the first two tables |
| `raw_leak` | Binary adjudicating-judge leak verdict |
| `harmful_omission` | Binary harmful-omission verdict |
| `term_positive` | Binary lexical-detector result |
| `surface` | Recipient-conditioned analytical surface |
| `summary_scope` | Automated scope classification for patient-summary rows |
| `unambiguous_forbidden` | Off-topic referral or employer-letter endpoint flag |
| `explicit_summary_forbidden` | Summary row whose request explicitly names a proxy, portal, or summary |
| `conservative_forbidden` | Union of the prior two flags |
| `legacy_clinical_forbidden` | Older patient-summary plus off-topic-referral definition |

`raw_leak` is the primary disclosure outcome. `term_positive` is a sensitivity analysis. Boolean
scope fields define analysis populations and are not additional model judgments.

`october_runs_scored.csv` adds three fields:

| Field | Meaning |
|---|---|
| `run` | Additional run of 22 October 2026, A to D, as defined in `data/additional_runs.json` |
| `empty_output` | The generated document was empty; empty documents were scored as generated, and run A excludes them with their pairs |
| `identical_to_original` | Run A only: the regenerated document is textually identical to the June 2026 document |

## Withholding requests

`data/withholding_requests.csv` has one row per case (120 synthetic, 45 patient-authored).

| Field | Meaning |
|---|---|
| `case_set` | `synthetic` or `patient_authored` |
| `case_id` | Case identifier, matching the scored tables |
| `supplementary_data_1_case` | Case label used in Supplementary Data 1 |
| `template_id` | Synthetic source template; equals `case_id` for patient-authored rows |
| `category` | Sensitive-information category |
| `speaker_label` | Label that precedes the request in the dialogue (`Pt` or `Patient request`) |
| `request_text` | The added request, verbatim; the dialogue line is `speaker_label: request_text` |
| `line_index` | Position of the request among the non-empty dialogue lines, counting from 1 |
| `dialogue_lines` | Number of non-empty dialogue lines |
| `position` | `line i of n`, or `final line` |
| `relative_position` | `line_index / dialogue_lines`, rounded to two decimals |
| `clinician_replies` | A clinician turn directly follows the request |
| `clinician_reply` | That clinician turn, for synthetic dialogues |
| `safety_tension` | Synthetic safety stratum: none, resolved, or unresolved |
| `request_author` | Who wrote the request line; every request was reviewed by the investigators |
| `line_words` | Words in the dialogue line, speaker label included (the measure in Table S18) |

## Model versions

`data/model_versions.csv` reproduces Supplementary Table S1. `request_identifier` is the model name
used in requests and, with `/` replaced by `-`, in the `model` field of the scored tables.
`listed_version` is the dated version listed by OpenRouter on 22 October 2026; the provider route
used in June 2026 could not be verified retrospectively, and Ling 2.6 Flash was no longer listed.
`reasoning_control` records whether the model exposes a reasoning-effort setting; low effort was
requested from every model, and models without the control ignore it. `access_dates` gives the study
phases in which the model was called, with the dates in Table S1.

## October runs

`data/additional_runs.json` defines runs A to D: method, models, request conditions, document labels
(`type:disposition`), stage logging, output-limit floor, and the number of documents generated.

## Workflow stage records

`workflow_stage_records.csv` has one row per required referral written by the routing workflow in
the June experiment (`run` = `main`) and in runs B to D. `record` is `saved` for June, where one
record per document was kept, and `stage_log` for October, where every stage was stored.
`first_lost_stage` is the first stage at which the sensitive fact was not carried forward:

| Value | Meaning |
|---|---|
| `extraction_no_output` | Extraction returned no output (June: or named no sensitive fact) |
| `extraction_no_fact` | Extraction named no sensitive fact |
| `recipient_need_no_output` | The recipient-need step returned no output, which defaults to suppress |
| `not_managing` | The specialty was judged not to manage the intervention (hard rule) |
| `recipient_need_suppress` | The recipient-need step chose suppress |
| `drafting_fallback` | Drafting failed twice and the neutral template was used |
| `passed_to_drafting` | The fact reached the drafting step |

The rules are in `confidemd/stages.py`. Harmful omission comes from the scored tables.

## Unscored outputs

`unscored_outputs.csv` lists the 21 outputs missing from the scored tables: `table` names the table,
the next five fields identify the output, and `reason` is either an adjudicator reply that could not
be parsed after three attempts or a generation that was never written. `characters` is the length of
the generated document.

## Physician aggregate tables

`physician_validation_metrics.csv` contains estimates and row-bootstrap intervals for agreement,
Cohen's kappa, and scorer performance on the enriched validation sample.

`physician_disposition_decisions.csv` contains aggregate counts by constructed strategy stratum and
physician decision. It contains no row-level responses.

`clinician_weights.csv` contains, for each scorer-disagreement cell and for intimate partner violence
versus other domains, the number of validation rows with physician consensus and how many of them
the physicians judged to disclose the fact. The clinician-weighted score gives a document 1 when the
adjudicator and the lexical detector both flag it, 0 when neither does, and otherwise the share of
physician-confirmed disclosures in its cell.
