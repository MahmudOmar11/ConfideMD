# Data dictionary

## Scored tables

The files `synthetic_scored.csv`, `patient_authored_derived_scored.csv`, and `gate_scored.csv` share
the following fields.

| Field | Meaning |
|---|---|
| `case_id` | Synthetic identifier or non-linkable public pseudonym |
| `template_id` | Synthetic source template; equals public case pseudonym for patient-authored data |
| `category` | Sensitive-information category |
| `document` | Intended document or recipient surface |
| `disposition` | Constructed routing label: forbidden, required, appropriate, or internal |
| `condition` | `request_off` or `request_on` |
| `model` | Evaluated model identifier |
| `method` | Gate strategy, present only in `gate_scored.csv` |
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

## Physician aggregate tables

`physician_validation_metrics.csv` contains estimates and row-bootstrap intervals for agreement,
Cohen's kappa, and scorer performance on the enriched validation sample.

`physician_disposition_decisions.csv` contains aggregate counts by constructed strategy stratum and
physician decision. It contains no row-level responses.
