# ConfideMD

ConfideMD tests whether clinical documentation systems route sensitive facts according to the
recipient and clinical need. It measures inappropriate disclosure and harmful omission together.

This repository contains the synthetic cases, the withholding-request text for both case sets, the
generation, comparator, workflow, and adjudication prompts, the screening instructions for the
patient-authored cases, dated model identifiers, sanitized scored data, analysis code, tests, and
reference results needed to reproduce the public study outputs. Included code makes no external
service calls.

## Release 0.2.0 (October 2026)

This release adds the additional analyses reported in the Supplementary Information (Tables S12, S13,
and S18 to S22) and the material behind them:

- `data/withholding_requests.csv`: all 165 withholding requests, with the position of each request
  in its dialogue and, for synthetic dialogues, the clinician's reply (also Supplementary Data 1).
  Patient-authored rows carry only the added request line and its position.
- `data/model_versions.csv`: dated model identifiers, roles, reasoning-control availability, and
  access dates (Supplementary Table S1).
- `data/additional_runs.json`: the four additional runs of 22 October 2026 (A to D) and their settings.
- `data/derived/october_runs_scored.csv`: sanitized scored data from those runs.
- `data/derived/workflow_stage_records.csv`: the first workflow stage at which each required fact
  was lost, for the June workflow experiment and runs B to D.
- `data/derived/clinician_weights.csv`: the aggregate physician-consensus counts behind the
  clinician-weighted score.
- `data/derived/unscored_outputs.csv`: the 21 outputs that could not be scored, with the reason.
- `confidemd/screening.py`: the keyword lexicon, candidate selection, screening instructions, and
  case construction for the patient-authored case set.
- `analysis/run_additional_analyses.py` and `results/additional/`: the additional analyses.

| Supplementary table | Reference output |
|---|---|
| S1, models used in the study | `data/model_versions.csv` |
| S12, synthetic-authorship sensitivity analysis | `results/additional/deepseek_exclusion.csv` |
| S13, scorer sensitivity analyses | `results/additional/scorer_sensitivity.csv`, `scorer_paired_differences.csv` |
| S18, where the withholding requests appear | `results/additional/request_positions.csv` |
| S19, run-to-run variation | `results/additional/rerun_variation.csv` |
| S20, where required facts were lost | `results/additional/workflow_stage_trace.csv` |
| S21, workflow across base models and output limits | `results/additional/workflow_base_models.csv` |
| S22, piloting cases and held-out cases | `results/additional/workflow_heldout.csv`, `workflow_heldout_paired.csv` |

`results/additional/` also holds the paired request effects on harmful omission by safety-tension
stratum, per-condition rates for runs B to D, and `summary_counts.csv`, which reproduces counts
reported in the text, such as the 18,732 distinct scored documents.

### Changes in 0.2.0

The prompts and workflow logic reproduce, word for word, the code that generated and scored the
study outputs, as quoted in the Supplementary Methods. The workflow gains the two options used in
October 2026: `trace=True` stores every stage, and `OutputLimitFloor` raises the output limit of every
stage (run D used 4,000 tokens). The request builders accept the condition and document filters of
the October runs, and generation requests record the low reasoning effort used throughout.

## Reproducibility scope

The synthetic benchmark can be reconstructed from the 120 included variants. The package provides:

- paired request and no-request transcript construction;
- prompts for unguarded, privacy-aware, contextual-integrity, and generate-then-redact methods, and
  the output rules applied before scoring;
- the recipient-conditioned routing workflow (ConfideMD Gate) and its dual disclosure-omission check;
- provider-neutral generation, workflow, and adjudication request builders, and the lexical detector;
- the screening instructions and selection rules for the patient-authored case set; and
- the dated model identifiers and generation parameters recorded for the study.

Executing model requests requires access to the selected model endpoints. The repository does not
contain credentials or a provider transport. Model behavior can change after an endpoint update, so
the sanitized scored tables and expected results are included for exact analysis verification. Run A
shows how much outputs can vary: regenerating identical requests in October 2026 changed the outcome
of 16.6% of primary-endpoint documents for the two models tested.

The patient-authored benchmark is reproducible at the analysis level only. Its 45 source queries and
source-row identifiers are not redistributed. The included pseudonymous scored table contains only
the fields required to reproduce the reported aggregate estimates. The withholding request added to
each query is included; it was proposed during screening and reviewed by the investigators, and it
contains no source text. Given the source corpus, obtained under its own terms, the candidate pool
selection in `confidemd/screening.py` is deterministic and returns the 261 candidates that were
screened; the screening calls themselves require model access.

## Contents

```text
confidemd/             Cases, prompts, routing workflow, adjudication, screening, stage tracing,
                       and release validation
data/synthetic/        120 synthetic case variants from 30 templates
data/                  Withholding requests, model versions, model settings, October run settings
data/derived/          Sanitized scored data, stage records, and aggregate validation tables
analysis/              Request construction, analysis, and validation commands
results/               Frozen reference result tables (results/additional/ for the additional analyses)
tests/                 Pipeline, workflow, scoring, screening, replication, and integrity tests
```

## Verify the frozen analysis

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
python3 analysis/validate_release.py
uv sync --frozen
uv run python analysis/validate_package.py
uv run python analysis/run_analysis.py --output-dir /tmp/confidemd-results
uv run python analysis/run_additional_analyses.py --output-dir /tmp/confidemd-additional
uv run python -m unittest discover -s tests
```

The analyses use 10,000 deterministic cluster-bootstrap resamples, seeded by a label for each
estimate. Fresh outputs must match the included result tables exactly. The tests also check the
additional-analysis tables against the values printed in Supplementary Tables S12, S13, and S18 to S22.

## Reconstruct requests

The following command writes the 7,280 unguarded requests defined by 520 case-document combinations,
two request conditions, and seven models:

```bash
uv run python analysis/build_requests.py \
  --method naive \
  --output /tmp/confidemd-naive-requests.jsonl
```

The same command accepts `privacy-aware`, `ci-reasoning`, and `generate-then-redact`. The latter
creates the first-pass request; `confidemd.prompts.build_redaction_request` constructs its second pass
after a draft is available, and `confidemd.prompts.finalize_output` applies the output rules. The
routing workflow is sequential and uses the injected `GateClient` interface in `confidemd/gate.py`;
`--method gate` writes one workflow bundle per case, condition, and base model.

The October runs used the same commands with filters. Run A regenerated the request-absent condition:

```bash
uv run python analysis/build_requests.py \
  --method naive \
  --model deepseek/deepseek-v4-flash --model meta-llama/llama-3.3-70b-instruct \
  --condition request_off \
  --output /tmp/confidemd-run-a.jsonl
```

Run C wrote employer letters and off-topic and required referrals with every stage stored:

```bash
uv run python analysis/build_requests.py \
  --method gate --trace \
  --model openai/gpt-5.4-mini \
  --document employer_letter:forbidden --document referral:forbidden --document referral:required \
  --output /tmp/confidemd-run-c.jsonl
```

Run D is run C on DeepSeek V4 Flash with `--output-floor 4000`; run B is `--document
referral:required` on both June base models. `data/additional_runs.json` lists every setting.

## Analysis definitions

- The primary privacy endpoint includes off-topic referrals and employer letters.
- Patient-summary disclosure is reported as sensitive-fact inclusion unless the request explicitly
  identifies a proxy, portal, or summary recipient.
- The raw adjudicating-judge verdict is primary. Lexical detection and the clinician-weighted score
  are sensitivity analyses.
- Synthetic uncertainty is clustered by the 30 source templates; harmful omission is clustered by the
  10 templates that contain required referrals. Patient-authored uncertainty is clustered by case.
- Request and method comparisons are paired within case, document, condition where applicable, and
  model.
- Missing scored cells remain missing and all denominators are explicit.
- Run A excludes regenerated documents that came back empty (failed calls), with their pairs.
- Stage tracing assigns each required referral to the first stage at which its sensitive fact was not
  carried forward. The June experiment saved one record per document, so an empty extraction reply
  and a ledger that named no fact are counted together there.

See [CODEBOOK.md](CODEBOOK.md) for field definitions and [NOTICE.md](NOTICE.md) for provenance and
redistribution boundaries.

## License

Code is licensed under the MIT License. The synthetic benchmark, the withholding-request text, and the
study-generated derived tables are licensed under CC BY 4.0. The third-party patient-authored source
corpus is not included or licensed by this repository.
