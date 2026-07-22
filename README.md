# ConfideMD

ConfideMD tests whether clinical documentation systems route sensitive facts according to the
recipient and clinical need. It measures inappropriate disclosure and harmful omission together.

This repository contains the synthetic cases, prompt and Gate implementation, adjudication logic,
sanitized scored data, analysis code, tests, and reference results needed to reproduce the public
study outputs. Included code makes no external service calls.

## Reproducibility scope

The synthetic benchmark can be reconstructed from the 120 included variants. The package provides:

- paired request and no-request transcript construction;
- prompts for unguarded, privacy-aware, contextual-integrity, and generate-then-redact methods;
- the recipient-conditioned ConfideMD Gate and its dual disclosure-omission check;
- provider-neutral generation and adjudication request builders; and
- the seven model identifiers and generation parameters recorded for the study.

Executing model requests requires access to the selected model endpoints. The repository does not
contain credentials or a provider transport. Model behavior can change after an endpoint update, so
the sanitized scored tables and expected results are included for exact analysis verification.

The patient-authored benchmark is reproducible at the analysis level only. Its 45 source queries and
source-row identifiers are not redistributed. The included pseudonymous scored table contains only
the fields required to reproduce the reported aggregate estimates.

## Contents

```text
confidemd/          Case handling, prompts, Gate, scoring, and release validation
data/synthetic/     120 synthetic case variants from 30 templates
data/derived/       Sanitized scored and aggregate validation tables
analysis/           Request construction, analysis, and validation commands
results/            Frozen reference result tables
tests/              Pipeline, endpoint, privacy, and integrity tests
```

## Verify the frozen analysis

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
python3 analysis/validate_release.py
uv sync --frozen
uv run python analysis/validate_package.py
uv run python analysis/run_analysis.py --output-dir /tmp/confidemd-results
uv run python -m unittest discover -s tests
```

The analysis uses 10,000 deterministic cluster-bootstrap resamples. Fresh outputs must match the
included result tables exactly.

## Reconstruct synthetic requests

The following command writes the 7,280 unguarded requests defined by 520 case-document combinations,
two request conditions, and seven models:

```bash
uv run python analysis/build_requests.py \
  --method naive \
  --output /tmp/confidemd-naive-requests.jsonl
```

The same command accepts `privacy-aware`, `ci-reasoning`, and `generate-then-redact`. The latter
creates the first-pass request; `confidemd.prompts.build_redaction_request` constructs its second pass
after a draft is available. The ConfideMD Gate is sequential and uses the injected `GateClient`
interface in `confidemd/gate.py`.

## Analysis definitions

- The primary privacy endpoint includes off-topic referrals and employer letters.
- Patient-summary disclosure is reported as sensitive-fact inclusion unless the request explicitly
  identifies a proxy, portal, or summary recipient.
- The raw adjudicating-judge verdict is primary. Lexical detection is a sensitivity analysis.
- Synthetic uncertainty is clustered by the 30 source templates. Patient-authored uncertainty is
  clustered by case.
- Request and method comparisons are paired within case, document, condition where applicable, and
  model.
- Missing scored cells remain missing and all denominators are explicit.

See [CODEBOOK.md](CODEBOOK.md) for field definitions and [NOTICE.md](NOTICE.md) for provenance and
redistribution boundaries.

## License

Code is licensed under the MIT License. The synthetic benchmark and study-generated derived tables
are licensed under CC BY 4.0. The third-party patient-authored source corpus is not included or
licensed by this repository.
