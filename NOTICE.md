# Data provenance and privacy notice

## Synthetic benchmark

The synthetic case bank contains 120 variants from 30 templates. These cases are constructed and do
not represent real patients.

## Patient-authored benchmark

The study used 45 patient-authored healthcare queries selected from the third-party
ChatDoctor/HealthcareMagic 100k corpus. The source corpus, source-row identifiers, patient text, raw
generated documents, and adjudication quotes are intentionally excluded from this repository.

`data/derived/patient_authored_derived_scored.csv` contains pseudonymized case identifiers and only
the analytical fields needed to reproduce aggregate rates. It must not be used to infer or attempt
to relink source authors.

`data/withholding_requests.csv` includes the one-sentence withholding request that the study added as
the final line of each patient-authored query. These requests were proposed by a screening model and
reviewed by the investigators; they were not written by the patients. For these cases the file holds
the request line, its position, the domain, and pseudonymous case labels. It contains no patient
text, no clinician reply, and no source-row identifier.

No claim is made that these public online queries are equivalent to electronic health record cases
or that they represent a clinical population. Users who want to reconstruct the source selection
must obtain the original third-party corpus under its applicable terms. `confidemd/screening.py`
contains the screening instructions and selection rules and makes no download.

The source dataset card did not specify a redistribution license when this package was prepared.
This repository therefore provides no patient-authored source text, source-row identifiers, or
content-derived quotations.

## Physician validation

Two physicians independently completed a blinded validation exercise. This repository contains only
aggregate validation metrics and the aggregate consensus counts behind the clinician-weighted score.
The filled workbooks and row-level validation key remain private.

## October 2026 runs

The scored data from the four additional runs contain the same analytical fields as the June tables
plus run-level flags. Generated documents, stage outputs, and adjudication quotes from these runs are
not included; the workflow stage of each required referral is given as a derived label.
