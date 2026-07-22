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

No claim is made that these public online queries are equivalent to electronic health record cases
or that they represent a clinical population. Users who want to reconstruct the source selection
must obtain the original third-party corpus under its applicable terms.

The source dataset card did not specify a redistribution license when this package was prepared.
This repository therefore provides no patient-authored source text, source-row identifiers, or
content-derived quotations.

## Physician validation

Two physicians independently completed a blinded validation exercise. This repository contains only
aggregate validation metrics. The filled workbooks and row-level validation key remain private.
