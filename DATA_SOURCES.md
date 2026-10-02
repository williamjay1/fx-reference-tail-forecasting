# Source, processing and information clock

Only ECB daily reference observations enter the current numerical study.
The ECB source snapshot has SHA256
c00fd71c91d37af565c5bb8334e0b5c8c065b20b7377f965e2bb776319079a01.
It was retrieved on 2026-10-02 and includes raw observations through 2026-10-01.
The extension analysis explicitly truncates this snapshot at 2026-09-30;
the later raw observation is not used in its published evaluations.
Its original acquisition copy is immutable; this repository contains an exact
byte copy. All stored files have compressed and decompressed checksums in
replication_manifest.json. Renewed API downloads may incorporate revisions
and need not reproduce this historical snapshot.

Keep valid positive observations with OBS_STATUS A, four common currency
dates (USD, GBP, JPY, CHF), and EUR denominator. Take reciprocals to obtain
EUR per currency unit and add EUR=1. No missing date is forward filled.
The main panel has 6,913 dates, 1999-01-04 through 2025-12-31; the expanded
panel has 7,104 dates through 2026-09-30. Cash-loss tasks explicitly retain
origin, future start and endpoint dates and fixed currency holdings.

ECB formation and publication are different events; the study uses a fixed
intended cutoff and starts the holding interval at the following observation.
Historical first-release timestamps and real-time vintages are not observed.
The data therefore support retrospective reference-price forecasting, not a
verified trading strategy or execution-profit claim. Official information:
https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html

Original acquisition manifests retain historical local paths and also refer
to Fed H10 materials used solely in the clock review. Those are archival
provenance strings, not runtime dependencies; Fed observations do not enter
this package's fitted models. The required files are precisely the inputs
enumerated in replication_manifest.json. No complete literature/raw directory
or unrelated earlier project has been published.

All evaluated periods had been exposed during research development, including
the adverse 2026 extension. This is documented exploratory reanalysis. No
retrospective preregistration or independent confirmatory holdout is claimed.
