# v1.0.0: Reference currency sensitivity and forecast combination for FX cash tail risk

This is the first public release of the replication package for the manuscript
*Forecasting foreign exchange cash tail risk under reference currency uncertainty*.

## What this release contains

- The official ECB daily reference exchange rate snapshot used in the study, with observation dates, status flags and quotation directions retained.
- Derived panels, datasets and table inputs.
- Recorded forecasts for every evaluated system, reference currency, cash book and holding interval.
- The portable research scripts, including the exact enumeration of the reference specific scenario banks.
- Provenance and verification records: environment description, publication validation and legacy audit scripts.
- Repository documentation: README, reproducibility guide, data sources, third party notices, and separate software and data licences.

## Verification

The package was checked by a continuous integration workflow on the default branch and on this tag:

- Frozen input files: 348
- Historical panel rows reconstructed: 6,913
- Cash outcome rows reconstructed: 27,640
- Actual forecast rows reconstructed: 240 at the two checked forecast origins
- Main statistical rows recomputed with a maximum numeric error of zero across model summary, primary score contrasts and calibration rows
- Fresh direct baseline fits: 4 systems, 24 rows

Limitations of this verification are stated explicitly in the provenance record. The full history was not refitted, the supplementary experiments were not all refitted, and no independent platform numeric validation was performed.

## Licensing

Original research software is released under the MIT licence. Original derived research outputs are released under CC BY 4.0. The ECB source observations remain subject to the ECB reuse conditions and are redistributed with source attribution.

## Citation

If you use this package, please cite the version specific DOI once the Zenodo deposit associated with this release has been created, or cite the repository directly in the meantime.
