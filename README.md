# FX reference currency tail forecasting

Replication software and data supporting **Forecasting foreign exchange cash
tail risk under reference currency uncertainty**, prepared for *Journal of
Forecasting*. The article has not been accepted or published. This repository
contains code, numeric tables and data, not the unpublished article text.

Author: **Junjie Zhang**, ORCID https://orcid.org/0009-0004-8821-4018.
Shanghai Academy of Global Governance & Area Studies and School of Economics
and Finance, Shanghai International Studies University, Shanghai 201620, China.
Correspondence: junjiezhang2024@shisu.edu.cn.
No external funding; no conflict of interest. Public financial observations
involve no human participants or identifiable personal data; ethics approval
and participant consent are not applicable.

## What the study tests

Reference currency changes can alter independently fitted coordinate risk
laws even when their cash valuation identities hold. The analysis evaluates
internally standardized joint FX scenario laws and whole-law probability
mixtures against historical/filtered simulation, direct VaR/ES forecasts,
forecast combinations and a dynamic multivariate benchmark. It reports
proper-score comparisons, date-block uncertainty, calibration, reference
dispersion, numerical stress failures and adverse temporal evidence.
No general predictive superiority, execution profit, or invariant numeric
VaR across economically different reporting currencies is claimed.

## Contents and immutable inputs

`scripts/`: portable scientific programs plus preparation and validation.
`replication_inputs/`: frozen ECB snapshot, panels, annual parameters,
recorded forecasts, statistics and selected native vector figures/tables.
`replication_manifest.json`: compressed and original byte checksums.
`provenance/original_scripts/`: original program bytes for historical hashes;
these archived originals are not the portable entry points.
`provenance/publication_validation.json`: bounded actual rerun results.

Read DATA_SOURCES.md, DATA_LICENSE.md and THIRD_PARTY_NOTICES.md before reuse.
CSV gzip compression is lossless. Historical execution manifests describe
the original runs, not a newly repeated complete training experiment.
A few early direct-baseline manifests reference an earlier shared-model
digest; their provenance is preserved, and bitwise full historical refitting
of every branch has not been claimed.

## Reproduce recorded forecasts and principal statistics

Python 3.12 was used. Install requirements.txt in your own virtual environment.
Pinning records the actual tested environment; independent platforms and
future dependency builds can differ. Work outputs go in FX_WORK_ROOT. On the
author's machine choose a fresh directory on D: (never the source project).

```powershell
$env:FX_WORK_ROOT = 'D:/MLWork/FXTailRiskReplica'
$env:NUMBA_CACHE_DIR = "$env:FX_WORK_ROOT/cache/numba"
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pip install -r requirements.txt
python -B scripts/prepare_replication.py
python -B scripts/replication_smoke.py --analysis
```

On Linux/macOS set `FX_WORK_ROOT` to a fresh writable directory, then use the
same Python commands. Preparation verifies every package input and refuses
to overwrite changed work files. Smoke validation rebuilds the historical
panel and cash outcomes, reconstructs all 120 system/task forecasts for each
of 2018-01-02 and 2020-03-17 at power16, and with `--analysis` recomputes the
full principal score and calibration tables from the recorded forecasts.
This checks a real parameter-to-forecast and forecast-to-inference path.
It does **not** refit the complete history or all supplementary experiments.

## Full model refitting

Use a second fresh work directory. Set FX_WORK_ROOT **before** launching any
Python command and run `prepare_replication.py --data-only`. The main chain is:

```text
python -B scripts/joint17_build_data.py
python -B scripts/joint17_run_scenarios.py --tag scenario_v2 --start-year 2013 --end-year 2025 --power 12
python -B scripts/joint17_internal_scenarios.py --tag internal_full_a --start-year 2013 --end-year 2015 --power 16
python -B scripts/joint17_internal_scenarios.py --tag internal_full_b --start-year 2016 --end-year 2018 --power 16
python -B scripts/joint17_internal_scenarios.py --tag internal_full_c --start-year 2019 --end-year 2021 --power 16
python -B scripts/joint17_internal_scenarios.py --tag internal_full_d --start-year 2022 --end-year 2025 --power 16
python -B scripts/joint17_merge_full.py
python -B scripts/joint17_direct_baselines.py --start-year 2013 --end-year 2017 --power 12 --out <WORK>/results/joint_fx_20261002/direct/warmup_2013_2017
python -B scripts/joint17_direct_baselines.py --start-year 2018 --end-year 2025 --power 12 --out <WORK>/results/joint_fx_20261002/direct/full_prior_aligned
python -B scripts/joint17_forecast_combinations.py --input <WORK>/results/joint_fx_20261002/internal_full/predictions.csv --additional-input <WORK>/results/joint_fx_20261002/direct/warmup_2013_2017/predictions.csv --additional-input <WORK>/results/joint_fx_20261002/direct/full_prior_aligned/predictions.csv --pool reliable --history 900 --out <WORK>/results/joint_fx_20261002/combination/full_reliable_v2
python -B scripts/joint17_analyse.py
python -B scripts/joint17_reference_uncertainty.py
```

Replace <WORK> with the absolute FX_WORK_ROOT value. CLI placeholders are not
literal arguments. Run ordering, seeds, extension experiments and costs are
described in REPRODUCIBILITY.md. Model fitting uses CPUs; no GPU, Torch,
paid database or secret API key is needed. Do not run full fits in a work
directory already populated with recorded model results.

## Figures, tables and versions

After installing frozen inputs, run jof20_nature_figures.py and
jof20_nature_tables.py. Outputs are native PDF/SVG plus 1200dpi PNG, with
geometric boundary checks. Only table source cells and captions are included.
Arial is not redistributed; substitute fonts may alter layout and need review.
The software Git tag v1.0.0 is the reproducibility snapshot. The author will
connect Zenodo and create the corresponding GitHub Release personally;
no DOI is currently asserted. See ZENODO_INSTRUCTIONS.md.

## AI assistance and accountability

OpenAI Codex assisted substantive research design, literature synthesis,
scientific programming, numerical analysis, figures and manuscript drafting
in September–October 2026. Exact model/build identifiers were not retained.
Automated numerical and provenance checks are reported with their scope.
AI is not an author. Scientific accountability and the final assessment of
the manuscript and deposit rest with the human author. The package does not
claim that an unperformed full human or independent-platform audit occurred.
