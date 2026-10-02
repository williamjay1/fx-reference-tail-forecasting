# Numerical scope, ordering and checks

The core has 2,046 evaluation origins (2018-01-02 through 2025-12-29), two
specified cash books, EUR/USD reporting, and tau=.95/.99. Historical warmup
covers 2013–2017. The 2026 extension is 189 origins through 2026-09-28;
its last endpoint is 2026-09-30. The h5/h20 extensions use 2,042/2,027 origins.
Loss is positive cash loss, in stated normalized units; FZ0 is computed
only for its specified VaR/ES domain. Changing units does not create profits.

Main seed: 20261002 (Sobol uses seed+year); dynamic BEKK: 20261003.
Direct fits use a deterministic task/year hash and fixed optimization starts.
Inference uses common date blocks 20/60 and 5,000 draws where stated.
Bootstrap of evaluation losses does not re-estimate model parameters; the
separate S11 experiment does, for three stated origins only.

After the main refitting chain in README, run extensions in this order:

```text
python -B scripts/joint18_temporal_run.py
python -B scripts/joint18_horizon_run.py
python -B scripts/joint18_horizon_unfinanced.py
python -B scripts/joint18_temporal_analyse.py temporal
python -B scripts/joint18_temporal_analyse.py horizons
python -B scripts/joint18_horizon_unfinanced_analyse.py temporal
python -B scripts/joint18_horizon_unfinanced_analyse.py horizons
python -B scripts/joint18_multivariate.py --tag full16 --power 16 --start-year 2018 --end-year 2025
python -B scripts/joint18_pit_precision.py --tag full96
python -B scripts/joint18_temporal_precision.py
python -B scripts/joint17_precision_audit.py
python -B scripts/joint17_internal_scenarios.py --precision --power 16 --tag internal_precision_16
python -B scripts/joint17_internal_scenarios.py --precision --power 18 --tag internal_precision_18
python -B scripts/joint17_precision_independent_summary.py
python -B scripts/joint18_inference.py
python -B scripts/joint18_inference_extensions.py
python -B scripts/joint18_inference_anchor_free.py
python -B scripts/joint18_inference_multivariate_compare.py
python -B scripts/joint18_inference_multivariate_reporting.py
python -B scripts/joint18_diagnostic_bounds.py
python -B scripts/jof19_forecast_dispersion.py
python -B scripts/joint18_inference_parameter_bootstrap.py --tag three_origins_B200_block20 --reps 200 --block 20 --years 2018 2021 2025
python -B scripts/joint18_inference_parameter_bootstrap.py --tag three_origins_B200_block60 --reps 200 --block 60 --years 2018 2021 2025
python -B scripts/joint18_inference_parameter_note.py
python -B scripts/joint18_inference_anchor_note.py
```

Some programs refuse existing completed destinations; use a genuinely fresh
work directory for full refits. Full S11 involves 24,000 coordinate GARCH fits
and can dominate runtime. Stored original manifests report actual runtimes;
total runtime depends on hardware and no newly measured full-refit cost is
promised. Budget disk for several gigabytes of uncompressed working inputs
and additional generated outputs. Do not infer thousands of independent
investors from correlated tasks; resampling keeps common-date shocks together.

Scientific originals are archived under provenance/original_scripts. Portable
copies change filesystem configuration and the lookup of the exact source version
named by a provenance digest, plus the table renderer's input filenames to the
table-only extraction. Statistical formulas and model logic are preserved. The
pilot-only independent engineering audit is archived rather than exposed as a
current runnable entry point. Historical manifests are byte-identical
copies and may contain old local paths; published_script_sha256 and source_sha256
in replication_manifest distinguish this portability revision. Current main
models and forecasts were spot reconstructed; early direct-baseline code-version
differences and untested independent operating systems remain disclosed.

Validation scope and actual numerical errors are recorded in
provenance/publication_validation.json. Calibration failures, sparse q99 tails,
large reference sensitivity and adverse extension outcomes are not removed.
