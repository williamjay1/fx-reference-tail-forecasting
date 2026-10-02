# Independent numerical integration and stress audit

2026-10-02. **Stage: numerical feasibility and method validation. Decision: CONDITIONAL for production inference.** This decision is not a judgment of journal acceptance or prediction performance. The exact finite IN_FHS comparisons can proceed; small q99 simulation-based differences require additional numerical verification. All 2018–2025 results are exposed development/reanalysis, not an untouched confirmatory sample.

## Scope and preserved outputs

The original-law audit reuses saved annual prior-only parameters and evaluates 14 fixed origins: 2015-01-13, 2015-01-15, 2015-01-16; 2020-03-10, 2020-03-17, 2022-09-22; and eight calendar-spaced origins 2018-02-15, 2019-08-15, 2020-02-17, 2021-08-16, 2022-02-15, 2023-08-15, 2024-02-15, 2025-08-15. March17 was added explicitly after the initial stress protocol. No annual parameter was re-estimated and no observation/scenario was clipped. The 4368 risk rows retain exact finite historical distributions and nested Sobol powers12/14/16.

The redesigned-law precision panel was selected separately by the root analyst: three January2015 dates plus nine 2018–2025 dates, including March10/17 and September22. It has 1440 risk forecasts at each power12/14/16/18. These dates do not coincide completely with the original audit, so before/after aggregate precision comparisons must not be described as a matched-date causal experiment. Each underlying experiment still uses fixed data, model parameters, seed and holdings across powers.

Power12/14/16/18 means respectively 4096/16384/65536/262144 scenarios per native model. A five-reference pool has five times the native count. Absolute native-versus-pool precision must therefore not be interpreted as a model-quality advantage. Here relative q/ES changes divide by the higher-resolution value. Earlier quick messages sometimes divided by the lower-resolution value; the saved audit CSV/JSON uses the stated higher-resolution denominator consistently.

Owned reproducibility scripts are `joint17_precision_audit.py` and `joint17_precision_independent_summary.py`; original yearly risk/changes/contrast tables are preserved in `math/precision`. The integrated summary, manifests, hashes and bound checks are in `NUMERICAL_PRECISION_AUDIT.json`. This report does not replace the original failed-run outputs.

## Original finite-support failure is economic, not an identity error

At the January15,2015 origin, EUR-training FHS250 for the net cashflow book reported in EUR has q99=0.042186 and ES99=1,226,498 in units of N. Its largest cash-loss scenario is 3,053,980×N; one path accounts for effectively all integrated ES. The largest simulated FX price multiplier is 15,269,900. The absolute cash-identity residual 2.018084e−10 is small relative to these magnitudes and explains the original absolute-tolerance stop, but replacing the guard with a relative tolerance does not repair the predictive distribution.

The mechanism is repeated historical standardized jumps multiplied by the new elevated conditional volatility, followed by exponentiation in the exact FX cash valuation. Empirical boundedness makes ES mathematically finite and still allows an unusably extreme distribution. Longer windows reduce the atom's probability but retain the failure: the corresponding FHS500/1250 ES values are approximately 612020/244514. The unfiltered HS500/1250 ES values are approximately 0.01414/0.01019. Preserve this as a model failure rather than hide it through clipping, a narrower benchmark universe, or a claim that the formal evaluation starts later.

The original t-copula and radial laws also have integration problems. On 2018–2025 fixed audit origins, native TCOP1250 maximum 12→16 relative ES change is 9.96% at q95 and 24.11% at q99. Original EWMA_RADIAL reaches 47.91%/91.52%. Increasing only to power14 does not settle this: native TCOP1250 maximum 14→16 ES change remains 5.55%/13.29%, and original radial 7.51%/14.19%. These are larger than the previously contemplated 1–5% performance differences. A deterministic common seed is useful for paired integration but does not make the integration error disappear.

## Redesigned law: verified boundedness and known-information clock

Independent tests confirmed the internal radial identity

`r_s' H_s^(-1) r_s = a / (.94 + .06 a)`, where `a = r_s' H_(s-1)^(-1) r_s`.

The maximum Sherman–Morrison identity discrepancy is 8.88e−16 and maximum cross-basis radius discrepancy 2.84e−14 on fixed origins. Across the full historical panel, the largest raw squared radius is 16.66266, below the theoretical 1/.06 bound16.66667. Rescaling the empirical radial distribution gives exactly E[R_scaled²]=4 before random direction integration, to error1.78e−15. With an independent uniform direction on the four-dimensional sphere this implies covariance H; it does not mean a finite simulated sample has exactly covariance H. Gaussian/radial controls hold current covariance fixed at both future nodes.

Scalar post-innovation residuals satisfy `|eps_s / sqrt(omega + alpha eps_s² + beta h_s)| <= 1/sqrt(alpha)`. Independent reconstruction of annual2015/2020/2022/2025 filters and prior windows passed. Across this subset minimum alpha is0.0148424, minimum internal-residual standard deviation0.848906, and maximum centered standardized prior shock5.87900. These are numerical observations, not universal constants or proof that every future origin has the same bounds.

Historical post-innovation variance at s<t is known at the forecast cutoff. Simulated first-node returns then enter the ordinary explicit GARCH update for node2. Perturbing future returns leaves the known covariance prefix unchanged exactly. The 20-observation EWMA initialization precedes every evaluated origin. No future endpoint is used to define the forecast residual window. The new law reuses prior-only parameters but changes their predictive innovation law; this must be reported as a post-stress exploratory redesign.

The new copula uses discrete empirical inverse CDFs. This preserves the prior standardized empirical marginal mean0 and variance1 as a probability-law statement. The adjacent-pair FHS first and second marginals each omit a different endpoint, so they are not separately exactly standardized; this small endpoint effect should not be described as an exact conditional variance identity.

The power18 panel has maximum cash-loss scenario0.37858×N across all retained dates/books/reporting currencies. At January15,2015 the redesigned EUR-member IN_FHS1250 net-cashflow ES99 is0.032134, with maximum loss0.043455; its five-reference pool ES99 is0.031916, maximum0.070230. These demonstrate removal of the prior astronomical support pathology. They do not establish calibration or superiority.

## Remaining integration error and reference contrasts

The following maxima concern the **nine formal-period fixed origins only**, comparing nested power16 to18. They are stability diagnostics, not confidence intervals.

| Model | max relative ES95 change | max relative ES99 change | max absolute FZ0 score change at q99 |
|---|---:|---:|---:|
| IN_TCOP1250 native references |0.728%|1.434%|0.92868|
| IN_TCOP_REF_POOL |0.159%|0.432%|0.11505|
| EWMA_SELFNORMALIZED |0.521%|0.478%|0.37817|
| EWMA_GAUSS |0.069%|0.260%|0.05904|
| HS1250, IN_FHS1250 and IN_FHS_REF_POOL |0|0|0|

Maximum q99 relative VaR changes are1.312% for native IN_TCOP,0.323% for its pool,0.881% for self-normalized radial and0.670% for Gaussian. Exact finite historical/pair mixtures do not depend on the Sobol power.

Each q level has144 native-reference-versus-EUR contrasts, pooling nine dates, two books, two reporting currencies and four alternative training references. At power16→18, q95 has one VaR and one ES sign change and no score sign change. At q99 there are four VaR, two ES and two score sign changes. At q99, integration changes exceed the final absolute contrast in eight VaR, six ES and four score comparisons. The largest change in a q99 reference score contrast is1.48723. Power14→16 had six q99 score sign changes and maximum score-contrast change4.06087, so power16 is a substantial improvement but not universal numerical resolution.

Reference changes are differences between independently refitted scenario distributions. A mechanical coordinate transformation of the same physical scenarios remains an exact identity and has no predictive spread. These two objects must remain distinct. Independent Sobol scrambles in separate bases also must not be claimed to be the same physical scenarios merely because they share a seed.

## Production conditions

1. Use exact finite IN_FHS and its full-distribution mixture for the primary training-reference sensitivity claim; avoid selecting a reference on the exposed evaluation score.
2. Keep internally standardized TCOP and the full-covariance controls as strong comparisons and sensitivity analyses. All must use the same resolution and temporal information convention. Preserve original-law failures as informative diagnostics.
3. Before interpreting a small aggregate q99 score difference, recompute the relevant paired aggregate contrast at power18 and with independent Sobol scrambles on a fixed, explicitly described set of origins. Compare integration variability with the claimed score difference and its date-block sampling uncertainty. Do not use one nested deterministic difference as an MC confidence bound.
4. If an observed contrast is of the same magnitude as its integration variability, state it as numerically unresolved, retain a wider interval, or restrict the conclusion to stable larger effects. A few individual flips do not invalidate every aggregate result, but cannot be silently dropped.
5. Keep fractional empirical ES and its proper-score domain checks. Report q99 calibration uncertainty based on the actual date count and exceedances; simulated scenario count is not the number of independent financial outcomes.

No fixed universal 1% or 2% numerical cutoff guarantees scientific relevance. The operational criterion is that numerical error is small compared with the particular claimed effect and its sampling uncertainty. The exact IN_FHS reference audit meets this integration criterion by construction; its financial performance and contribution must still be assessed against the strong baselines.

## Source and diagnostic caveats

The independent bound check read internal source SHA5db58f689e462be6d8b5f70ee5f4bab936e05a83b94973d2d710b37f307c5a03 and core-model SHA64a987f50b415ebe1e055c84c9419f17db6f7b60590286552c5afa84bd54eea9. Saved power12/14/16 manifests record internal SHAe8e9611557c1a45d0f2f730d98237ed5203a371fbd6b97d18628864f514e0d0b; power18 records the current SHA. Independently checked annual copula parameter JSONs agree exactly between powers16/18 in all nine fitted years, and all non-MC forecasts agree exactly across powers. Retain the individual run hashes; do not claim that the currently read source was literally the older imported source.

The original2015 audit precedes the addition of the relative-identity output column, so those original rows lack that field. Do not replace missing values with measured zeros. In the redesigned script, pool identity diagnostics are currently literal zero placeholders after mixture of already checked member cash arrays; they are not independent measurements. Use the inherited member maximum or label that diagnostic accordingly. This does not change the pooled risk forecasts.
