# Completed strong baselines and independent arithmetic audit

The usable formal output is `full_reliable_v2`. The original `full_reliable` directory contains a failure record only: default CSV float parsing triggered two near-zero-score equality checks. The correction uses round-trip CSV parsing without changing the statistical method, model pool, score, history, tolerance or reference-selection rule. No failed predictions have been merged into the final output.

## Actual completed computation

| Output | Years | Annual fitted records | Prediction rows | Runtime (seconds) | Skipped |
|---|---:|---:|---:|---:|---:|
| Direct warmup | 2013–2017 | 20 GARCH fits | 30,672 | 189.05 | 0 |
| Direct prior-year aligned | 2018–2025 | 32 GARCH fits | 49,104 | 344.66 | 0 |
| Reliable six-candidate functional combinations and reference selectors | 2018–2025 | 320 weight records + 16 global reference choices | 114,576 | 64.86 | 0 |

Each direct file includes GARCH_CASH_T_2STEP, HS_CASH500 and HS_CASH1250. Each combination file contains TAYLOR_OPT, TW_RELATIVE, TW_MIN_RIDGE, FUNCTIONAL_EQUAL, PAST_BEST, IN_FHS_REF_PAST_BEST and IN_TCOP_REF_PAST_BEST. Every system has the same 2,046 formal origin dates, four cash tasks, and q95/q99. Genuine historical predictions from 2013–2017 supply completed labels for the 900-observation combination estimation history.

All 52 direct annual GARCH fits converged from all three starts; the minimum fitted degrees of freedom in the formal window is 4.03354. For both TAYLOR_OPT and TW_MIN_RIDGE, all 64 selected annual fits and all 128 starts per method report successful local convergence. All 1,600 ridge validation training fits report successful convergence. These flags do not establish global optimality. The relative-score method selected a refined local candidate 37 times, a grid candidate 25 times, and the exact infinite-lambda best-candidate limit twice. Full weights, penalty trials and optimizer status are retained in the annual JSON files and `optimizer_diagnostics.csv`.

The independently recomputed reference choices are, for 2018 through 2025: IN_FHS selects USD, EUR, USD, JPY, USD, JPY, USD, USD; IN_TCOP selects USD, EUR, USD, JPY, JPY, JPY, USD, USD. Each choice is made from prior q95 scores across all four tasks and 900 completed dates, then reused across both quantiles and all tasks.

## Independent output checks

The audit uses Python's CSV and arithmetic routines without importing either fitting script. It checks all 114,576 forecast keys, source labels, endpoints, score/hit calculation, all functional weight reconstructions, simplex constraints, training-history endpoints, the relative-weight rule, ridge penalty selection and all 16 reference choices. Result: PASS. Maximum prediction reconstruction discrepancies are 1.74e−18 for VaR and 3.47e−18 for ES; emitted FZ0 discrepancy is zero; normalized actual-loss discrepancy against the original outcomes is 2.00e−16; stored optimization objective discrepancy is 1.78e−15. Source-label differences between separately serialized inputs are at most 1.01e−16. The audit does not independently retrain GARCH, establish global optima, or verify upstream joint scenario integration.

Direct first-shock integration uses 4,096 scrambled Sobol draws and an analytic conditional Student-t second-step tail. On the recorded first/middle/last annual origin checks, moving to 16,384 draws has maximum relative q95 VaR/ES changes of 0.0320%/0.0891% and q99 changes of 0.0784%/0.2115% in the formal window. Warmup maximum relative ES change is 0.3188%. These are sampled convergence diagnostics, not universal integration bounds.

## Manuscript method text

We fit a direct Student-t GARCH comparator to normalized basket cash losses, rather than exponentiating Student-t log returns. Annual parameters use the latest 1,250 completed outcomes available at the last reference observation of the preceding year. Because the future-start loss indexed by origin t becomes observable only at t+2, the forecast at t conditions on the latest completed loss at t−2. We update the next conditional variance from that observation, integrate the unknown innovation at t−1, and obtain the conditional t-distribution at t. Its exceedance probability and upper partial moment give a numerical quantile and an analytically integrated conditional expected shortfall. The comparator supplies task-specific cash-loss forecasts and does not supply a joint FX scenario distribution.

The fixed six-candidate pool comprises common-scenario HS1250, coordinate-equivariant Gaussian EWMA, coordinate-equivariant internally normalized EWMA, EUR-reference internally normalized FHS1250, EUR-reference internally normalized t-copula1250, and the direct two-state cash-loss Student-t GARCH. Combination weights are estimated separately for each predefined cash book, reporting currency and quantile using the latest 900 genuine forecasts whose target losses were completed by the preceding year's cutoff. Taylor-style minimum-score combination uses separate simplex weights for VaR and the positive ES-minus-VaR spacing. Relative-score combination applies the same weights to VaR and ES, with weights proportional to the exponential of negative lambda times the sum of historical component FZ0 scores; lambda is optimized using that same prior history, including the equal-weight and limiting best-candidate cases. The regularized minimum-score comparator selects two quadratic penalties on a fixed grid using an ordered 675/225 training/validation split and then refits on all 900 completed observations. These are FX adaptations of the verified scoring rules, not replications of the original equity pools or evidence that generic forecast combination is novel.

The functional combinations do not define one complete joint FX law; conversion identity diagnostics are therefore inapplicable to them. A separate prior-score reference selector selects a single scenario reference per family and year from q95 scores averaged over all four tasks on 900 completed dates, and reuses that reference for all tasks and q99. No current-year outcome or q99 selection enters that decision.

## Empirical caveats and preserved results

The reliable innovation law and pool were adopted following a documented stress-date integration failure in the ordinary copula specification; all 2018–2025 estimates are historical reanalysis, not untouched confirmatory evaluation. Original ordinary FHS/copula engineering output and the earlier 2017 combination pilot remain retained and are excluded from reliable combination estimation. q99 results require the upstream scenario-integration caveats stated by the precision review.

Across the four tasks, formal mean FZ0 is −5.18238/−4.71927 for TAYLOR_OPT, −5.19555/−4.72814 for TW_MIN_RIDGE, −5.17764/−4.70871 for TW_RELATIVE and −5.19282/−4.73927 for FUNCTIONAL_EQUAL at q95/q99 respectively. These arithmetic summaries are not date-clustered inferential conclusions and do not establish calibration. All component outcomes, failed methods and unfavorable results remain available.

Final combination prediction SHA256: `a8debfa38e895ed5b47752109201c4ad4c129e73bd30953f872f040ff4dd8e7d`.
Combination script SHA256 at run: `97ebe0be7490dd9fa6a7cfa2ae22664c6703e6cdf76a2b85fad3be795083868a`.
