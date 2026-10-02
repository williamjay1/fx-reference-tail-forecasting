**Table 1. Forecast scores and descriptive breach rates.**

| Forecast system | Score 95% | Score 99% | Hits 95% | Hits 99% |
| --- | --- | --- | --- | --- |
| Common HS1250 | -5.09302 | -4.61463 | 4.67% | 1.14% |
| Ordinary FHS1250, EUR fit | -5.16904 | -4.68451 | 4.74% | 1.00% |
| IN-FHS, EUR fit | -5.17835 | -4.72156 | 4.58% | 1.14% |
| IN-FHS reference mixture | -5.17798 | -4.72930 | 4.25% | 1.00% |
| Gaussian EWMA | -5.18140 | -4.59084 | 5.17% | 2.09% |
| Internal elliptical EWMA | -5.18215 | -4.59770 | 5.08% | 2.05% |
| Dynamic scalar BEKK, internal radial | -5.14745 | -4.64087 | 3.96% | 1.42% |
| Direct cash GARCH | -5.18043 | -4.70620 | 5.12% | 1.30% |
| Taylor optimized | -5.18238 | -4.71927 | 4.83% | 1.43% |
| Taylor–Wang relative | -5.17764 | -4.70871 | 4.81% | 1.19% |
| Taylor–Wang ridge | -5.19555 | -4.72814 | 4.50% | 1.36% |
| Equal functional mean | -5.19282 | -4.73927 | 4.39% | 1.21% |
| Prior-selected IN-FHS reference | -5.16555 | -4.69449 | 4.09% | 1.27% |

**Table 2. Reference-specific VaR and ES forecast dispersion.**

| Fixed cash task | VaR span 95% | ES span 95% | VaR span 99% | ES span 99% |
| --- | --- | --- | --- | --- |
| Assets, EUR | 27.32% | 26.11% | 26.50% | 29.81% |
| Assets, USD | 27.41% | 26.34% | 26.43% | 30.12% |
| Flows, EUR | 22.69% | 22.54% | 24.83% | 26.36% |
| Flows, USD | 22.70% | 22.84% | 25.51% | 26.87% |

**Table 3. Mixture-minus-comparator score differences.**

| Comparator | Delta 95% | 95% interval | Delta 99% | 95% interval |
| --- | --- | --- | --- | --- |
| IN-FHS, EUR fit | 0.0004 | [-0.0193, 0.0200] | -0.0077 | [-0.0508, 0.0354] |
| Ordinary FHS1250, EUR fit | -0.0089 | [-0.0323, 0.0145] | -0.0448 | [-0.1014, 0.0119] |
| Gaussian EWMA | 0.0034 | [-0.0384, 0.0453] | -0.1385 | [-0.2844, 0.0075] |
| Internal elliptical EWMA | 0.0042 | [-0.0369, 0.0452] | -0.1316 | [-0.2720, 0.0088] |
| Taylor optimized | 0.0044 | [-0.0168, 0.0256] | -0.0100 | [-0.0699, 0.0499] |

**Table 4. Paired scores against the dynamic joint control.**

| Tail confidence | Block length | Mixture FZ0 | BEKK FZ0 | Mixture minus BEKK | Simultaneous 95% interval, two-tail family |
| --- | --- | --- | --- | --- | --- |
| 95% | 20 | -5.17798 | -5.14745 | -0.03053 | [-0.05402, -0.00704] |
| 95% | 60 | -5.17798 | -5.14745 | -0.03053 | [-0.05892, -0.00214] |
| 99% | 20 | -4.72930 | -4.64087 | -0.08843 | [-0.15554, -0.02132] |
| 99% | 60 | -4.72930 | -4.64087 | -0.08843 | [-0.16342, -0.01344] |

**Table 5. VaR breach counts by cash target.**

| Forecast system | Tail | Assets EUR | Assets USD | Flows EUR | Flows USD |
| --- | --- | --- | --- | --- | --- |
| IN-FHS, EUR fit | 95% | 81 | 81 | 106 | 107 |
| IN-FHS, EUR fit | 99% | 17 | 17 | 30 | 29 |
| IN-FHS reference mixture | 95% | 71 | 72 | 102 | 103 |
| IN-FHS reference mixture | 99% | 14 | 12 | 28 | 28 |
| Gaussian EWMA | 95% | 91 | 90 | 121 | 121 |
| Gaussian EWMA | 99% | 28 | 26 | 58 | 59 |
| Internal elliptical EWMA | 95% | 89 | 90 | 119 | 118 |
| Internal elliptical EWMA | 99% | 26 | 26 | 57 | 59 |
| Direct cash GARCH | 95% | 86 | 87 | 123 | 123 |
| Direct cash GARCH | 99% | 14 | 16 | 38 | 38 |
| Dynamic scalar BEKK, internal radial | 95% | 72 | 70 | 91 | 91 |
| Taylor optimized | 95% | 79 | 81 | 117 | 118 |
| Dynamic scalar BEKK, internal radial | 99% | 19 | 17 | 40 | 40 |
| Taylor optimized | 99% | 14 | 14 | 44 | 45 |
| Taylor–Wang ridge | 95% | 73 | 74 | 110 | 111 |
| Taylor–Wang ridge | 99% | 17 | 17 | 38 | 39 |

**Table 6. Longer-horizon mixture breach counts.**

| Cash task | h=5: breaches 95% / 99% | h=20: breaches 95% / 99% |
| --- | --- | --- |
| Assets, EUR | 78 / 12 | 80 / 6 |
| Assets, USD | 78 / 12 | 82 / 7 |
| Flows, EUR | 101 / 32 | 149 / 51 |
| Flows, USD | 100 / 33 | 148 / 53 |
| Financed treasury, EUR | 85 / 28 | 107 / 40 |
| Financed treasury, USD | 85 / 26 | 108 / 40 |
| Unfinanced treasury, EUR | 85 / 28 | 107 / 40 |
| Unfinanced treasury, USD | 80 / 31 | 77 / 23 |

**Table 7. Unfinanced treasury forecasts by reporting currency.**

| Holding intervals | Mean mixture ES95, EUR / N | Mean mixture ES95, USD / N_USD | Mean absolute realized EUR/USD normalized loss difference |
| --- | --- | --- | --- |
| 1 | 0.006115 | 0.003449 | 0.002030 |
| 5 | 0.013595 | 0.007819 | 0.004559 |
| 20 | 0.022599 | 0.015020 | 0.008760 |

**Table 8. Mixture breach counts in the 2026 extension.**

| Cash book | EUR: 95% / 99% | USD: 95% / 99% |
| --- | --- | --- |
| Funded assets | 2 / 1 | 2 / 1 |
| Net cash flows | 7 / 5 | 8 / 5 |
| Unfinanced treasury | 9 / 2 | 9 / 3 |

