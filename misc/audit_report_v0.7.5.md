# Scientific Audit Report - MoeaBench v0.7.5

Audit compares:
1. **Analytical**: Current `exp.optimal()` vs Legacy `opt_front.csv` (Goal: Identity).
2. **Empirical**: Legacy `front.csv` vs Current `exp.optimal()` (Goal: Scientific integrity).

## Analytical Integrity (Formula Check)
| MOP   |   M |          IGD | KS_Match   | Verdict      |
|:------|----:|-------------:|:-----------|:-------------|
| DPF1  |   3 |   0.00127324 | PASSED     | ❌ DIVERGENT |
| DPF1  |   5 |   1.38448    | FAILED     | ❌ DIVERGENT |
| DPF1  |  10 | 244.601      | FAILED     | ❌ DIVERGENT |
| DPF2  |   3 |  14.5545     | FAILED     | ❌ DIVERGENT |
| DPF2  |   5 | 102.606      | FAILED     | ❌ DIVERGENT |
| DPF2  |  10 | 186.68       | FAILED     | ❌ DIVERGENT |
| DPF3  |   3 |   0.0260702  | FAILED     | ❌ DIVERGENT |
| DPF3  |   5 |   0.866033   | FAILED     | ❌ DIVERGENT |
| DPF3  |  10 |   1.27028    | FAILED     | ❌ DIVERGENT |
| DPF4  |   3 |   0.119666   | FAILED     | ❌ DIVERGENT |
| DPF4  |   5 |   1.54827    | FAILED     | ❌ DIVERGENT |
| DPF4  |  10 |   2.34889    | FAILED     | ❌ DIVERGENT |
| DPF5  |   3 |   0.113295   | FAILED     | ❌ DIVERGENT |
| DPF5  |   5 |   0.340946   | FAILED     | ❌ DIVERGENT |
| DTLZ1 |   3 |   0.0172939  | PASSED     | ❌ DIVERGENT |
| DTLZ1 |   5 |   0.143013   | FAILED     | ❌ DIVERGENT |
| DTLZ1 |  10 |   0.378784   | FAILED     | ❌ DIVERGENT |
| DTLZ2 |   3 |   0.0272397  | FAILED     | ❌ DIVERGENT |
| DTLZ2 |   5 |   0.383506   | FAILED     | ❌ DIVERGENT |
| DTLZ2 |  10 |   1.64184    | FAILED     | ❌ DIVERGENT |
| DTLZ3 |   3 |   0.0266198  | FAILED     | ❌ DIVERGENT |
| DTLZ3 |   5 |   0.406102   | FAILED     | ❌ DIVERGENT |
| DTLZ3 |  10 |   0.869381   | FAILED     | ❌ DIVERGENT |
| DTLZ4 |   3 |   0.00196043 | FAILED     | ❌ DIVERGENT |
| DTLZ4 |   5 |   0.00610175 | FAILED     | ❌ DIVERGENT |
| DTLZ4 |  10 |   0.0100582  | FAILED     | ❌ DIVERGENT |
| DTLZ5 |   3 |   0.00197843 | FAILED     | ❌ DIVERGENT |
| DTLZ5 |   5 |   0.230892   | FAILED     | ❌ DIVERGENT |
| DTLZ5 |  10 |   1.23473    | FAILED     | ❌ DIVERGENT |
| DTLZ6 |   3 |   0.0021599  | PASSED     | ❌ DIVERGENT |
| DTLZ6 |   5 |   0.22943    | FAILED     | ❌ DIVERGENT |
| DTLZ6 |  10 |   1.24871    | FAILED     | ❌ DIVERGENT |
| DTLZ7 |   3 |   0.0292397  | FAILED     | ❌ DIVERGENT |
| DTLZ7 |   5 |   0.152023   | FAILED     | ❌ DIVERGENT |
| DTLZ7 |  10 |   0.481037   | PASSED     | ❌ DIVERGENT |
| DTLZ9 |   3 |   1.97529    | FAILED     | ❌ DIVERGENT |
| DTLZ9 |   5 |   2.83396    | FAILED     | ❌ DIVERGENT |
| DTLZ9 |  10 |   4.09328    | FAILED     | ❌ DIVERGENT |

## Empirical Performance (Legacy Front Check)
| MOP   |   M |         IGD | KS_Match   | Verdict   |
|:------|----:|------------:|:-----------|:----------|
| DPF1  |   3 |  0.0388109  | PASSED     | ❌ BROKEN |
| DPF1  |   5 |  0          | FAILED     | ❌ BROKEN |
| DPF1  |  10 |  0          | FAILED     | ❌ BROKEN |
| DPF2  |   3 |  6.48727    | FAILED     | ❌ BROKEN |
| DPF2  |   5 |  9.69219    | FAILED     | ❌ BROKEN |
| DPF2  |  10 | 14.7892     | FAILED     | ❌ BROKEN |
| DPF3  |   3 |  0.0115807  | FAILED     | ❌ BROKEN |
| DPF3  |   5 |  0.924619   | FAILED     | ❌ BROKEN |
| DPF3  |  10 |  0          | FAILED     | ❌ BROKEN |
| DPF4  |   3 |  0.121579   | FAILED     | ❌ BROKEN |
| DPF4  |   5 |  0.525968   | FAILED     | ❌ BROKEN |
| DPF4  |  10 |  0          | FAILED     | ❌ BROKEN |
| DPF5  |   3 |  0.101612   | FAILED     | ❌ BROKEN |
| DPF5  |   5 |  1.40094    | FAILED     | ❌ BROKEN |
| DTLZ1 |   3 |  0.0331287  | FAILED     | ❌ BROKEN |
| DTLZ1 |   5 |  0.759249   | FAILED     | ❌ BROKEN |
| DTLZ1 |  10 |  0          | FAILED     | ❌ BROKEN |
| DTLZ2 |   3 |  0.041919   | FAILED     | ❌ BROKEN |
| DTLZ2 |   5 |  1.01262    | FAILED     | ❌ BROKEN |
| DTLZ2 |  10 |  0.974658   | FAILED     | ❌ BROKEN |
| DTLZ3 |   3 |  0.27829    | FAILED     | ❌ BROKEN |
| DTLZ3 |   5 |  1.01282    | FAILED     | ❌ BROKEN |
| DTLZ3 |  10 |  1.03929    | FAILED     | ❌ BROKEN |
| DTLZ4 |   3 |  0.00958311 | FAILED     | ❌ BROKEN |
| DTLZ4 |   5 |  0          | FAILED     | ❌ BROKEN |
| DTLZ4 |  10 |  0          | FAILED     | ❌ BROKEN |
| DTLZ5 |   3 |  0.0421068  | PASSED     | ❌ BROKEN |
| DTLZ5 |   5 |  0.297856   | FAILED     | ❌ BROKEN |
| DTLZ5 |  10 |  1.96975    | FAILED     | ❌ BROKEN |
| DTLZ6 |   3 |  0.0747886  | FAILED     | ❌ BROKEN |
| DTLZ6 |   5 |  0.954689   | FAILED     | ❌ BROKEN |
| DTLZ6 |  10 |  1.97017    | FAILED     | ❌ BROKEN |
| DTLZ7 |   3 |  0.0463763  | PASSED     | ❌ BROKEN |
| DTLZ7 |   5 |  0.287853   | FAILED     | ❌ BROKEN |
| DTLZ7 |  10 |  1.00687    | FAILED     | ❌ BROKEN |
| DTLZ9 |   3 |  0.410896   | FAILED     | ❌ BROKEN |
| DTLZ9 |   5 |  0.73188    | FAILED     | ❌ BROKEN |
| DTLZ9 |  10 |  1.13787    | FAILED     | ❌ BROKEN |

---
### ⚠️ Divergence Summary
Total Issues Detected: 76

- **DPF1** (M=3, Analytical (Optimal)): IGD=1.27e-03, KS=PASSED
- **DPF1** (M=3, Empirical (Exec Result)): IGD=3.88e-02, KS=PASSED
- **DPF1** (M=5, Analytical (Optimal)): IGD=1.38e+00, KS=FAILED
- **DPF1** (M=5, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DPF1** (M=10, Analytical (Optimal)): IGD=2.45e+02, KS=FAILED
- **DPF1** (M=10, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DPF2** (M=3, Analytical (Optimal)): IGD=1.46e+01, KS=FAILED
- **DPF2** (M=3, Empirical (Exec Result)): IGD=6.49e+00, KS=FAILED
- **DPF2** (M=5, Analytical (Optimal)): IGD=1.03e+02, KS=FAILED
- **DPF2** (M=5, Empirical (Exec Result)): IGD=9.69e+00, KS=FAILED
- **DPF2** (M=10, Analytical (Optimal)): IGD=1.87e+02, KS=FAILED
- **DPF2** (M=10, Empirical (Exec Result)): IGD=1.48e+01, KS=FAILED
- **DPF3** (M=3, Analytical (Optimal)): IGD=2.61e-02, KS=FAILED
- **DPF3** (M=3, Empirical (Exec Result)): IGD=1.16e-02, KS=FAILED
- **DPF3** (M=5, Analytical (Optimal)): IGD=8.66e-01, KS=FAILED
- **DPF3** (M=5, Empirical (Exec Result)): IGD=9.25e-01, KS=FAILED
- **DPF3** (M=10, Analytical (Optimal)): IGD=1.27e+00, KS=FAILED
- **DPF3** (M=10, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DPF4** (M=3, Analytical (Optimal)): IGD=1.20e-01, KS=FAILED
- **DPF4** (M=3, Empirical (Exec Result)): IGD=1.22e-01, KS=FAILED
- **DPF4** (M=5, Analytical (Optimal)): IGD=1.55e+00, KS=FAILED
- **DPF4** (M=5, Empirical (Exec Result)): IGD=5.26e-01, KS=FAILED
- **DPF4** (M=10, Analytical (Optimal)): IGD=2.35e+00, KS=FAILED
- **DPF4** (M=10, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DPF5** (M=3, Analytical (Optimal)): IGD=1.13e-01, KS=FAILED
- **DPF5** (M=3, Empirical (Exec Result)): IGD=1.02e-01, KS=FAILED
- **DPF5** (M=5, Analytical (Optimal)): IGD=3.41e-01, KS=FAILED
- **DPF5** (M=5, Empirical (Exec Result)): IGD=1.40e+00, KS=FAILED
- **DTLZ1** (M=3, Analytical (Optimal)): IGD=1.73e-02, KS=PASSED
- **DTLZ1** (M=3, Empirical (Exec Result)): IGD=3.31e-02, KS=FAILED
- **DTLZ1** (M=5, Analytical (Optimal)): IGD=1.43e-01, KS=FAILED
- **DTLZ1** (M=5, Empirical (Exec Result)): IGD=7.59e-01, KS=FAILED
- **DTLZ1** (M=10, Analytical (Optimal)): IGD=3.79e-01, KS=FAILED
- **DTLZ1** (M=10, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DTLZ2** (M=3, Analytical (Optimal)): IGD=2.72e-02, KS=FAILED
- **DTLZ2** (M=3, Empirical (Exec Result)): IGD=4.19e-02, KS=FAILED
- **DTLZ2** (M=5, Analytical (Optimal)): IGD=3.84e-01, KS=FAILED
- **DTLZ2** (M=5, Empirical (Exec Result)): IGD=1.01e+00, KS=FAILED
- **DTLZ2** (M=10, Analytical (Optimal)): IGD=1.64e+00, KS=FAILED
- **DTLZ2** (M=10, Empirical (Exec Result)): IGD=9.75e-01, KS=FAILED
- **DTLZ3** (M=3, Analytical (Optimal)): IGD=2.66e-02, KS=FAILED
- **DTLZ3** (M=3, Empirical (Exec Result)): IGD=2.78e-01, KS=FAILED
- **DTLZ3** (M=5, Analytical (Optimal)): IGD=4.06e-01, KS=FAILED
- **DTLZ3** (M=5, Empirical (Exec Result)): IGD=1.01e+00, KS=FAILED
- **DTLZ3** (M=10, Analytical (Optimal)): IGD=8.69e-01, KS=FAILED
- **DTLZ3** (M=10, Empirical (Exec Result)): IGD=1.04e+00, KS=FAILED
- **DTLZ4** (M=3, Analytical (Optimal)): IGD=1.96e-03, KS=FAILED
- **DTLZ4** (M=3, Empirical (Exec Result)): IGD=9.58e-03, KS=FAILED
- **DTLZ4** (M=5, Analytical (Optimal)): IGD=6.10e-03, KS=FAILED
- **DTLZ4** (M=5, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DTLZ4** (M=10, Analytical (Optimal)): IGD=1.01e-02, KS=FAILED
- **DTLZ4** (M=10, Empirical (Exec Result)): IGD=0.00e+00, KS=FAILED
- **DTLZ5** (M=3, Analytical (Optimal)): IGD=1.98e-03, KS=FAILED
- **DTLZ5** (M=3, Empirical (Exec Result)): IGD=4.21e-02, KS=PASSED
- **DTLZ5** (M=5, Analytical (Optimal)): IGD=2.31e-01, KS=FAILED
- **DTLZ5** (M=5, Empirical (Exec Result)): IGD=2.98e-01, KS=FAILED
- **DTLZ5** (M=10, Analytical (Optimal)): IGD=1.23e+00, KS=FAILED
- **DTLZ5** (M=10, Empirical (Exec Result)): IGD=1.97e+00, KS=FAILED
- **DTLZ6** (M=3, Analytical (Optimal)): IGD=2.16e-03, KS=PASSED
- **DTLZ6** (M=3, Empirical (Exec Result)): IGD=7.48e-02, KS=FAILED
- **DTLZ6** (M=5, Analytical (Optimal)): IGD=2.29e-01, KS=FAILED
- **DTLZ6** (M=5, Empirical (Exec Result)): IGD=9.55e-01, KS=FAILED
- **DTLZ6** (M=10, Analytical (Optimal)): IGD=1.25e+00, KS=FAILED
- **DTLZ6** (M=10, Empirical (Exec Result)): IGD=1.97e+00, KS=FAILED
- **DTLZ7** (M=3, Analytical (Optimal)): IGD=2.92e-02, KS=FAILED
- **DTLZ7** (M=3, Empirical (Exec Result)): IGD=4.64e-02, KS=PASSED
- **DTLZ7** (M=5, Analytical (Optimal)): IGD=1.52e-01, KS=FAILED
- **DTLZ7** (M=5, Empirical (Exec Result)): IGD=2.88e-01, KS=FAILED
- **DTLZ7** (M=10, Analytical (Optimal)): IGD=4.81e-01, KS=PASSED
- **DTLZ7** (M=10, Empirical (Exec Result)): IGD=1.01e+00, KS=FAILED
- **DTLZ9** (M=3, Analytical (Optimal)): IGD=1.98e+00, KS=FAILED
- **DTLZ9** (M=3, Empirical (Exec Result)): IGD=4.11e-01, KS=FAILED
- **DTLZ9** (M=5, Analytical (Optimal)): IGD=2.83e+00, KS=FAILED
- **DTLZ9** (M=5, Empirical (Exec Result)): IGD=7.32e-01, KS=FAILED
- **DTLZ9** (M=10, Analytical (Optimal)): IGD=4.09e+00, KS=FAILED
- **DTLZ9** (M=10, Empirical (Exec Result)): IGD=1.14e+00, KS=FAILED
