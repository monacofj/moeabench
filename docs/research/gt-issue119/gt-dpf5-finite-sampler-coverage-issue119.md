<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF5 finite reference sampler: branch-conditional coverage — issue #119

**Status:** read-only empirical diagnostics, not a reference promotion nor a certified error bound. This uses the independent, mathematically characterized **published DPF5 evaluator** on the **current unchanged production PS decisions**, preserving original decision-space P/D branch labels and original multiplicities. Source: `scripts/audit/gt_dpf5_finite_sampler_issue119.py`, new isolated tests, CI [run #44641941286](https://github.com/monacofj/moeabench/actions/runs/44641941286), **65 new tests PASS**.

Two independent 512-point decision-parameter random probes are generated separately in each strict P and inclusive D stratum using a fixed seed, then mapped through the published equation reconstruction. Nearest-neighbor objective-space distances are computed **within each matching stratum** against exact-unique sampled objective vectors, without reweighting the two strata or selecting any normative mixture probability. Values are finite-probe diagnostics, *not* worst-case supremum, true Hausdorff distance, certified epsilon, or a uniformity score. Cardinality and exact duplicates are reported separately from effective coverage.

## Measurements at requested budget 512

| M,D | Actual PS | Unique published objective vectors | P decisions / P unique | D decisions / D unique | P median / max probe distance | D median / max probe distance |
|---|---:|---:|---:|---:|---:|---:|
| 3,2 | 484 | 169 | 154 / 154 | 330 / 15 | 0.02977 / 0.07854 | 0.01596 / 0.03738 |
| 5,2 | 512 | 512 | 183 / 183 | 329 / 329 | 0.12904 / 0.32301 | 0.00128 / 0.01118 |
| 6,3 | 512 | 512 | 171 / 171 | 341 / 341 | 0.16216 / 0.49741 | 0.02042 / 0.09446 |
| 6,4 | 512 | 512 | 171 / 171 | 341 / 341 | 0.16216 / 0.49741 | 0.04946 / 0.18339 |

At requested 128, M3/D2 produces 121 PS rows and 51 distinct published objective vectors (P44, D7), with 70 exact duplicates in D. For M5/D2 requested 128, 47 P and 81 D decisions yield 128 distinct published objectives; P median/max probe distances are 0.19280 / 0.50507 whereas D median/max are 0.00464 / 0.05498. This strongly suggests an inadequately dense P patch under the current **allocation**, but no calibrated quantitative quality threshold has been adopted.

**Important boundary-specific reinterpretation of earlier M3 numbers:** The earlier 190 distinct vectors for requested 512 refer to the **current implemented evaluator**, which assigns exact x1=1/3 to P. Under the **published** convention P uses strict x1<1/3 and equality belongs to D: the same 22×22 grid has P154, D330, and only **169 distinct published objective vectors**. This difference is expected from changing the branch definition, **not** a contradiction between datasets. For the same 484 decision rows, the current implementation has 190 unique objective vectors; the published reconstruction has 169. At requested 128, no grid x1 equals 1/3 (11-point side), so both have 51 unique.

## Interpretation

1. **Membership:** the previous analytic proof shows all these *published* g=0 reference points are Pareto-optimal, but it does not guarantee that they cover the entire continuous PF well at finite n.
2. **Requested count:** for M3 current grid, n=512 produces 484 actual PS and n=128 gives 121; the printed n is an input target rather than an achieved count. Exact uniqueness is another independent dimension.
3. **Mixed-dimensional support:** the P portion has dimension M−1, while D has dimension D−1. Cardinality allocation by uniform decision coordinates is not equivalent to equal geometric covering accuracy. There is **no mathematically compulsory uniform P/D probability mixture**; ADR 0053 is only proposed and includes historical FAIR-specific research which is excluded from the active production fix.
4. **Extremes:** the script exposes coordinate-wise minimum/maximum and proximity to 0 and 1 for every M; those diagnostics are not a proof that all essential component boundary curves and branch interfaces are represented.
5. **Sampling rule needed:** future candidate generators should report branch-specific coverage and intended geometry, preserve deterministic reproducibility, and expose the actual returned and unique counts. Any balancing or mixture weight is a scientific decision, not something inferred from these observed distances.

## Next independent scientific gate

Build mathematically justified branch-specific covering controls that include exact component extrema and treat P as an (M−1)-dimensional angular patch and D as a (D−1)-dimensional angular component. At fixed **effective unique objective** budgets, compare finite independent probe coverage, p95 and extreme preservation at bounded cost; use multiple probe seeds and report uncertainty before selecting numerical thresholds. That is still an experimental reference proposal, not permission to modify `ps()`, `pf()`, `calibrate()`, accepted ADRs, original regressions, sidecars, or frozen references.
