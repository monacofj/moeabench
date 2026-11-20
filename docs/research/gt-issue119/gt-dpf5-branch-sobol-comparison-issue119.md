<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF5 branchwise equal-effective-budget Sobol comparison — issue #119

**Research-only; candidate not selected.** A nonmutating comparator was introduced in `scripts/audit/gt_dpf5_branch_sampler_compare_issue119.py`, with 5 new focused tests and an isolated CI artifact. [Actions #44642578113](https://github.com/monacofj/moeabench/actions/runs/44642578113) PASSED **70 focused tests**. No production sampler, mathematical evaluator, archived GT, existing regression or ADR was changed.

## Controlled design

For each DPF5 (M,D)=(3,2),(5,2),(6,3),(6,4) and requested budget 128/512, take actual production PS decisions, evaluate them with the independent **published** DPF5 equation reconstruction, and determine **exact distinct objective counts by branch**. The counterfactual uses a deterministic scrambled Sobol sequence *within each branch* plus limited first-angle/other-angle boundary anchors. For P it samples M−1 angular parameters; for D it samples only D−1 essential angular parameters. It returns **exactly the same number of distinct objectives in each branch** as the incumbent published-evaluated sample (e.g. M5,D2,n512 uses P183/D329, not a proposed new allocation). Both candidates are compared against **three** independently generated, identical 512-point random probe sets *per branch*, with separate median, p95 and observed maximum nearest-objective distances.

This is a paired fixed-budget study, **not** a selection of a P/D mixture measure, an objective-space uniformity proof, an analytic worst-case bound, or a production GT proposal. The extra anchors preserve certain corners and first-angle endpoints; they do not certify the entire boundary topology or every coordinate extreme. The script records candidate-generation elapsed wall time for descriptive diagnostics, not a controlled performance benchmark.

## Empirical median coverage improvements

Numbers below are `median_nearest_distance(incumbent) − median_nearest_distance(Sobol)`; **positive is better for Sobol**. Values are per independent probe seed (201,202,203).

| M,D | Requested | P improvements | D improvements |
|---|---:|---|---|
| 3,2 | 128 | +.00262, −.00102, +.00247 | −.00423, −.00839, −.00731 |
| 3,2 | 512 | −.00156, −.00110, −.00012 | −.00072, −.00096, +.00025 |
| 5,2 | 128 | −.00755, −.00685, +.00014 | +.00029, +.00072, +.00070 |
| 5,2 | 512 | +.00435, +.00546, +.00815 | +.00036, +.00034, +.00038 |
| 6,3 | 128 | −.01026, −.02384, −.01700 | +.00294, +.00386, +.00189 |
| 6,3 | 512 | +.00281, −.00250, +.00244 | +.00204, +.00201, +.00081 |
| 6,4 | 128 | −.01026, −.02384, −.01700 | +.00021, −.00123, +.00218 |
| 6,4 | 512 | +.00281, −.00250, +.00244 | +.00169, +.00318, +.00069 |

The three probe-seed variations indicate genuine scenario sensitivity but are not a statistical confidence interval over alternative sampler-generation seeds; the scrambled Sobol candidate seed is fixed in this run. The stored full JSON includes p95/max and costs; median-only summaries cannot justify accepting a sampler.

**Decision:** a single uniform/Sobol replacement is not scientifically supported by these measurements. In particular, the mixed-dimensional P/D coverage disparity documented previously cannot be corrected simply by changing the *within-branch* sampler while preserving the same branch allocations; the branch budget and the target coverage objective remain separate scientific decisions. A follow-up should compare geometry-specific candidate coverage designs and budget allocations using multiple **sampler** seeds (not only probe seeds), hold total effective objective budgets constant, report per-branch worst-case probes/extremes and wall-time at a reproducible scale, and avoid claiming a unique branch probability measure.

The global Pareto-membership theorem still applies only to the reconstructed **published** DPF5, not to the current evaluator for all (M,D). PR #139 remains Draft. Production `ps()/pf()/calibrate()`, existing regressions, accepted ADRs, frozen GT and baseline/certification archives remain unchanged.
