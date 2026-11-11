# ADR 0053 — Probability measure on the mixed-dimensional DPF5 Pareto front

**Status:** Proposed; *NO sampling measure adopted or baseline authorized*  
**Date:** 2026-11-11  
**Issue:** [#78](https://github.com/monacofj/moeabench/issues/78)  
**Source audit:** [issue #75](https://github.com/monacofj/moeabench/issues/75), [ADR 0051](0051-fair-conformance-audit.md)  
**Research branch:** `research/78-dpf5-measure-sensitivity`

## Problem: geometry does not determine a single uniform law

In `moeabench/mops/DPF5.py`, with **M=3, D=2, g=0** on the
analytical Pareto set, let
`theta_1=(pi/2)*x_1`, `theta_2=(pi/2)*x_2`.
The actual conditional objective map is

```text
P (0 <= x1 <= 1/3):
  F=(cos(theta1)*cos(theta2), cos(theta1)*sin(theta2), sin(theta1))
  theta1 in [0,pi/6], theta2 in [0,pi/2]

D (1/3 < x1 <= 1):
  F=(cos(theta1)/sqrt(2), cos(theta1)/sqrt(2), sin(theta1))
  theta1 in (pi/6,pi/2], independent of x2.
```

Both branches are on the positive unit sphere. **P** is a 2D patch;
**D** is a **1D curve**. There is no canonical "uniform point" of
mixed-dimensional geometry without specifying a *mixture measure*.
One defensible family is
`mu_lambda = (1-lambda)*mu_P + lambda*mu_D`, where `lambda`
and both **conditional distributions** must be defined explicitly.
This is **not a matter that can be decided from nearest-GT distance**.

The present `MOP.ps()` generates an approximately uniform decision
grid in `(x1,x2)`. Its **pushforward** preserves multiplicity:
the measure of D is approximately `2/3`, because it occupies
`x1>1/3`. However, `independent_analytical_reference()`
applies `np.unique(objectives, axis=0)`. Under an `n x n` grid,
the P branch contributes O(n²) distinct objective points whereas
D contributes only O(n); hence **the distinct-objective measure's
probability of D tends to zero as n grows**.

Uniform **2D** Hausdorff area on the whole objective image assigns
zero mass to D (the 1D branch). It is not a neutral default; it
implicitly decides `lambda=0`. Similarly, an **equal 50:50 branch**
mixture is a useful third sensitivity case, **not** a mathematically
required law. Nor does choosing a uniform parameter density mean
uniform physical arc length/area.

## Measurements: identical geometry, identical frozen GT

`calibration/scripts/study_dpf5_branch_measure_issue78.py` records
the same analytical points/decision-grid, source/GT SHA256, frozen
normalization and exact `s_K` under each of three **counterfactual**
weights:

1. **Decision pushforward:** equal weight for each original decision
   grid row, *including* repeated objective rows on D.
2. **Distinct objective vectors:** equal weight for each unique
   objective row, intentionally reproducing the effective
   deduplication change.
3. **Equal branch mass:** 50% P/50% D with conditional equal weights
   within each unique-objective set (illustration, not a proposal).

All three use the same empirical **lower weighted median** of
nearest-neighbor distances to the *unchanged normalized* GT. There
is no Monte Carlo noise confounding the three laws.

[GitHub Actions #43089992262](https://github.com/monacofj/moeabench/actions/runs/43089992262)
**PASSED** (3 numerical unit tests, structural and geometric guards),
artifact `issue78-dpf5-three-law-paired-sensitivity` ID
**13887318534**. Results:

| Grid | Raw mass of D | Distinct-objective mass of D | Median dist, raw | Median dist, distinct | Median dist, equal branches |
|---|---:|---:|---:|---:|---:|
| 32×32 | 65.625% | 5.630% | 0.004606437 | 0.005963568 | 0.005118262 |
| 64×64 | 65.625% | 2.897% | 0.004533319 | 0.006401565 | 0.004887748 |
| 128×128 | 66.406% | 1.521% | 0.004751715 | 0.006192380 | 0.005159516 |

**At K=100, using frozen `s_K`:**

| Grid | epsilon raw decision | epsilon unique objectives | epsilon 50/50 | Max-min epsilon |
|---|---:|---:|---:|---:|
| 32×32 | 0.05070086 | 0.06563815 | 0.05633427 | 0.01493729 |
| 64×64 | 0.04989608 | 0.07045898 | 0.05379711 | 0.02056290 |
| 128×128 | 0.05229986 | 0.06815658 | 0.05678834 | 0.01585672 |

At K=40, maximum epsilon spreads across the tested grids were
0.008746, 0.012040, 0.009284; at K=200 they were
0.021423, 0.029492, 0.022742, respectively. Changes do **not**
vanish under denser grids. They are due to different reference
probability measures, not a PRNG seed.

The frozen baseline, GT, and `calibration/audit_report.html` were
never overwritten or retagged; software version is not a reason to
recalibrate. A real paired NSGA-II Q-score sensitivity check is
part of this research branch and its result must be logged before
any recommendation to promote a candidate calibration.

## Scientific decision required

**Do not adopt a law silently.** To publish DPF5 Closeness under
ADR 0037, an explicit scientific choice must specify:

- Mixture weight `lambda` between the 2D P patch and the 1D D curve;
- What "uniform" means **within each branch** (decision parameters,
  objective area/arc length, or other experimentally motivated law);
- Whether the frozen GT supports the proposed source measure with
  known biases and adequate coverage (and whether a new GT reference
  is required);
- An independent analytic PF sample generator **without automatic
  `np.unique` reweighting**, deterministic provenance, and paired
  Q-score sensitivity on real final fronts; and
- A separate, versioned, reviewed Closeness-only baseline revision.
  DPF5 remains **HOLD_ARCHIVAL_REFERENCE** until this decision.

**Conservative working preference (subject to review):** preserve the
existing `ps()`-specified *decision-parameter pushforward law* as
the baseline interpretation **if and only if** ADR 0037 is clarified
to intend sampling uniformly in decision Pareto-set parameters.
That would give D nonzero mass rather than erasing it by dedup.
This is an explicit interpretation, **not** a claim that it is the
unique fair or physically uniform objective-space measure.

## Chronology

- **2026-11-11:** Inspected the exact DPF5 M3 objective formulas and
  source grid; derived the 2D/1D branches and the asymptotic
  `np.unique` weighting pathology. Chose to compare laws on
  *identical* points and frozen GT, never replacing official data.
- **2026-11-11:** Added a standalone non-destructive study, focused
  unit tests and temporary CI, verified [Actions #43089992262](https://github.com/monacofj/moeabench/actions/runs/43089992262)
  **PASSED**. Numerical results above. No normative law adopted.
- **2026-11-11:** Added real NSGA-II paired Q-sensitivity script that
  uses *in-memory* counterfactual Closeness-only baselines and
  checks exact unchanged P0/Pf/GT/`s_K` plus five other FAIR/Q
  families. CI results pending. Append both failures and successes.
- **2026-11-11 — FAILED, engineering-only:** [GitHub Actions #43090638106](https://github.com/monacofj/moeabench/actions/runs/43090638106)
  passed all three sensitivity tests and the three-law GT calculation,
  but the real-NSGA-II script aborted on direct CLI invocation with
  `ModuleNotFoundError: No module named 'calibration'`. This was
  due to the repository root not being present on `sys.path` when a
  script under `calibration/scripts` is executed by filename,
  **not** a numerical or scientific contradiction.
  Fixed by inserting the resolved repository root before importing the
  sibling research modules, commit `4493a8c1`. The resumed CI must
  independently confirm real-MOEA behavior; never claim the failed run
  passed the Q experiment.
- **2026-11-11:** Opened isolated [Draft PR #80](https://github.com/monacofj/moeabench/pull/80),
  indexed this proposed ADR 0053. **No merge planned until a
  sampling-law scientific decision**; source code, archived GT,
  baselines and canonical report remain unmodified.


### 2026-11-11 — REAL paired NSGA-II Q sensitivity PASS after CLI repair

- **PASS — [GitHub Actions #43091096784](https://github.com/monacofj/moeabench/actions/runs/43091096784):** Repeated 4/4 numerical/record-preservation tests, the complete three-law sensitivity (32×32, 64×64, 128×128; K40/100/200), and two real DPF5 M3/D2 NSGA-II experiments with K=100, 300 generations and seeds 1 and 7. The CLI import failure recorded above was successfully fixed.
- **PAIRED EXPERIMENT:** For each seed, `mb.clinic.audit(exp)` was run on the same original archive and three in-memory Closeness-only counterfactual baseline dictionaries; exact SHA256 P0/Pf/GT fingerprints, `s_K` and original calibration metadata remained unchanged. Explicit tests compared all five other FAIR physical values and Q-scores at 1e-12; all unchanged. The original frozen baseline was restored after every counterfactual audit, which was independently verified. **No baseline file, GT, report, or production source changed.**
- **Seed 1:** historical Q-CLOSENESS `0.94042786158407`; parameter-grid pushforward `0.9578723105053564`; unique objectives `0.9618996429181158`; illustrative equal-branch mixture `0.9591212133397684`. Maximum counterfactual law-to-law Q shift `0.004027332412759432`.
- **Seed 7:** historical Q-CLOSENESS `0.8518897113402047`; parameter-grid pushforward `0.8675769011795502`; unique objectives `0.8709905954517703`; illustrative equal-branch mixture `0.8686352688132152`. Maximum counterfactual law-to-law Q shift `0.003413694272220047`.
- **INTERPRETATION:** These two genuine optimization trajectories are informative and demonstrate that a measure choice changes the observable Q-score even when all underlying P0/Pf/GT and all five other FAIR values/Q are unchanged. The measured max difference `0.00402733` is **descriptive**, not an accepted tolerance or proof of scientific equivalence. The comparison with the historical Q is larger, because the archived `ideal_res=0` is a different (flawed) calibration, not a fourth independently justified sampling law. Only two optimizer seeds and one K were checked; no statistical generalization or certification follows.
- **ARTIFACT:** `issue78-dpf5-three-law-paired-sensitivity`, ID **13889201253**, contains JSON for three grid/K scenarios and both paired real-MOEA trajectories. All test and frozen-file-preservation gates passed. **No law was adopted and no baseline was regenerated or installed.** Next decision remains explicit mixed-dimensional measure `mu_lambda` and archival GT compatibility.
