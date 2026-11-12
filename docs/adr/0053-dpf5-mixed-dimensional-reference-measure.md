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


## Recommended precise reference interpretation — 2026-11-11

**Proposal for DPF5 M=3, D=2 Closeness only:** interpret the ADR 0037
independent ideal-front sample as the pushforward of **independent iid
uniform decision parameters** on the analytical Pareto set, with
`x1,x2 ~ U[0,1]`, `x3=x1`, all trailing `x_i=0.5`.
This corresponds to the explicit mixed-dimensional probability law

```text
mu = (1/3) mu_P + (2/3) mu_D,
mu_P = (DPF5_P)_# Uniform([0,1/3] × [0,1]),
mu_D = (DPF5_D)_# Uniform((1/3,1]).
```

Here `#` denotes pushforward. The weight `2/3` is **not** a guessed
fairness preference; it is the Lebesgue measure of the existing
`x1>1/3` region of the original decision-parameter sampler.
It preserves the one-dimensional D branch rather than erasing it.
The law is *not* uniform on the 2D objective surface; on D,
uniform `x1` is also uniform in physical arc length, whereas P's
conditional law is parameter-uniform with a mild nonconstant
objective-area Jacobian `cos(theta1)`.

**Important scientific distinction:** The **frozen GT is the
reference geometric *support*** used for nearest-neighbor distances,
not a probability sample with the same law. Its archived point
multiplicities cannot be used to estimate `mu`. A new independent
ideal sample must be obtained by continuous direct decision-parameter
draws and **must not pass through objective deduplication**.
The normalized historical GT and `s_K` can be retained *as geometric
reference data* if their support and coverage are independently
verified. No other FAIR reference distributions are reinterpreted
by this DPF5-specific proposal.

**Decision status:** The formula above is the **preferred, explicit
candidate interpretation**, supported by mathematical source
semantics and the quantitative checks below. It is **not yet an
accepted normative baseline release**, and it is not a claim that
other mathematically valid mixtures are wrong. The future versioned
Closeness release must record this measure, its random seeds and
sample size, source/GT SHA256, numerical uncertainty, and paired Q
behavior. Until that review, the production generator's
`HOLD_ARCHIVAL_REFERENCE` guard remains in force.

### 2026-11-11 — FROZEN GT reconstructed exactly; provenance hypothesis checked

- **FIRST FAIL — [CI 43128009733](https://github.com/monacofj/moeabench/actions/runs/43128009733):**
  one new test incorrectly assumed that a 32×32 evenly spaced
  grid includes `x2=1/2`. It does not, so its purported diagonal
  P-meridian point was absent. This was a test-grid parity mistake,
  not a scientific contradiction. Corrected the test by constructing
  an explicit `x1=0.2,x2=0.5` point to prove the boundary
  ambiguity without relying on the grid size.
- **PASS — [CI 43128097424](https://github.com/monacofj/moeabench/actions/runs/43128097424):**
  **6/6 focused tests** and the first read-only archived-GT
  examination. The frozen DPF5 reference in the actual
  v0.14.0 JSON and NPZ is **3,367 unique objective points**,
  all 3,367 non-dominated within the exact archived set,
  lying on the positive unit sphere within numerical precision.
  Merely calling `MOP.pf(n_points=3367)` yields a coarser
  58×58 decision grid and is **not** the archival source.
- **EXACT RECONSTRUCTION — [CI 43128403905](https://github.com/monacofj/moeabench/actions/runs/43128403905):**
  Generated the *actual current* `DPF5(M3,D2).pf(n_points=10000)`,
  normalized it using the **unchanged frozen** package ideal/nadir,
  then applied `np.unique(...,axis=0)`. The resulting set **matches
  the archived 3,367-point GT exactly** (maximum nearest-neighbor
  distance **0**). There are **3,300 distinct P-branch vectors**
  and **67 distinct D-branch vectors**. The original 100×100
  Pareto-set grid had **3,300 P decision samples** but **6,700 D
  samples**, all 6,700 collapsing to those 67 objective vectors.
  Hence the D branch has `0.67` raw decision mass versus
  `67/3367 = 0.0198990198990199` archived distinct-objective
  mass, an exact **33.67-fold** effective probability change.
  This is stronger than geometric resemblance: the reconstructed
  objective *support* is identical. It does **not**, on its own,
  prove the original historical pipeline's authorship or commit
  beyond that exact equivalence.
- **CONCLUSION:** The existing frozen DPF5 GT has demonstrable
  geometrically correct *support*, but its **empirical frequency
  measure** corresponds to unique objectives rather than uniform
  Pareto-set decision samples. The exact support match means
  regeneration of GT is **not automatically required** for the
  geometric nearest-neighbor calculation. The scientific question
  shifts to which *independent ideal sampling probability law* to
  use for finite-GT residual calibration and whether the branch
  coverage/resolution is sufficiently fine.

### 2026-11-11 — Independent continuous PF sampler and numerical stability

- Added `calibration/scripts/study_dpf5_independent_pf_issue78.py`
  and `tests/unit/test_dpf5_independent_pf_issue78.py`.
  Unlike the earlier grid study, this draws independent continuous
  `x1,x2` variates (private `default_rng(seed)`), evaluates
  the same MOP at `g=0`, and **never deduplicates** objective
  points. Structural guards verify nonnegative unit-sphere PF,
  exact D branch formula, independent sample–frozen GT
  nonoverlap, exact archive reproduction and frozen `s_K`.
  No baseline or report is created.
- **PASS — [CI 43128887381](https://github.com/monacofj/moeabench/actions/runs/43128887381):**
  all **9/9 focused tests** passed (six frozen GT/measure,
  three independent source). Tested **2 sample sizes (4,096
  and 16,384)**, **four independent seeds (123–126)** and
  **K=40/100/200** against unchanged frozen GT and normalization.
  D-branch empirical mass ranged `0.659912–0.665039` for
  4,096 samples and `0.662903–0.669861` for 16,384,
  consistent with target `2/3`.
- **K=100 results:** At 4,096 independent samples the
  `ideal_res` ranged **0.052009716–0.053300252** (max
  across-seed span **0.001290537**); at 16,384 samples
  **0.052099946–0.052697600** (span **0.000597654**).
  This is finite-sample variability, not a certified
  confidence interval or an acceptance tolerance. At 16,384
  samples the observed max across-seed epsilon span was
  **0.000349938** for K40 and **0.000857170** for K200.
- **EVIDENCE:** CI artifact
  `issue78-dpf5-frozen-gt-measure-provenance` ID
  **13903946958**, containing both a frozen-GT 100×100
  reconstruction JSON and the independent continuous
  sampling/stability JSON. The previous real two-seed NSGA-II
  Q pilot in this ADR remains valid as a *grid-based three-law
  sensitivity experiment* but does not replace a future
  real-MOEA Q verification under the proposed **continuous**
  sampler. The archived baseline, GT, and official audit
  report remain untouched.


### 2026-11-11 — Continuous reference real Q test passes; candidate law is quantitatively coherent, still NOT published

- **PASS — [GitHub Actions 43129430135](https://github.com/monacofj/moeabench/actions/runs/43129430135):** **9/9** DPF5 source/GT/independent-sampler unit tests passed. Evaluated the original same-population DPF5 M3/D2 NSGA-II trajectories (K100, 300 generations, optimizer seeds 1 and 7) against the frozen clinical baseline and a *single* in-memory candidate with independent **16,384-point continuous iid Pareto-set** reference from seed 123. The candidate uses `ideal_res=0.052697600117886496` and corrects exactly one Closeness ECDF and its physical distance. It has **not** been committed or installed as an official baseline.
- **REAL Q VALUES:** With optimizer seed 1, historic Q-CLOSENESS `0.94042786158407` → continuous-reference `0.9579859119108037` (absolute change `0.017558050326733743`); seed 7 `0.8518897113402047` → `0.867672277859615` (absolute change `0.0157825665194103`). The **maximum absolute difference from historical** was `0.01755805` (≈1.756 percentage points). **Do not describe the single-law "between-laws" spread of zero as an agreement or tolerance:** that field is only meaningful when comparing two or more candidate laws.
- **CROSS-CHECK:** The same seed 1/7 trajectories were previously used for grid-backed sensitivity; Q under the independent continuous law is close to the grid pushforward candidate (`0.9578723105053564` for seed 1; `0.8675769011795502` for seed 7), but these are different ideal-sample constructions. This is descriptive and not a hypothesis acceptance threshold.
- **INVARIANTS:** Under both optimizer seeds, exact P0, Pf, archived GT, resolution `s_K`, and the five other FAIR physical measures and clinical Q-scores were independently asserted unchanged (numeric tolerance 1e-12 for FAIR). The original frozen baseline was restored after each temporary in-memory candidate audit; unchanged official baseline, reference-package and canonical audit report checked by CI.
- **ARTIFACT:** `issue78-dpf5-independent-continuous-PF-real-Q-not-certified`, ID **13904841436**, contains the numerical paired experiments, original/candidate Q and SHA256 source fingerprints. Temporary CI workflow `issue78-dpf5-continuous-q.yml` was removed after verified PASS (commit `087e3fa2`), as were earlier exploratory CI workflows. No new sampling law is automatically active in production.
- **SCIENTIFIC INTERPRETATION:** Independent iid continuous decision sampling resolves the **artifact of deduplication in the ideal reference**, and the frozen GT is geometrically verified to cover both branches precisely as the unique original 100×100 grid. Nevertheless, using a decision-parameter pushforward with `lambda=2/3` rather than a 2D-area or other mixture distribution is a *normative modeling choice*. It is now **explicitly specified, mathematically justified and numerically reproducible**, but formally **PROPOSED**, not a certified scientific release. The failure-profile ECDF of GT-normal blur also has its own empirical GT weighting and must be documented as a distinct reference distribution when reviewing Q interpretation. Do not adopt `ideal_res=0.0526976` as a universal constant; it depends on K, frozen GT, sampling law, sample count and seed. Versioned K-wise Closeness baseline release requires a separate decision.
