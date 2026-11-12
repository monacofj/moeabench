# ADR 0054 — One-sided dominated-side GT half-normal blur

**Status:** Proposed (implementation on review branch; no baseline adoption)  
**Date:** 2026-11-12  
**Issue:** [#83](https://github.com/monacofj/moeabench/issues/83)  
**Scientific calibration owner:** [#81](https://github.com/monacofj/moeabench/issues/81)  
**Branch:** `fix/83-outward-half-normal-blur`

## Context: intended one-sided error versus the actual code

[ADR 0036](0036-half-normal-closeness-v0.12.0.md) prescribed the
half-normal Closeness null as a *one-sided outward perturbation* of
the reference Pareto manifold. The current, later
`calibration/scripts/generate_baselines.py` implementation did
generate a magnitude `abs(N(0,sigma²))`, **but also sampled a random
sign ±1**, and accepted negative displacements if the resulting
objective coordinates remained nonnegative. Retrying an invalid
negative result by flipping sign did not enforce a one-sided law.
This was a semantic defect, not a scientific decision superseding
the original half-normal null.

The original normal estimator chose sign from
`dot(normal, GT[i])>0`. This may happen to produce the dominated-side
direction for some smooth convex benchmark fronts, but it is **not
a general dominance guarantee** on nonconvex, discontinuous or
mixed-dimensional fronts. Relative position to the origin and Pareto
dominance are different geometric relationships.

## Decision and proof

For minimization, if a true Pareto point is `g` and the blur is
`b=g+t d` with `t>0` and
`d_i>=0 for every objective i`, `d!=0`, then **g dominates b**:
every objective at `b` is no better, and at least one strictly worse.
Thus `b` lies in the *dominated*, rather than infeasible improving,
side of the frontier. This is an **unambiguous sufficient condition**
for the intended direction even if the front is disconnected or
nonconvex. It is not a claim that the objective vector `b` has a
feasible inverse decision in the MOP.

Continue estimating the *unoriented* local PCA eigenvector using the
historical `k_pca=clip(round(sqrt(N)),10,30)` neighborhood policy.
For each unit eigenvector `n`, consider `n` and `-n`, project both
onto the closed nonnegative orthant by zeroing negative components,
and select the projection with the larger Euclidean norm. Normalize
the selected vector to a unit direction `d`. Equivalently this
selects the closest nonnegative unit direction to either
unoriented PCA normal. For exactly/numerically zero normals use the
positive unit diagonal.

**Lower-dimensional Pareto fronts require one refinement.** A
one-dimensional curve embedded in three objectives has a
two-dimensional normal *plane*, not a unique normal vector. Its
smallest local covariance eigenvector is arbitrary in that plane and
can have mixed signs even when another vector in the same plane lies
entirely on the dominated side. The implementation now detects a
clear PCA low-variance eigensubspace using a **2% relative eigenvalue
threshold** against the largest local eigenvalue (keeping at least
one tangent axis), projects the nonnegative unit diagonal into
this normal subspace, and uses that direction when nonnegative.
Only otherwise does it project the individual lowest eigenvector
onto the positive dominance cone. This is a deterministic
**heuristic intrinsic-rank cutoff**, not a global proof of the
manifold's dimension. The read-only audit reports the frequency
of true-normal-subspace recovery versus the fallback.

Sample `t = abs(N(0,sigma²))`, and set `b=g+t*d`. **No independent
random sign** and no retry that can cross into improving directions.
The generator checks finite inputs, positive finite sigma, exact
objective-shape compatibility and the domination inequality.

**Limit/important scientific caveat:** In degenerate/boundary regions
the PCA eigenvector has no sign compatible with the full positive
dominance cone (mixed-sign components). The projection changes its
direction and is **not a mathematically exact normal vector** there.
The resulting distribution remains a valid one-sided
dominated-side geometric error null, but its directional behavior
must be described as a projected local PCA surrogate, not an exact
surface-normal distribution. The 14-MOP audit below quantifies
projection frequency so that any scientifically problematic cases
remain explicit for issue #81. No generalized global feasible-region
oracle has been assumed.

## Intensity policy unchanged

The original calibrated target is retained **without adjustment**:

```text
median(d(blurred_GT, finite_GT) / s_K) ~= 2
s_K = median nearest-neighbor distance inside a fixed-seed
      K-point farthest-point-sampling subset of that GT
```

The `calibrate_sigma` procedure still starts at
`sigma=2*s_K`, attempts at most six multiplicative adjustments and
targets a relative median tolerance of 5%. The changed direction
and half-normal sign correctly imply that the **previous** frozen
`sigma` and failure ECDF must *not be silently reinterpreted* as
calibrated by the new generator; new calibration is a later
separately reviewed scientific publication in issue #81.
The old GT, v0.14.0 baseline, independent `ideal_res`, and canonical
`calibration/audit_report.html` remain frozen.

## Scientific meaning

The half-blur is an **algorithm-independent, geometry-controlled
reference for insufficient Closeness**, not a distribution of
actual final MOEA populations. It need not be a feasible image of
decision-space coordinates for this chosen null interpretation.
Using a budget-limited NSGA-II as the baseline would encode that
algorithm's dynamics and operator/budget biases in a common
cross-algorithm quality scale, and is therefore not adopted.
Such optimizer trajectories may still validate the null externally
and will not define new baseline values.

## Chronological audit log

### 2026-11-12 — Original defect reproduced in source; fix proposed

- Read actual `generate_baselines.py`, ADR 0036, ADR 0037 and
  scientific DPF5 ADR 0053. Confirmed the old random-sign code
  could accept `p_i - t*n_i` with nonnegative objective coordinates
  even if `n_i` pointed to the dominated side. This violates the
  one-sided requirement and can create an improving-side sample.
- Created [issue #83](https://github.com/monacofj/moeabench/issues/83)
  and isolated `fix/83-outward-half-normal-blur` branch off `main`.
- Implemented `dominance_outward_direction()`, updated
  `estimate_gt_normals()` to use the explicit dominance cone, and
  replaced `generate_half_normal_blur()` random-sign/retry logic
  with strictly positive half-normal magnitudes along nonnegative
  directions. No other FAIR, MOP or runtime metric changed.
- Added focused regression tests covering an old-code sign
  counterexample, strict minimization domination in every draw,
  smooth DTLZ2, disconnected DTLZ7, mixed-dimensional DPF5,
  global RNG isolation, degenerate and mixed-sign PCA vectors,
  input validation, small GT sets and preserved sigma target.
- Added an independently reproducible **14-MOP frozen-GT read-only
  geometry audit** with a bounded deterministic 512-row subsample
  per MOP, exact frozen GT SHA, the degree of PCA projection,
  dominated-side fractions and old `2*s_K` calibration residual.
  The audit is explicitly **NONCERTIFIED**, not a replacement
  for the full archived 10K GT or publication baseline.
- Temporary CI runs and exact numeric results will be added to
  this append-only log after their actual completion, including
  errors and necessary corrections. No official baseline, GT or
  audit report is modified.

## Release gates

1. Prove every generated blur point is weakly dominated by its
   own GT origin, and strictly worse for positive magnitudes.
2. Audit actual registered GTs from all 14 MOPs and quantify
   how often the PCA normal requires projection, reporting
   any excessive deviation rather than hiding it.
3. Preserve the `median/s_K ~=2` six-iteration controller and
   test no regression to historical FAIR computations.
4. Run focused tests, complete unit regression and representative
   integration tests with an immutable official baseline/GT/report
   check. Remove temporary workflow after green CI.
5. Merge the **source correction** only after review; issue #81
   remains solely responsible for **scientific baseline
   recalibration**, including DPF5's ideal/negative measure choices.


### 2026-11-12 — First full source audit: PASS, important normal-fidelity limitation discovered

- **PASS — [CI #43235004493](https://github.com/monacofj/moeabench/actions/runs/43235004493):** 13/13 focused and 335/335 full unit tests passed; bounded, 512-row per-MOP tests on all **14 actual frozen MOP GT packages** found **100%** of generated blur points weakly dominated by their own GT origin, and all strictly worse in at least one objective. The unchanged `2s_K` sigma controller fitted within its original ±5% target for all fourteen bounded samples. No frozen baseline/GT or official `audit_report` changed. Artifact `issue83-outward-half-normal-14mop-not-certified` ID **13952560461**.
- **SCIENTIFIC/GEOMETRIC LIMITATION DISCOVERED (not hidden):** The first implementation projected a *single* PCA lowest eigenvector into the nonnegative dominance cone. That was sufficient for domination but often abandoned the *true local normal subspace*. Among 512 sampled points of each archived GT, the original raw lowest eigenvector was mixed-sign at **100% of DTLZ5, DTLZ6 and DTLZ9** locations, **67.77% DPF1**, **97.85% DPF2**, and **59.38% DTLZ7**. The old raw GTs of DPF2/DTLZ7 are independently known to include dominated points; they are unsuitable for scientific calibration. Even for valid curved 1D PFs the smallest PCA normal is arbitrary because the null space is multidimensional. A naive orthant projection would misrepresent the actual local normal.
- **REPAIR OF THE REPAIR:** Added `dominance_outward_pca_direction` to select a *nonnegative vector of the true low-variance PCA normal subspace* when the covariance spectrum indicates a rank-deficient manifold (eigenvalues ≤2% of the largest eigenvalue, keeping at least one tangent dimension). Only ambiguous/non-Pareto cases fall back to the conservative nonnegative cone projection. Added focused rank-1 curve and rank-2 surface spectral proof tests. This correction is still algorithm-independent and retains the original sigma controller unchanged.

### 2026-11-12 — Normal-subspace refinement PASSED, 14-MOP sampled evidence

- **PASS — [CI #43235806488](https://github.com/monacofj/moeabench/actions/runs/43235806488):** **14/14 focused tests and 336/336 complete unit tests**, all fourteen 512-point frozen GT subsamples, all strict domination checks and old 2-unit-resolution sigma tolerance checks. No official scientific data changes. Audit evidence artifact `issue83-outward-half-normal-14mop-not-certified` ID **13953095756**.
- **Geometrical fidelity improved:** For all **512 sampled points each** of DTLZ5, DTLZ6 and DTLZ9, the low-dimensional normal-subspace criterion recovered a genuine normal direction in the dominated cone (subspace alignment **1.0**), instead of forcing the arbitrary raw smallest eigenvector onto a coordinate cone boundary. For DPF1 the normal subspace recovered **347 of 347** of its mixed-sign lowest PCA eigenvectors; for DPF3 **49 of 49**; for DPF4 **25 of 25**. DTLZ1,2,3,4, DPF5 needed no raw-normal cone projection in this bounded test.
- **Residual scientific caveats:** In the *historically invalid GT* of DPF2, 501 of 512 raw normals have mixed sign and only 261 are recovered in the low-rank normal plane. DTLZ7's historical GT has mixed-sign PCA normals at 304/512 sampled points with estimated median normal-subspace alignment 0.9945. Those GTs are **not certified true PFs**; no generator revision can make them so. DTLZ8 had only 13/512 mixed normals; its independent ideal reference remains unsupported. For remaining cases an outward cone projection is explicitly a **surrogate**, not an exact manifold normal. These cases are visible in the read-only audit and assigned to scientific baseline issue #81.
- **Controller preservation:** All 14 bounded audit cases attained calibrated median normalized distance between **1.9004 and 2.0827**; the target remains 2.0 with 5% relative tolerance. These are 512-point *bounded subsample* controller observations, **not** certified results for the full frozen 10K GT or a new versioned baseline.
- **Release interpretation:** This is a source-level correctness fix for the **outward-only null**, independently from the still unapproved DPF5 branch-mixture sampling law. Basic integration and final source/docs cleanup remain release gates; the new failure ECDF must be scientifically recalibrated and reviewed under issue #81 before anyone publishes new clinical Q scores.
