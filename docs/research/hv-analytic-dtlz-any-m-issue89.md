<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Exact Hypervolume of continuous DTLZ1 and DTLZ2 fronts in arbitrary M (#89)

**Research only.** The existing public `scale="abs"` continues to divide
by the calibrated finite GT. This note proposes no API change, no new
uncertainty output and no replacement for an accepted ADR or frozen reference.

## The geometric observation

Assume a **minimization** MOP, the *complete, original* DTLZ1 or DTLZ2
continuous Pareto front, a common normalized ideal at the origin and
the complete normalized axis nadir at `(1,...,1)`. Assume the
normalized Hypervolume reference is `r=(1.1,...,1.1)`.

With `M` objectives, the nondominated region inside the
hyperrectangle `[0,1.1]^M` is:

- DTLZ1: the standard M-simplex `x_j>=0, sum(x_j)<1`, volume `1/M!`;
- DTLZ2: the positive orthant of the unit M-ball
  `x_j>=0, sum(x_j^2)<1`, volume
  `π^(M/2)/(2^M Γ(M/2+1))`.

Both regions are entirely within the reference box because `r>=1`.
Subtracting their volumes from the box gives **exact**
continuous-front Hypervolume denominators:

```text
HV_true(DTLZ1,M) = 1.1^M - 1/M!
HV_true(DTLZ2,M) = 1.1^M - π^(M/2)/(2^M Γ(M/2+1))
```

The formulas cover **any integer M>=2**, not only two-objective
cases. No numerical integration or Monte Carlo estimate is required
to compute the exact mathematical denominator.

For any **feasible** population evaluated under the **same**
exact axis normalization, its dominated region is a subset of the
continuous true-front dominated region, so
`HV(P) / HV_true <= 1`. This is a mathematical ceiling.
It does not authorize assuming a feasible population in an arbitrary
external normalization context has the same denominator.

## Numerical validation without changing production

The independent file `scripts/audit/hv_analytic_dtlz_issue89.py`
computes the analytic denominator for M=2,3,4,5,8, and at M=2,3,4
cross-checks the **actual** `mop.pf(n_points=64)` generators and
public exact raw-HV backend against the proved continuous ceiling.
The tests reside in a **new** file
`tests/unit/test_hv_analytic_dtlz_issue89.py`. These tests do not
modify the public metric, calibration, tests, baselines or ADRs.

This check intentionally uses only the public raw scale: it neither
creates synthetic persistent calibrations nor changes the established
finite-GT meaning of `H_abs`. Earlier PRs #161, #165 and #166
already independently tested the public absolute ratio for specific
low-dimensional examples.

## Caveats and decision boundary

The formulas require complete original DTLZ1/DTLZ2 fronts, objective
feasibility and the stated **normalized common geometry**. They do
**not** establish formulas for DTLZ5/6, DPF projections, constrained,
disconnected, empirically calibrated or arbitrary user-defined fronts.
A lower-resolution GT may not preserve exact ideal/nadir anchors for
other MOPs; changing the normalization also changes the denominator.
Numeric backend approximation to `HV(P)` can introduce small
numerical exceedances even if the denominator is analytic.

Crucially, **the exact denominator need not be an exception in the
returned metric shape**: it is still one scalar. The key question is
whether the framework can establish a mathematically valid reference
for every supported MOP/context, without silently mixing genuinely
exact denominators with sampled estimates under the same semantics.

A globally uniform public `abs` contract remains a **future
architectural decision**, requiring an explicit approved ADR.
The next investigation should test DTLZ/DPF families where no closed
form is available, verify feasibility and normalization, and
determine whether a deterministic, converged finite calibration can
serve as a uniform scalar rule without asserting a nonexistent
guarantee. We specifically **do not** add `HV ± uncertainty` or
confidence intervals, following the maintainer's decision to retain
a consistent scalar interface across indicators.
