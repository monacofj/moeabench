<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Reference-density sensitivity at 2,000 and 10,000 points (#89)

> **Historical research status (2026-11-23):** The experiment and any
> investigation-time statements below are preserved as research evidence.
> Since this study, the maintainer accepted
> [ADR 0058](../adr/0058-scalar-absolute-hv-certified-reference.md)
> and merged implementation PR #170: `H_abs` now uses an exact
> continuous-front denominator **only in mathematically certified,
> provenance-verified normalization contexts**, retaining the original
> finite sampled-GT scalar otherwise. This study is **not** a normative
> description of the current API. See the [GT specification](../gt.md)
> and the [consolidated issue #89 memorial](hv-issue89-research-memorial.md)
> for the current outcome and chronology.

**Status:** independently derived 2D evidence; source-bound CI still required.
This work does not change any public metric, GT sampler, ADR or baseline.

## Study protocol

Run `python -m scripts.audit.hv_reference_density_issue89`.
The program invokes the **actual** DTLZ1(M=2) and DTLZ2(M=2)
`mop.pf(n_points=n)` implementations. It uses frozen ideal/nadir
from the resulting GT, the standard 1.1 normalized HV endpoint,
and a separate 2D area-sweep oracle that does **not** call pymoo
or moocore. It independently checks the existing public exact
`mb.metrics.hv(..., scale="raw")` and MOP-owned `scale="abs"`
contracts, calibrating only to a temporary sidecar.

For each GT, it inserts a single **mathematically Pareto-optimal**
point in the largest gap in objective `f1`: a midpoint on
`f1+f2=0.5` for DTLZ1 and a mid-angle point on
`f1²+f2²=1` for DTLZ2. The evaluated front is GT plus this
point, not a purportedly representative MOEA run. Both numerator
and denominator are computed using identical fixed bounds.
Therefore any increase proves finite-reference undercoverage,
without invoking solver error or a changed geometry.

## Independent numerical predictions

These values follow from a separate NumPy implementation of the
published PF sampling laws. **Predictions must not be described as
source-bound MoeaBench test results until the new PR CI passes.**

| Problem | GT budget | GT HV | H_abs of GT plus midpoint | HV increment |
|:--|--:|--:|--:|--:|
| DTLZ1 M=2 | 2,000 | 0.7094994616 | 1.000004529975 | 3.2140e-6 |
| DTLZ1 M=2 | 10,000 | 0.7099008219 | 1.000000324143 | 2.3011e-7 |
| DTLZ2 M=2 | 2,000 | 0.4244054697 | 1.000000000071 | 3.0325e-11 |

The DTLZ2 effect is at a much smaller scale; it is especially sensitive
to numerical tolerance, and is **not** evidence that any real MOEA
routinely exceeds unity. Actual sampler behavior and independent
public-API agreement must be checked in CI.

For two-objective DTLZ1 with endpoints and normalized reference
`(1.1,1.1)`, the **continuous** Pareto front has exact
HV `1.1² - 1/2 = 0.71`. A finite GT defines a staircase in the
dominated plane, with lower HV even when every GT point lies on
the true front. On any DTLZ1 gap of normalized horizontal width
`h`, inserting the line midpoint increases HV by `h²/4`.
This gives a constructive explanation of the finite-GT gap,
including the effect at realistic adopted point budgets.

## Exact continuous-front denominator for these two special cases

In the *same* normalized 2D geometry, the continuous DTLZ1 front
is `x+y=1`, and DTLZ2 is the first-quadrant circle
`x²+y²=1`. For minimization with reference point `(1.1,1.1)`,
the complement of each dominated region inside the full
`1.1×1.1` square is respectively a right triangle of area
`1/2` and a quarter disk of area `π/4`. Thus the **exact**
mathematical continuous-front denominators are

- `HV_true(DTLZ1,M=2) = 1.21 - 0.5 = 0.71`
- `HV_true(DTLZ2,M=2) = 1.21 - π/4 ≈ 0.4246018366`

For any **feasible** evaluated front in these two problems,
its exact dominated HV cannot exceed the continuous-front HV.
Consequently `HV(P)/HV_true ≤ 1` is a mathematically defensible
bounded score in these special cases. This is different from simply
clipping `HV(P)/HV(GT_sample)` or projecting onto a finite GT:
it uses a proved, geometry-matched denominator instead of an
approximation. The script now computes both the unchanged public
finite-GT ratio and this **separate exploratory continuous-bound
ratio**, and verifies `HV(GT) ≤ HV(GT∪{p}) ≤ HV_true`.
These formulas assume the exact true front, minimization and the
specified explicit bounds; they are **not** valid general formulas
for a different normalization context, more objectives, constrained
fronts, nonoptimal or infeasible points, or an unproved sampler.

A production change to use a mathematical true-front denominator
would redefine the `abs` scale and require a maintainer-approved
ADR, consistent handling of the remaining MOPs, and tests protecting
comparability and persistence. None is introduced here.

## Scientific scope and next gates

The current evidence distinguishes mathematical undercoverage from
implementation errors in **continuous two-objective** problems.
It does not resolve the entire issue: the three-or-more-objective
DPF family, degenerate/disconnected GT geometry, reference
normalization choices, stochastic MOEA populations, exact-vs-Monte-Carlo
sampling error, ranking changes and feasible certified denominators
remain to be studied before a new public policy is adopted.
In particular, the analysis neither clips ratios nor silently
replaces the currently calibrated GT. `H_rel > 1` remains a
different, valid competitive-reference result under ADR 0047.

Relevant existing policy: ADR 0018 (GT density), 0020 (HV geometry),
0035 (three scales), 0042 (diagnostics), 0043 (backend choice),
0047 (relative semantics). A future definition of automatic uncertainty,
certified true-PF bound or alternative `H_abs` denominator would
require a **separate, approved ADR** and independent numerical evidence.
