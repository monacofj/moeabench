<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Exact finite-reference HV for a degenerate three-objective DPF1 (#89)

**Status:** focused mathematical and numerical research. The public
metric and reference-calibration contract remain unchanged.

## Why this case matters

DPF1 with `M=3,D=2` projects the DTLZ1-like two-objective
base `(a t, a(1-t))`, where `a=0.5`, to a redundant
third objective `f3 = w1 f1 + w2 f2`. Its deterministic
chaos weights satisfy `0 < w1 < w2`. If the reference sample
contains both endpoints, the **same fixed ideal/nadir normalization**
used by the public metric transforms the continuous front into
`(t, 1-t, 1-t)`, `0<=t<=1`, in three objectives.

This is a one-dimensional, degenerate Pareto front embedded in
three-dimensional objective space. Its exact continuous-front HV
at the normalized minimization endpoint `(1.1,1.1,1.1)`
is mathematically accessible, *without a sampled GT*.

## Independent exact three-dimensional oracle

Sweep `t` (the normalized first objective) in ascending order.
As `t` increases, the other two normalized objectives both
decrease. At each sweep location, the most recently activated
sample point therefore dominates all earlier points in the
`(f2,f3)` cross-section. Its cross-sectional area is exactly
`(1.1-f2)*(1.1-f3)`; the dominated volume is the sum of
this area times the width of each x-slab, including the final
slab up to 1.1. No generic 3D HV backend is involved in
this independent calculation.

For the **continuous** curve the cross-sectional area is
`(t+0.1)^2` on `0<=t<=1`, followed by a final
`0.1`-wide slab of area `1.1^2`. Therefore,

```text
HV_true = integral_0^1 (t+0.1)^2 dt + 0.1*(1.1)^2
        = ((1.1)^3-(0.1)^3)/3 + 0.121
        = 0.5643333333333333...
```

This is a **conditional exact true-front denominator for the
specified DPF1 degeneracy and normalization only**. It is
not an upper bound for arbitrary DPFs or objective geometry.

## Real-sampler density predictions (independent arithmetic)

At the historical 10,000-point DPF1 budget, the real
`mop.pf(n_points=10000)` sampler uses evenly spaced
2D positions with exactly anchored endpoints. Adding the
Pareto-valid midpoint of the final objective-`f1` gap
produces a larger dominated volume than the finite GT,
but remains below the exact continuous-front HV.

Independent predictions from a vectorized 3D swept-volume
implementation (not yet source-bound CI results):

| GT points | Finite GT HV | GT + true-front midpoint HV | Current finite-GT ratio |
|--:|--:|--:|--:|
| 2,000 | 0.564033224967 | 0.564033362557 | 1.000000243941 |
| 10,000 | 0.564273329000 | 0.564273334500 | 1.000000009748 |

A dedicated manual audit checks the **actual** DPF1 PF
sampler at both budgets, independently derives the fixed-box
volume and computes the exploratory ratio to the true
continuous-front HV. At 2,000 points it additionally
cross-checks the production exact 3D `raw` and
MOP-calibrated `abs` public metric APIs. The heavier
10,000-point public exact-3D backend comparison is not
part of the routine unit test; its mathematical 3D sweep
oracle remains independently testable.

## Limits and policy

This special DPF1 case provides evidence that a rigorous
continuous-front denominator can also exist for **degenerate
three-objective** fronts. It does not justify simply
projecting solver points to sampled GT, changing the
existing `H_abs` denominator, clipping at one, declaring
an exhaustive answer for higher-dimensional DPFs, or
relaxing historical regression fixtures.

The remaining #89 gates concern irregular/disconnected
and unavailable fronts, exact-vs-Monte-Carlo uncertainty,
population and seed sensitivity, ranking preservation
and a separately maintainer-approved API/ADR decision.
Until then, `H_abs` remains a reproducible ratio to
a finite MOP-owned calibrated reference, not a universal
mathematical optimality certificate.
