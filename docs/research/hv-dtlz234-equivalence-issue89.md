<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Extending the exact continuous Hypervolume reference to DTLZ3/DTLZ4 (#89)

**Research only; no change to the public normalized Hypervolume, calibration,
baseline or architectural decisions.**

## Geometric proof

The original DTLZ3 problem has the same objective-space spherical map as
DTLZ2, but replaces its distance function with multimodal Rastrigin-style
`g`. On the complete optimal `g=0` set its objective image is precisely
the positive orthant of the unit sphere. The multimodality changes optimization
difficulty but not the true Pareto front.

Original DTLZ4 also has the DTLZ2 spherical geometry, with each of the first
`M-1` decision variables mapped to an angle by
`theta_i = (π/2) x_i^100`. The function `x→x^100` is continuous and
surjective on `[0,1]`; therefore every angle in `[0,π/2]` remains
attainable. This strongly biases uniform decision-space sampling but does not
remove any part of the continuous true PF. Optimal distance variables yield
`g=0` as in DTLZ2.

Hence for *original* DTLZ2, DTLZ3 and DTLZ4, any integer `M>=2` and
exact common axis normalization with positive reference point `r=1.1`:

```text
HV_true(M) = 1.1^M - π^(M/2)/(2^M Γ(M/2 + 1))
```

This result extends the exact denominator derivation from PR #167 without
altering `H_abs=HV(P)/HV(GT_sample)`. It is only a candidate reference
formula for a later approved architecture. Under a fixed shared geometry, it
is mathematically valid for feasible evaluated populations; other MOPs,
external or self-normalization and numerical integration need separate
consideration.

## Source-bound independent checks

`scripts/audit/hv_dtlz234_equivalence_issue89.py` calls the actual
DTLZ2/DTLZ3/DTLZ4 `ps()` and `evaluation()`, checks every resulting point
lies on the analytical sphere, verifies ideal/nadir axis anchors, and compares
the current exact raw-HV finite sampled value with the proven continuous
value. It uses M=2,3,4 and only 64 points to bound routine CI cost.
The independent tests verify the closed-form formula at M=2,3,4,5,8, plus
DTLZ3's zero-distance objective equivalence and DTLZ4's angular surjectivity.

The study adds **new research-only files** and does not alter the public
indicator, GT storage, any accepted ADR or existing regression fixtures.
The evidence must pass source-bound CI before being reported as verified.

## Broader #89 decision remains unresolved

Analytical denominators now have proofs for original DTLZ1 and DTLZ2/3/4
under the specified common normalization. Original DTLZ5/6 can include
additional `g>0` Pareto geometry for M>=4 and cannot be treated as a
simple sphere subset in a denominator proof. Disconnected DTLZ7,
constrained DTLZ8/9, projected DPFs, empirically calibrated and
user-supplied MOPs may require entirely different methods.

The user's decision to retain a **uniform scalar indicator result** rules
out special-case `HV ± uncertainty` output. A future solution needs a
consistent policy for analytically known and unavailable true fronts,
compatible with the existing ADRs and explicit approval before any
change to `scale="abs"`.
