<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Fixed-geometry GT refinement as a scalar HV denominator study (#89)

**Research status:** validation in PR CI required. No public metric, calibration
persistence, approved ADR, frozen regression or certification is modified.

## Question

Can increasingly dense deterministic Pareto-front sampling provide a stable
scalar reference when the continuous true-front HV is not available?

Compare two explicitly different strategies using the **actual** MOP PF
generators, rather than idealized grids alone:

1. **Regenerate at each budget**: `mop.pf(n_points=n)`. This is how current
   default GT calibration generates an analytical sample.
2. **Nested prefix**: take the first `n` points of one larger deterministic
   `mop.pf(n_points=N)` sample. These sets are nested by construction.

For both strategies, use exactly one fixed external normalization context
(ideal and nadir obtained from a dense anchor) and the same normalized
Hypervolume reference point `1.1`. Otherwise an observed change in HV could
arise from a moving bounding box, not GT refinement.

## Independent numerical oracles and cases

The script `scripts/audit/hv_gt_refinement_issue89.py` reuses previously
validated **backend-independent exact** area/volume sweeps:

- DTLZ7 with two objectives: a discontinuous Pareto front, using an independent
  2D dominated-area sweep;
- DPF1 with three objectives and intrinsic dimension two: a degenerate
  1D front, using a special-case independent exact 3D slab-volume sweep.

Budgets are 64, 128, 256, 512 and 1,024 against 2,048 anchor points. A
new unit test verifies the geometric oracles against the unchanged public
exact raw-HV backend at a smaller budget, plus the nested monotonicity
invariants. No GT sidecar is generated.

The DTLZ7 sampler's fixed seeded stratification produces identical prefix
samples at these budgets. The DPF1 sampler instead repositions interior
grid points when its requested size changes; regenerated samples need not
contain the previous samples. Nesting therefore depends on the MOP sampler
and is **not a general property** of deterministic calibration.

## Interpretation / important logical limit

For nonempty finite minimization fronts evaluated in **exactly the same**
normalization box, set inclusion implies monotonic Hypervolume:
`A ⊆ B ⇒ HV(A) ≤ HV(B)`. Thus nested refinement is monotone and useful
for reproducible numerical diagnostics. But monotonicity alone does not
establish that the last sampled value equals the continuum's HV.

Even if `HV(n+1) - HV(n)` is very small (or zero), an unobserved region of
the actual PF can still contribute substantial additional dominated volume.
This is particularly important for disconnected, mixed-dimensional,
poorly stratified, or user-defined fronts. A fixed numerical convergence
tolerance provides **no certified mathematical upper bound** without
additional assumptions such as a validated coverage radius and
regularity of the true front.

Likewise, normalizing by `HV(GT_sample)` cannot be guaranteed <=1 for
arbitrary feasible solver populations, regardless of the finite sample
budget. A deterministic calibration can improve numerical reproducibility
and typical accuracy, while a rigorous universally bounded scalar requires
a separate certified continuous-front bound or explicit additional
mathematical assumptions.

An all-purpose public rule that **silently selects exact denominators for
some MOPs and unconstrained estimates for others while labeling both
'H_abs = theoretical optimality'** would misrepresent its semantics.
A future compatible design must first resolve the public meaning of
`scale='abs'` and its behavior for black-box MOPs, in a new explicitly
maintainer-approved ADR. The user has rejected the proposal to return
`HV ± uncertainty` only for this metric; retain the consistent scalar
result interface.

## Next evidence before a policy decision

Repeat at the actual default DTLZ/DPF calibration budgets, quantify
CPU/memory costs, inspect the MC backend under fixed seeds and shared
sampling, and assess external-vs-self reference geometry. If a
mathematically justified bound is impossible, make the limitation
explicit in ordinary calibration/report metadata, without introducing
HV-specific uncertainty-shaped numerical return values. This is a
research proposal, not authorization to alter production behavior.
