<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3/DPF5: Inaugural Corrected GT Calibration and Regression Migration (#119)

This report records why pre-correction diagnostic tests were incompatible
with the mathematically correct DPF3/DPF5 implementation and how those
checks were migrated following **explicit maintainer authorization on
2026-11-20**. It also describes the **first valid DPF5 scientific GT
calibration** under the published equations, its validation protocol,
and its relationship to numerical stability and performance certification.

This is an implemented corrective migration in
[PR #140](https://github.com/monacofj/moeabench/pull/140), not an
endorsement of silently updating reference values whenever a test fails.
The dated authorizations are recorded in [ADR 0056](../../adr/0056-versioned-certification-and-performance-reference-lifecycle.md)
and [ADR 0057](../../adr/0057-proposed-purpose-specific-gt-budgets.md).

## 1. Why the old defect-observation tests needed replacement

The audit in [research PR #139](https://github.com/monacofj/moeabench/pull/139)
intentionally included tests that *required* the old program to exhibit
specific defects: DPF3 projected thresholds in source order, collapsing
small-angle components, DPF5 beta/gamma equations differing from the
paper, exact-boundary misassignment, and nonsquare sampling requests
returning fewer unique objectives.

Those tests made sense for a scientific **diagnosis of the old evaluator**.
Once the published equations and MOP-specific generators were restored,
the tests failed **because the defects had disappeared**. Keeping these
negative predicates as live release regressions would require reintroducing
the very bugs they were meant to expose.

Following explicit approval, **15 failing assertions across eight existing
research test files** were changed to enforce positive correctness:

| Existing research test file | Cases | Positive replacement contract |
|---|---:|---|
| `test_gt_dpf5_allocation_issue119.py` | 1 | Requested and effective P/D budget accounting |
| `test_gt_dpf5_finite_sampler_issue119.py` | 1 | 128 requested decisions produce 128 distinct published objectives with both branches |
| `test_gt_dpf5_fixed_impact_issue119.py` | 3 | Corrected evaluator matches independently reconstructed published values at fixed decisions |
| `test_gt_dpf5_general_d_issue119.py` | 2 | High-M P and D=4 gamma satisfy the published equations and spherical norm |
| `test_gt_dpf5_offpareto_issue119.py` | 3 | P branch agrees with published scaled spherical equations for g>0 |
| `test_gt_dpf_branch_diagnostics_issue119.py` | 1 | Nonsquare 48-point request remains exactly 48 unique Pareto objectives |
| `test_gt_dpf_equation_probes_issue119.py` | 2 | Published sorted projections and D ownership of x1=1/3 |
| `test_gt_dpf_numerics_kernel_issue119.py` | 2 | Stable positive DPF3 complement, correct DPF5 spherical invariant |

With `mpmath` installed in the independent research workflow, a
**sixteenth** historical assertion appeared in
`test_gt_dpf_high_precision_issue119.py`. It expected the old
decision-uniform x^100 sampler to generate positive high-precision values
that collapsed to zero in float64. The corrected angular sampler no
longer exhibits that avoidable concentration; the new test compares
the sampled objectives against the high-precision reference and checks
that 127/128 are correctly represented as positive, with the single
mathematical zero at the origin.

Two existing research diagnostic scripts were also aligned with the
published semantics: DPF5 branch assignment uses **x1>=1/3**, and
branch-at-equality is determined from an independent paper-equation
reconstruction rather than hard-coded to the old P branch.

The historical *scientific explanations* of the defects remain in the
research reports and the original Git commit. Their numerical values
are **not** maintained as a previous validated calibration baseline.

## 2. First DPF5 corrected GT calibration

The maintainer explicitly authorized treating the replacement as the
**first valid scientific DPF5 calibration**, rather than promoting the
old evaluator's provisional geometric fixture to an immutable
scientific-reference version.

The calibration uses **`DPF5(M=3,D=2)`** with
**`pf(n_points=1000)`** from the corrected source and a pinned
Python 3.14.4/NumPy 2.5.4/SciPy 1.18.1 reference runtime.
The workflow independently validates every objective against the
published DPF5 reconstruction, checks `g=0`, unit norm, branch
membership, deterministic replay and numerical uniqueness.

Observed properties of the inaugural fixture:

| Property | Verified result |
|---|---|
| Requested objectives | 1,000 |
| Returned numerically unique objectives | 1,000 |
| Full-dimensional P component | 967 |
| Degenerate D component | 33 |
| Maximum production-versus-independent published-formula difference | 0 (in checked float64 values) |
| GT array SHA-256, C-order little-endian float64 | `39d5c21488e5c65369a2392e8c9e2d640add93f6c4368614eb680cf88d517a5a` |
| Reference generation | [Pinned CI run 44852783165](https://github.com/monacofj/moeabench/actions/runs/44852783165) |

The recorded **1000-by-3 array** is stored at
`tests/stability_gt_geometry_reference.json` under `problems.DPF5`,
with calibration metadata and the SHA-256 above. The existing DPF5
`forward_max` and `backward_max` numerical stability thresholds in
`tests/stability_reference_gt.json` were initialized to **zero**.
The preexisting test continues to allow only its explicit additive
numerical tolerance of **1e-12**, in both directed nearest-neighbor
directions. This is **stricter** than the original provisional bounds,
not a relaxation to force a passing outcome.

Every other MOP entry in those fixtures remains unchanged. The source
`tests/test_stability_basic.py` still performs the same two-sided
geometric comparison; no bypass, skip or blanket tolerance increase
was introduced.

In accordance with the maintainer's direction, the old DPF5 provisional
numbers are **not saved as a separate release or calibration version**.
Ordinary Git history retains the earlier source bytes for auditability.

## 3. What this baseline does and does not certify

The fixture tests **numerical reproducibility of a specific DPF5
reference** under the approved runtime. The separate MOP scientific
tests certify published equation agreement and Pareto membership
properties, while the DPF5 angular-cover theorem gives a conditional
**real-arithmetic** covering bound when full component grids fit the
budget.

This does not establish a rigorous all-rounding IEEE interval
certificate, universal sampling density, or a numerical accuracy
tolerance and automatic budget rule for every possible DPF5
configuration. The **automatic `mop.calibrate()` default** is unchanged
and remains a separate ADR 0057 decision.

The replacement of this provisional **GT geometry fixture** does not
modify the independently accepted **0.18.0 statistical performance
baseline**, its certified experiment design, old performance
observations, or its reference hashes. The ADR 0056 performance
certification lifecycle remains binding.

## 4. Validation and future reference governance

The scientific correction workflow tests DPF3/DPF5 equations against
independently reconstructed oracles and verifies deterministic
MOP-specific finite references. The former diagnostic research workflow
tests current correctness instead of requiring old defects. Complete
unit and pinned numerical stability checks must pass on the **same
head revision**.

Once this inaugural corrected DPF5 reference is accepted, the
**immutable-reference safeguard again applies normally**: future
changes require explicit justification and approval. The present
case is a documented one-time correction of a provisional
scientifically invalid reference, not an automatic calibration renewal.
