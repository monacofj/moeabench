<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Issue #89: Finite GT, Absolute Hypervolume and Certified Reference — Research Memorial

**Purpose:** preserve the scientific questions, evidence, failed candidate
solutions, mathematical assumptions, experiments, and decision history
behind the operational absolute-Hypervolume reference rule. This is
**historical/research documentation**. The current technical
specification is [`docs/gt.md`](../gt.md), especially Sections 3.4 and
9.1–9.3. The normative adopted decision is
[ADR 0058](../adr/0058-scalar-absolute-hv-certified-reference.md).
Neither this memorial nor the older exploratory reports supersedes an ADR.

## Original problem and minimal independent reproducer

Before the adopted hybrid policy, the `abs` denominator was always
the exact or approximate HV of a finite calibrated MOP-owned sampled
front. For DTLZ1 with M=2, reference
`G={(0,0.5),(0.5,0)}`, and normalized box reference `(1.1,1.1)`,
a direct exact two-dimensional area union yields `HV(G)=0.21`.
A valid true-PF midpoint `P={(0.25,0.25)}` gives `HV(P)=0.36`.
The resulting `H_abs=12/7>1` proves that a feasible optimal
population can exceed its **discretized reference**. Neither the
Hypervolume implementation nor the true PF is inconsistent.

The previous result must remain reproducible whenever the user
**explicitly calibrates with this finite GT**, because the accepted
policy intentionally retains finite-reference semantics in that case.
See `tests/unit/test_hv_finite_reference_issue89.py` and
PR #161 ([minimal evidence](hv-finite-reference-issue89.md)).

## Experimental progression and source-bound evidence

The evidence matured in the following sequence. The research PRs
initially made **no public metric or ADR changes**; those claims were
correct at the time. The operational change occurred subsequently,
under ADR 0058.

| Stage | Evidence and method | Result | Repository PR |
|---|---|---|---|
| Minimal exact two-dimensional witness | Independent rectangle-area union with true-PF points | Finite sample can underestimate the true-front dominated volume | #161 |
| Actual GT sampling density | DTLZ1 M=2 at 2,000 and 10,000 points; DTLZ2 M=2 at 2,000; exact 2D geometric oracle compared with unchanged public raw/abs APIs | Gaps remain even at the adopted densities; continuous DTLZ1/2 2D volumes can be proved separately | #165 |
| Three-objective degeneracy | Real DPF1 M=3,D=2 sampler, exact special-case 3D slab-volume integral and public exact API check at 2,000 points | Continuous-front scalar reference `0.564333333... ` in the proved axis geometry, with finite GT still underestimating | #166 |
| Any-M complete analytic fronts | Simplex and positive-orthant unit-ball complement volumes for original DTLZ1/2 | Exact `HV_true` expressions for any supported M>=2 in proved normalization | #167 |
| Spherical family equivalence | DTLZ3 optimal distance g=0; DTLZ4 angular power transformation covers the full angle range | Original DTLZ2, 3 and 4 share the continuous spherical HV denominator | #168 |
| Deterministic finite refinement | Actual DTLZ7 M=2 disconnected sampling and DPF1 M=3,D=2 degenerate sampling, fixed geometry, independent fresh vs nested fronts, exact geometric sweep | Nested HV is monotone; neither small increments nor higher finite density certifies the continuous optimum | #169 |
| Approved operating rule | New ADR, guarded resolver with analytic GT provenance and exact normalization, sampled fallback, report diagnostics and new tests | Scalar exact-when-proven / finite-sampled-otherwise semantics, preserving raw/rel | #170 |

For the original complete DTLZ1 and DTLZ2/3/4 fronts, normalized
ideal `(0,...,0)`, nadir `(1,...,1)`, and reference
`(1.1,...,1.1)`, the exact volumes are respectively

```text
DTLZ1:     HV_true(M) = 1.1^M - 1/M!
DTLZ2–4:   HV_true(M) = 1.1^M - pi^(M/2)/(2^M Gamma(M/2+1))
```

The original DTLZ1 front is first scaled by its true nadir
`(0.5,...,0.5)`. The DTLZ1 nondominated complement is the unit
simplex; the DTLZ2/3/4 complement is the positive orthant of the
unit M-ball. These analytic volumes require the *whole* original
continuous front and exactly matching reference geometry. They are
not formulas for arbitrary user MOPs, disconnected DTLZ7, DTLZ5/6,
or every DPF configuration.

For original DPF1 M=3,D=2, when its deterministic projection
weights and proven axis extrema apply, the normalized true PF
is `(t,1-t,1-t)`; integrating 3D slice volume gives

```text
HV_true = integral_0^1 (t+0.1)^2 dt + .1*(1.1)^2
        = ((1.1)^3-(0.1)^3)/3 + .1*(1.1)^2
        = 0.564333333333...
```

The above are *geometrically proved denominators*, not statements
that every finite GT sample has the same HV.

## Hypotheses tested or rejected

**Clipping `H_abs` at 1.** Rejected because it hides valid
differences between populations; a clipped score would no longer
represent the sampled-reference ratio.

**Projecting evaluated points onto sampled GT or onto PF.**
Projection onto a finite GT cannot recover the unrepresented
continuous dominated volume. Projection onto the true PF is also
insufficient: the minimal DTLZ1 counterexample already consists of
true-PF points.

**Increasing the sampled GT budget until HV seems stable.**
Useful for approximate reference fidelity, and often reduces
exceedances. Not a certified fix: an unseen disconnected component
may contribute volume even after negligible successive increments.
Nested samples guarantee monotone exact HV for an unchanging
normalization box, whereas independently regenerated deterministic
grids need not be nested. Thus apparent convergence is not a
proof of an upper bound.

**Returning `HV ± uncertainty`.** Considered and rejected by
the maintainer because other indicators return a scalar and introducing
one special-case uncertainty result would break interface uniformity.
In addition, finite discretization has systematic one-sided
undercoverage; it is not automatically representable by a
statistical confidence interval. This decision does not eliminate
numerical error analysis from research.

**Assuming a shared formula for all MOPs.** Rejected. A correct
certificate depends on the original MOP's actual full PF geometry,
feasibility, true ideal/nadir, and selected normalized reference.
A formula proved for DTLZ2/3/4 cannot be transferred by visual
resemblance to other DTLZ variants or to arbitrary DPF projections.

**Changing `raw` or `rel`.** Not needed. `raw` is the dominated
volume in its chosen context. `rel` measures competitive-reference
performance and may legitimately exceed 1 under ADR 0047, a different
property from an `abs` finite-GT denominator.

## Accepted conclusion and implementation boundary

The maintainer approved a *documented hybrid scalar* resolution on
2026-11-23. PR #170 implemented it after four source-bound CI checks:

- Only proven original built-in geometry, original class identity,
  analytically generated MOP-owned GT provenance, and exact matching
  normalization can activate `continuous_exact` for DTLZ1–4 or
  the proven DPF1 M=3,D=2 configuration.
- In every other case, including externally supplied GT, empirically
  calibrated GT, unknown legacy provenance, altered/subclassed
  samplers and incompatible bounds, `sampled_gt` preserves the
  historical *unclipped* finite-reference ratio.
- `mop.calibrate()` remains required even for the certified scalar
  denominator. No GT sidecar schema migration or implicit
  recalibration is performed. The result adds
  `hv_abs_reference_basis` to existing
  `hv_abs_denominator`/`hv_abs_gt_points` diagnostics.
- `mode="fast"` continues to estimate the numerator using seeded
  Monte Carlo. An analytic denominator cannot certify that an MC
  estimate never exceeds 1; only a feasible front with exact
  numerator and a valid proof has that mathematical guarantee.
- Source-bound CI passed; **no pre-existing regression test,
  frozen baseline, ADR history or numeric/performance certificate
  was overwritten**.

The software specification, API reference and narrative report were
aligned in PR #170. The approved design is *not* a universally
bounded absolute score for every black-box MOP: where the true PF
cannot be certified, finite-reference meaning and limitations are
documented transparently. No FAIR metric was reintroduced.

## Scientific relation to FAIR, without reopening that workstream

The maintainer noted that a finite-GT denominator failing to capture
continuous-front volume reinforces the motivation to **take finite
discretization explicitly into account** when evaluating Pareto
approximations. This is a scientific connection to the deferred FAIR
research, **not** a dependency of GT calibration or an instruction to
restore FAIR/Q-scores to the runtime. The observation is preserved
here and in issue #89 for future research only.

## Reproduction map

The original reports remain separate to retain dates, methods,
numerical predictions, provisional interpretations and negative
evidence:

- [Finite-reference minimal example](hv-finite-reference-issue89.md)
  and `tests/unit/test_hv_finite_reference_issue89.py`.
- [Real-sampler density study](hv-reference-density-issue89.md),
  `scripts/audit/hv_reference_density_issue89.py` and
  `tests/unit/test_hv_reference_density_issue89.py`.
- [DPF1 degeneracy](hv-dpf1-degenerate-issue89.md),
  `scripts/audit/hv_dpf1_degenerate_issue89.py` and
  `tests/unit/test_hv_dpf1_degenerate_issue89.py`.
- [Arbitrary-M DTLZ analytical denominators](hv-analytic-dtlz-any-m-issue89.md),
  `scripts/audit/hv_analytic_dtlz_issue89.py` and
  `tests/unit/test_hv_analytic_dtlz_issue89.py`.
- [DTLZ3/4 exact-front equivalence](hv-dtlz234-equivalence-issue89.md),
  `scripts/audit/hv_dtlz234_equivalence_issue89.py` and
  `tests/unit/test_hv_dtlz234_equivalence_issue89.py`.
- [Fixed-context deterministic refinement](hv-gt-refinement-issue89.md),
  `scripts/audit/hv_gt_refinement_issue89.py` and
  `tests/unit/test_hv_gt_refinement_issue89.py`.
- [Accepted architecture](../adr/0058-scalar-absolute-hv-certified-reference.md)
  and `tests/unit/test_hv_absolute_reference_policy_issue89.py`.

See [issue #89](https://github.com/monacofj/moeabench/issues/89)
for original acceptance questions and maintainer discussion.
The [GT specification](../gt.md) and [ADR 0058](../adr/0058-scalar-absolute-hv-certified-reference.md)
take precedence for current behavior and normative requirements.
