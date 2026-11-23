<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0058: Scalar Absolute Hypervolume with Certified Continuous References

## Status

Accepted (maintainer approval on 2026-11-23; implementation tracked by issue #89).

## Context

The canonical conventional HV modes are `raw`, `rel` and `abs`.
Previously, `H_abs=HV(P)/HV(GT_sample)` for every MOP. The denominator
is the MOP's calibrated **finite** GT, not generally the mathematical
continuous Pareto-front volume. Consequently a feasible solution can have
`H_abs>1` solely because a finite GT misses true-front regions.
Increasing GT density or checking numerical convergence cannot certify
an upper bound for arbitrary black-box/disconnected fronts. Clipping
at one would distort valid differences, while returning `HV ± error`
only for this indicator would break the uniform scalar metric interface.

Research and independent checks in PRs #161, #165–#169 demonstrate finite
sampling undercoverage, derive exact true-front denominators for original
DTLZ1–4 under certified axis normalization, and establish the special
degenerate DPF1(M=3,D=2) continuous-front integral.

The maintainer authorized a **documented, hybrid, scalar** policy on
2026-11-23 after being advised of the conflict with ADR 0035's
unconditional finite-GT denominator and its unconditional interpretation
of 1.0 as mathematical perfection.

## Decision

1. Preserve `raw` and `rel` exactly as governed by ADRs 0020, 0043
   and 0047. Preserve the ordinary `MetricMatrix` scalar data shape,
   fixed normalization context, exact/Monte Carlo engine choice, GT-owned
   calibration prerequisite, and all unmodified historical regression
   fixtures and reference certificates.
2. For `scale="abs"`, use a **mathematically exact continuous true-PF
   denominator** only when all of the following are proven for the
   evaluation: (a) an *exact built-in original* MOP class with a verified
   formula; (b) an MOP-owned GT created by analytical calibration
   (`mop.calibrate()` or a sidecar explicitly recording that source);
   (c) selected ideal/nadir normalization bounds equal the proved true
   per-axis ideal/nadir in that problem's objective units; and
   (d) the reference point is the unchanged normalized 1.1 in each
   objective. The exact formula applies in this **same** geometry.
   Do not extrapolate from the class name to subclasses/custom MOPs, or
   instances whose `pf`, `ps` or `evaluation` methods have been overridden
   (their analytical generation is no longer the certified built-in one).
3. The current certified formulas for any integer M>=2 are
   `DTLZ1: 1.1^M - 1/M!` and
   `DTLZ2/3/4: 1.1^M - pi^(M/2)/(2^M Gamma(M/2+1))`.
   For original `DPF1(M=3,D=2)`, when its projection weights and
   extremal normalization establish the verified
   `(t,1-t,1-t)` geometry, use
   `((1.1)^3-(0.1)^3)/3 + 0.1*(1.1)^2`.
   Other DTLZ/DPF families and configurations are **not certified** by
   analogy or a dense sampler alone.
4. In every other case, including external/empirically supplied GT,
   legacy sidecars without reliable analytical provenance, shifted or
   self-derived non-exact normalization, user MOPs, unsupported front
   geometries and unproved assumptions, retain the conventional
   finite-sample ratio `HV(P)/HV(GT_sample)`. This scalar may exceed 1
   and must **not** be clipped, labeled a certified optimum or silently
   altered by increasing the GT sampling budget.
5. Continue to require calibrated GT even in the exact case; no
   implicit disk writes or recalibration. Existing `mop_gt_v1`
   sidecars already record `gt_source`; read this provenance when
   loading and retain it in memory without changing the file schema.
   Unsupported/missing provenance means finite-reference fallback.
6. Report the denominator and its basis as diagnostics:
   `hv_abs_denominator`, `hv_abs_gt_points`, and
   `hv_abs_reference_basis` (`continuous_exact` or `sampled_gt`).
   These must use the same selected reference geometry; the exact
   denominator is not derived from the evaluated population's score.
   The result remains one numeric metric, with no HV-specific
   uncertainty interval. User documentation must explain that
   `H_abs=1` is *theoretical* only for `continuous_exact`.
7. `mode="fast"` approximates the **numerator** by Monte Carlo;
   the exact continuous denominator does not remove numerical engine
   approximation error. Consequently small >1 observations from an
   approximate numerator are not automatically evidence of a
   mathematical exceedance. No automatic clipping or hidden tolerance
   policy is authorized.

## Compatibility and regression policy

This ADR partially supersedes the unconditional finite-GT
`scale="abs"` denominator and unconditional 1.0=perfection wording in
ADRs 0019/0035. Those ADRs remain historical records. It preserves
ADR 0018's default GT point budgets and finite-reference terminology,
ADR 0057's calibration controls, ADR 0020's normalization geometry,
ADR 0043's backend choice and ADR 0047's `rel` semantics.

The original finite-sample result remains applicable when the user
explicitly supplied a GT or when the exact applicability conditions
fail. No old regression test, frozen numerical/performance baseline
or certification reference may be silently modified. New independent
positive tests must exercise certified applicability, fallback,
sidecar load/reload, changed reference geometry, external GT,
and preservation of `raw`/`rel`.

## Limits / remaining issue #89 work

A general user-defined MOP with an unknown continuous front cannot
mathematically guarantee `H_abs<=1` from any finite sampled GT alone.
Comparing scalars across different basis classes does not establish
identical interpretations or common continuous-optimality guarantees.
Extending certified formulas requires independent geometric proofs,
bounded feasibility assumptions and source-bound validation before
whitelisting. The research link to finite-discretization-sensitive
FAIR methodology remains deferred, not a production dependency.

## Normative GT / HV contract and documentation authority

The precise **current technical specification** of GT provenance,
sampling responsibilities and how a certified continuous denominator
differs from a sampled GT is maintained in
[`docs/gt.md`](../gt.md), Sections 3.4 and 9.1–9.3.
The implementation of these normative conditions is in
`moeabench/mops/calibration.py`,
`moeabench/metrics/_absolute_hv_reference.py` and
`moeabench/metrics/evaluator.py`.

This ADR defines **what is mandatory**; the specification explains
how the current code fulfills it. Research narratives under
[`docs/research/`](../research/hv-issue89-research-memorial.md)
record hypotheses and historical experiments and cannot
override this adopted decision. In particular, references to the
previous all-sampled `H_abs` behavior in earlier research notes
describe their **investigation-time** behavior, not the
current operating contract.

A calibrated GT and a certified denominator are separately
identified objects. Merely labelling a GT source `analytical`,
raising the point count, observing numerical convergence, or
attaching an external reference does not certify the continuum
volume. Ineligible contexts **must** use the sampled-GT basis,
not an unsupported extrapolation of an exact formula. Neither
`ps()` nor `pf()` implicitly calibrates or writes a sidecar;
no implicit GT budget or schema migration is authorized by this
ADR. The `hv_abs_reference_basis` diagnostic is a required
provenance field; it does not replace the scalar metric result.

## Evidence

See `docs/research/hv-finite-reference-issue89.md`,
`hv-reference-density-issue89.md`,
`hv-dpf1-degenerate-issue89.md`,
`hv-analytic-dtlz-any-m-issue89.md`,
`hv-dtlz234-equivalence-issue89.md` and
`hv-gt-refinement-issue89.md`.
