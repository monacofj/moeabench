<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0059: Measurable GT Quality and Independent Acceptance

## Status

**Accepted (maintainer authorization 2026-11-23).** Initial opt-in API and
MOP report integration implemented for structural diagnostics and independent
DTLZ1–4, DTLZ7–9 and DPF1–5 objective-space geometric oracles.
MOP-specific scientific certification for DTLZ5/6 and other
unsupported properties remains incomplete; unsupported claims report
\`unverified\`, not \`pass\`.

## Motivation and architectural boundary

MoeaBench has independently validated published mathematical Pareto
fronts and corrected analytical samplers (issues #119 and #65), but a
test that confirms shape or deterministic output alone does not
establish an adequate reference for scientific indicators. In
particular:

- globally true Pareto membership and constraint feasibility are
  **correctness conditions**;
- presence of every known mathematical component, extremum and seam
  is a **support-completeness** condition;
- geometric covering radius, spacing regularity, effective point
  density, indicator sensitivity and cost characterize **finite
  sampling quality** and depend on MOP geometry and sampling budget.

These must not be collapsed into one universally weighted GT quality
score. A finite-probe covering gap is an estimate, not a certified
continuous upper bound. Lack of a proof or missing diagnostic must be
reported as **unverified**, never silently interpreted as passing.

The goal is that users can **inspect the properties of the reference
they actually employ**, while tests can compare the same declared
properties to MOP-specific acceptance criteria, without duplicating
diagnostic mathematics in two unrelated implementations.

## Implemented initial API and report behavior

```python
assessment = mop.assess_gt(
    reference=None,  # uses in-memory calibrated GT by default
    n_points=None,   # opt-in read-only on-demand analytical candidate
    probes=128,
    seed=0,
    tolerance=1e-9,
)
assessment.summary()
assessment.to_dict()
assessment.validity
assessment.coverage
assessment.regularity
mop.report(show=False)  # includes assessment when calibrated in memory
```

The implementation is read-only and does not calibrate, load sidecars,
change GT, or consume global RNG. `reference` and `n_points`
are mutually exclusive. `mop.report(show=False)` uses the
standard `Reportable` convention and reports that the GT is
uncalibrated when none has been explicitly loaded; it never
implicitly creates one. `assess=False` suppresses detailed
diagnostics in the MOP report.

A frozen `GTAssessment` contains status-tagged `GTProperty`
measurements and a machine-readable `to_dict()`. The initial
validator scope is intentionally conservative: independent
DTLZ1 simplex and DTLZ2–4 sphere membership oracles, plus
corresponding true axis extrema. Other built-in and user MOPs
receive structural/spacing diagnostics but have
`validity=unverified` until independent specialized oracles
are implemented. Component-weighted regularity and continuous
Hausdorff upper bounds are not currently certified.

Decision-space feasibility cannot be inferred solely from
objective GT coordinates and is reported `unverified`.
Covering and spacing are quantitative, but, absent a
scientifically justified MOP/budget-specific threshold,
their acceptance status remains `unverified` rather than
an invented `pass`. Reproducibility cannot be asserted
without independent replay and is also `unverified`.

**This initial implementation is an infrastructure step, not
a claim that all MOP GTs are certified or a release gate has
passed.** The remaining MOP-specific oracle coverage and
scientifically grounded thresholds are required work.

## Property schema and scientific interpretation

| Property | Measurement/observation | Required qualification |
|---|---|---|
| Shape, finite values, cardinality | objective array `(n,M)`, NaN/Inf counts, requested/returned difference | Generic, exact counts |
| Effective uniqueness | exact rows and a tolerance-declared near-duplicate count | Float64 tolerance and distance declared |
| Decision feasibility | evaluate actual generated PS decisions with the original constraints | Not inferred from GT objective points alone |
| True PF membership | independent MOP-specific analytical residual and/or proved membership | **Unverified** without independent global oracle |
| Component/boundary occupancy | points per mathematically known disconnected region, patch, branch, seam and true extremum | Component IDs and expected presence justified by the original geometry |
| Covering / resolution | directed nearest-reference distance from **independent** on-front probes to GT; where available, a proved bound | Declare original or normalized objective-space norm and whether value is empirical or certified |
| Spatial regularity | within-component nearest-neighbor distances, CV or other geometry-appropriate discrepancy | Compare only within declared component/measure; not blindly between intrinsic dimensions |
| Replay / determinism | same-request deterministic PS/PF or audit under declared seed and environment | Not proof of mathematical validity |
| Indicator impact | optional reference-sensitivity analysis for fixed objective populations (IGD, IGD+, HV) | Report normalization/basis and never modify certified baselines |
| Runtime/memory | bounded audit cost and measured wall-time/peak memory where instrumented | Environment, budget, complexity and reproducibility caveats |

A `pass` for a residual only asserts membership under the scope of
the oracle actually used. A `pass` for component occupancy does
not assert uniform coverage, and a finite probe sample cannot
certify continuous Hausdorff bounds. Invalid or infeasible points
on the *actual sampled reference* must be reported separately from
finite sampling coverage defects.

## MOP-specific validators

Validators are registered against the **original exact built-in
MOP implementation** and its configuration, rather than inferring
mathematical correctness from a class name or finite nondominance.
They reuse **independent** published-equation or geometric oracles
established in the issue #119 and #65 studies and exercised by tests,
not `mop.pf()` as its own oracle.

- DTLZ1: complete 0.5-simplex; DTLZ2–4: full positive
  unit sphere, accounting for DTLZ3 multimodal distance and
  DTLZ4 nonlinear inverse angles.
- DTLZ5/6: original full Pareto geometry including `g>0`
  regions for M>=4; do not incorrectly test only `g=0`.
- DTLZ7: independently characterized disconnected record
  components; high M can have more components than the point
  budget, and this **must** remain visible.
- DTLZ8: published constrained line plus M-1 patches;
  feasibility, component occupancy, seam and nonmultiple-N
  behavior.
- DTLZ9: constrained one-dimensional objective curve, endpoints
  and arc-length diagnostic, with explicit float64 feasibility.
- DPF1–5: MOP-specific redundant projections, intrinsic D,
  DPF2 disconnected regions, DPF3/DPF4 independent chaotic
  threshold semantics, and DPF5's separate P/D components.

Validator claims are scoped to proven and tested parameter
ranges. An unsupported dimension or missing independent oracle
is `unverified` and must not automatically count as an accepted GT.

## Acceptance tests and CI policy

Use the **same diagnostic implementation** in the API and new
independent test modules. The tests must not simply compare the GT
against itself or use the same sampler implementation to certify
its own correctness. At minimum:

1. Exercise a scientifically justified finite matrix of supported
   MOP configurations, with objective counts including representative
   M=2,3,5,10 where valid; DPF intrinsic D and constrained
   N/nonmultiple N must be covered explicitly.
2. Run structural validity, independently proved PF membership,
   constraints, expected extremes/components and reproducibility.
   Inject an intentionally off-front candidate, a missing component
   and a duplicate-heavy sample and verify the appropriate
   diagnosis **fails**.
3. Quantify covering and regularity at fixed declared budgets and
   independent probes. Compare to approved **MOP/component-specific**
   limits, not an arbitrary universal quality threshold.
4. Exercise reference sizes and calibration defaults
   (2,000 DTLZ, 10,000 DPF) in a separate appropriate integration or
   opt-in scientific certification suite if the full cross-product
   makes routine unit tests expensive. Unit tests should remain
   fast, with deterministic analytical oracles.
5. A critical mathematical failure (off-front/infeasible or
   a provably absent required component at an eligible budget)
   fails the applicable gate. A scientific claim without enough
   evidence is labeled `unverified` and **blocks certification of
   that claim**, but does not spuriously report a numerical defect.
   Warn on quality degradation using independently justified
   budget- and geometry-specific thresholds.
6. Preserve all pre-existing regression tests and immutable
   numerical/performance references. Any proposal to revise an
   existing regression, baseline or accepted ADR has to be
   separately warned about and explicitly approved beforehand.

The acceptance matrix must be explicit and versioned. Its thresholds
must be scientifically justified and not obtained by copying current
results into the expected values. If the available audit cannot
support a pass/fail threshold, expose `unverified` and retain
the evidence for separate review. A release-readiness assertion
requires a source-bound completed matrix, not merely an issue
status or a green generic CI run.

## Relationship to existing decisions

- ADR 0006: native MOP-owned mathematical analytical PS/PF.
- ADR 0018: existing explicit calibration densities (2,000/10,000).
- ADR 0056: protected certification artifacts and versioned
  reference migration.
- ADR 0057: accepted GT generation/calibration ownership; this
  proposal **adds an optional inspection and scientific acceptance
  contract** but does not silently replace its current runtime
  policy or historical regression conditions.
- ADR 0058: certified scalar HV denominators require independent
  geometry and provenance; a GT quality assessment is **not by
  itself** a new proof of continuous HV or a license to whitelist
  other analytical denominators.

The current technical specification is `docs/gt.md`; the
original investigations remain in `docs/research/gt-issue119/`.
Until this ADR is explicitly accepted and implemented, the proposed
`assess_gt()` API and quality status fields are not available
as public guarantees.

## Remaining implementation and scientific validation tasks

- Final API name and return type, including how MOP-specific
  component data and unavailable diagnostics are represented.
- Mechanism for pluggable independent oracles for custom MOPs
  without trusting arbitrary user-declared certification.
- Scientific acceptance thresholds and supported configuration
  matrix by MOP, with bounded CI resource cost.
- Whether an approved quality result may be persisted alongside
  (but not into the existing schema of) the GT sidecar; initial
  implementation should be **in-memory, read-only, opt-in**.
