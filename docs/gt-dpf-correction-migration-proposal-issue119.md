<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Proposed DPF3/DPF5 correction and versioned migration — decision draft (#119)

**Status: proposal for maintainer review, not an approved change.** This document deliberately contains no implementation instructions authorizing production modification. Source: read-only PR #139; current runtime, old tests, ADRs, frozen references and certificates remain unchanged.

## Verified evidence and separate defects

**DPF3 (evaluation and reference generation):** paper-required increasing ordering of chaotic thresholds is absent for M−D>=2. With D=2 and x1 uniform, float64 direct `1−cos((π/2)x1^100)` erases 423 mathematically positive values for 512 sampled decision coordinates; the stable `2sin²(θ/2)` still erases 12 due to finite representability. Current `pf(512)` returns 88 after an in-sample nondominance filter. A locally nondominated subset cannot certify global Pareto membership or complete coverage. Changing only the trigonometric expression does **not** prove recovery of a valid GT.

**DPF5 (published equation):** independently reconstructed published beta/P and gamma/D maps satisfy `sum(F_i²)=(1+g)²`, at g=0 and g>0, for sampled general M>D>=2. Current beta/P disagrees for multiple dimensionalities including M5,D2; gamma/D also disagrees at D4 (e.g. M6,D4). Exact x1=1/3 currently belongs to P, but the publication assigns it to D. Current M3 grid silently returns square-grid counts such as 484 for a request of 512, and many duplicate objectives in the degenerate branch. These are distinct implementation, boundary, and sampling defects requiring distinct tests.

**Measured scientific impact:** paired research study using identical decision matrices finds up to ~0.85 coordinate deviation for the sampled M5,D2 GT; when fixing the *objective population* bitwise and changing only the in-memory reference, the original MoeaBench IGD/IGD+ and GT-based absolute HV ratio can change (e.g. seed 101, M5,D2: +0.02731 / +0.00270 / +0.02663). These values are **exploratory finite-sample sensitivities**, not accepted replacement calibrations or proof that one candidate GT is globally correct. Existing IGD(+) uses reference-specific zero-to-one normalization, and both HV computations use the same fixed bounds.

## ADR and regression compatibility gate

- **ADR 0006:** requires MOP-owned analytical PS/PF with correct mathematical benchmark semantics, vectorized evaluator and deterministic reproducibility. A mathematically justified correction should preserve these architectural contracts, but changes the *numeric behavior*.
- **ADR 0017:** requires scientifically valid, independently audited analytical Ground Truth and versioned calibration evidence.
- **ADR 0018:** records historical 2000 DTLZ / 10000 DPF calibration densities and sampled-reference HV meaning. Do **not** change these runtime defaults or reinterpret historical records as already calibrated with the revised equations.
- **ADR 0057:** approves on-demand MOP-specific GT, separate `pf()` and `calibrate()`, and an automatic geometry-aware budget *in principle*, but no operational numerical rule or migration. Do **not** silently select a new default calibration budget during this mathematical repair.
- **ADR 0056:** requires immutable, versioned numerical/performance certification evidence. Never relabel, overwrite, regenerate or weaken historic regression baselines merely to make new source pass.
- **Existing regression tests:** keep original files and thresholds unchanged until their purpose, science and provenance are reviewed. Some checks test metric invariants and might remain valid; others may correctly flag expected formula-induced differences. Any planned modification to an existing regression test requires advance warning and explicit maintainer authorization.
- Any discovered incompatibility with the literal normative wording of an accepted ADR must be reported before implementation and is a **hard approval gate**.

## Proposed sequence and explicit approval boundaries

**Stage A — finish independent mathematics (research-only; permitted without runtime changes).** Verify original published beta/gamma and DPF3 equations for each supported M/D/K/N, threshold equality and nonzero g, including edge cases. Prove Pareto membership, complete connected/mixed-dimensional support and extremes analytically where possible, or clearly classify configurations as uncertified. Validate with separate new research tests and prospective precision/geometry tolerances. A sphere identity alone is not a Pareto proof.

**Stage B — controlled candidate evaluator (not yet authorized for production).** In a **separate experimental module** recreate the exact published equations. Reuse identical fixed decision vectors, independently derived component outputs and high-precision landmarks. Report normalized and unnormalized objective errors, strict-branch handling, minimum/maximum objective vectors, feasibility and runtime cost. For DPF3 compare direct float64, stable float64 and high precision; explicitly decide scientific treatment of unrepresentable values, rather than assuming exact cardinality is achievable by subtractive formula change.

**Stage C — independently construct/verify reference samplers (not yet approved as public defaults).** For each branch/component, derive viable PS and PF, certify globally nondominated membership and point coverage, report requested versus returned and **unique effective** objective counts, extreme preservation, deterministic seeds and quality/cost. Treat DPF3 chaotic projection and DPF5 mixed-dimensional beta/gamma branches separately. Compare reference error against independent analytic constructions, not only against a denser copy of the same possibly wrong generator. Establish quantitative stopping rules before proposing `calibrate()` policy.

**Stage D — pre-change impact review and versioned migration decision (requires maintainer approval before production).** Prepare an explicit file-by-file diff plan, including `moeabench/mops/DPF3.py`, `DPF5.py`, `base_dpf.py`, affected `ps()/pf()` and any `calibrate()` orchestration. Locate the historical consumers and current fixtures, especially `tests/stability_reference_gt.json`, `tests/stability_gt_geometry_reference.json`, `tests/unit/test_dpf3_emd_regression.py`, calibration sidecars and preexisting reference/certificate archives. Record legacy and candidate MOP formulation IDs/configuration and GT provenance; keep historical outputs readable and explicitly labeled as legacy-formula data. Describe public numerical behavior and release notes. List each existing regression that might need an authorized *new* expected target; preserve old fixtures until separately approved.

**Stage E — implementation and certification (only after Stage D approval).** Submit a separate narrowly scoped production PR with independent tests, original regressions run unmodified, fixed-decision paired numerical comparison, and reproducibility and numerical-stability CI. Where old tests fail for documented scientific corrections, report the failure; do not silently patch or delete them. After explicit decisions, add versioned successor expectations and a clean history-preserving migration. Re-run reference-dependent indicator sensitivity and the approved numeric certification. Handle any new performance baseline under the separate ADR 0056 procedure and **never** infer it from a numerical certificate.

**Stage F — geometry-aware automatic `mop.calibrate()` budget (separate ADR 0057 operational approval).** Only after mathematics and sampler reliability are established, review error targets, max cost, default-budget selection and provenance. Do not conflate the mathematical evaluator fix with this policy change.

## Approval request to be presented after Stage A–D evidence

The prospective maintainer decision will separately ask whether to (1) correct published DPF3/DPF5 evaluation equations in production, (2) change PS/PF sample generation/cardinality contracts, (3) introduce versioned successor scientific GT references while preserving old archives, (4) authorize *specifically named* existing regression edits, and (5) approve a new calibration budget policy. **No approval is presumed** from instructions to continue the audit.

## Present action

Continue with read-only research only. PR #139 stays Draft and unmerged, with no change to old regression tests, existing ADR files, runtime evaluation or frozen numerical GT and performance references.

## Progress update — 2026-11-20: proven DPF3 D=2 finite coverage

Stage A's **global Pareto membership and complete analytical parameterization** is now established for *both* published DPF3 (increasingly sorted eta) and published DPF5 (both branches), by independent algebraic proofs. Production mathematical agreement remains falsified for several M,D configurations; neither proof certifies the incumbent evaluator. The open scientific requirement in Stage C concerns **finite GT accuracy**, not continuous PF completeness.

A first constructive Stage C theorem is now available for **DPF3 D=2**: endpoint-inclusive angular grid of n points, mathematically exact published F and sorted eta, guarantees directed PF-to-GT Euclidean covering radius and Hausdorff distance **<= pi/[4(n−1)]**. Independent numerical checks for n=128 and 512 and M=3/5/10 produce exactly n distinct published-objective float64 points (without claiming a rigorous rounding bound) against only 23/88 returned by the current production PF for those n. See [analytic finite covering proof](gt-dpf3-d2-angular-cover-proof-issue119.md) and the new isolated script/tests. This initial D=2 result has since been generalized in a separate proof to all D>=2 by tensor angular grids; see [higher-dimensional angular-cover proof](gt-dpf3-nd-cover-and-acceptance-issue119.md). Neither result selects a calibration budget or certifies a floating-point rounding envelope.

**Proposed scientifically auditable finite-GT gate (not yet a normative numeric threshold):** verify correct published evaluator and analytically complete global PF membership first; for a candidate finite sample record actual/unique cardinality and exact PS provenance; demand a justified continuous covering/error bound in the *declared metric space* where possible, explicitly distinguishing a bound from stochastic probe estimates; check objective extrema, disconnected or mixed-dimensional components, numeric representation error, deterministic seeds and bounded runtime/memory. For DPF5 no universal P/D mixing probability is implied by the Pareto-set geometry; the demonstrated relative branch-budget tradeoffs must remain a separate explicit choice. ADR0057's automatic `calibrate()` budget awaits chosen tolerances, cost bounds and maintainer approval.

**Warning and approval boundary unchanged:** it is not scientifically valid to silently change the production DPF3/DPF5 evaluator and then regenerate old regressions or frozen references to restore green CI. Before doing so, present exact affected existing regression names and ADR conflicts and seek express authorization; preserve historical versioned GT/certification data. The current research-only PR #139 does not grant implementation approval.

## Pre-merge scientific readiness update — 2026-11-20

The higher-dimensional DPF3 covering theorem is now in [the independent report](gt-dpf3-nd-cover-and-acceptance-issue119.md): `d_H(PF,GT) <= sqrt(D−1)*pi/[4(k−1)]` for a k-point-per-angle tensor-grid image under the published **sorted-threshold** equation, in exact real arithmetic. The grid has `k^(D−1)` nominal rows and may have fewer unique objective vectors at hyperspherical boundary singularities. The empirical high-precision inverse-angle round-trip study records order 1e−14 discrepancies, **not** a formally enclosed floating-point guarantee.

**Research-PR merge interpretation:** the research PR can be reviewed and merged as **scientific evidence only** after all checks pass and the maintainer accepts its research-only scope; this does not authorize a production mathematical correction, a calibration-policy change, replacement of frozen baselines, or changes to existing regression tests. Future production migration must use the separate Stage D/E gate and may require explicit additional authorization. No accepted default error, budget, cost cap, DPF5 branch mixing law or float64 certified tolerance has been selected.
