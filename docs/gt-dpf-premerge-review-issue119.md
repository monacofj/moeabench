<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Research PR #139 — pre-merge scope and scientific review (issue #119)

**Scope:** this checklist applies exclusively to merging **read-only independent research evidence** from PR #139. It is **not** approval to change the numerical definitions of DPF3/DPF5 in production, regenerate ground truth, alter existing regression tests, relax an ADR, or select an automatic calibration budget.

## Repository diff inventory

The complete comparison `main...research/119-dpf3-dpf5-branch-diagnostics` was reviewed on 2026-11-20. All changed paths were **new** files under `scripts/audit/`, `tests/unit/test_gt_*_issue119.py`, `docs/gt-*issue119.md`, or the new isolated workflow `.github/workflows/gt-dpf-branch-research-119.yml`. No production `moeabench/` file, existing regression, accepted ADR, existing CI workflow, frozen numerical reference, GT archive or baseline is modified.

One pre-merge integration defect was caught: new DPF3 D>2 high-precision script imported optional `mpmath` during pytest collection, breaking the *otherwise unchanged* general unit/numerical workflows. It was corrected by moving the import inside the research precision function; the two research tests involving the previously isolated precision module and the seven new higher-D tests skip locally when mpmath is absent. The isolated research CI explicitly installs mpmath. **The final complete CI on the new head must be green before the PR can be merged.**

## Scientifically established

1. **Published DPF3:** for increasingly sorted chaotic thresholds and standard box feasibility, exact global PS is `g=0` and PF is its complete published objective image. Its scalar ordered min/max projection is order-reflecting. The actual source uses unsorted thresholds in several dimensions and suffers first-objective float64 cancellation.
2. **Published DPF5:** exact global PS is `g=0` and PF is its full two-branch published beta/gamma image. The actual beta/P implementation deviates for some larger M; gamma/D also deviates for intrinsic D=4, and exact `x1=1/3` branch assignment differs. The established global theorem **does not** certify the current evaluator.
3. **Finite DPF3 published model:** with a k-point-per-angle tensor-grid image on the exact g0 PS, a **real-arithmetic** Euclidean Hausdorff bound `sqrt(D−1)*pi/[4(k−1)]` follows from the angular map's 1-Lipschitz property, for all intrinsic D>=2. Tensor complexity is `k^(D−1)`. Some boundary fibers yield duplicate objective vectors; float64 error is estimated by high precision but **not certified by interval arithmetic**.
4. **Finite DPF5 samples:** current sampler has notable cardinality/duplicate problems for M3,D2 and uneven P/D conditional coverage in higher dimensions. Equal-effective-budget Sobol and 50%/two-thirds P allocations have mixed outcomes. No universal branch probability, optimal allocation, or accepted reference sample is established.
5. **Numerical sensitivity:** fixed-decision and bitwise-frozen objective-population comparisons show that correcting the evaluator and replacing the GT are *distinct* changes and can affect IGD, IGD+ and relative HV. Historical references must remain versioned and traceable.

## Explicitly unapproved post-merge work

- Any correction to `moeabench/mops/DPF3.py`, `DPF5.py`, `base_dpf.py`, or `ps()`/`pf()`/`calibrate()`.
- A published-formula successor GT, a new numerical certificate, replacement of `tests/stability_reference_gt.json`, `tests/stability_gt_geometry_reference.json`, or similar historical baselines.
- Changes to any existing regression test (must give advance warning and receive explicit approval), or any contradiction to an accepted ADR (must warn and receive authorization **before** proceeding).
- Adoption of an automatic accuracy/cost budget under ADR0057, an epsilon target, or mixed-dimensional DPF5 distribution.

## Pre-merge checklist

- [x] Diff inventory: all changes are research additions, with no production or existing-test edits.
- [x] Mathematical hypotheses separated from the actual current evaluator.
- [x] Analytic proofs and explicit real-arithmetic/float64 qualifications documented.
- [x] Research reproducibility scripts and independently introduced tests present.
- [x] Optional high-precision dependency isolated from normal test collection.
- [x] Original frozen references, accepted ADRs, and historical regression tests preserved.
- [x] Isolated research CI successful on audited commit `89235a1c` (94 tests).
- [x] Complete unit-contract regression CI successful on audited commit `89235a1c`.
- [x] Complete numerical stability CI successful on audited commit `89235a1c`.
- [ ] Maintainer authorizes the **research-only** merge; a separate approval is required for any later production correction.

**Merge posture:** all three CI workflows passed for audited commit `89235a1c`; PR #139 is now **Ready for review** (GitHub `mergeable=true`), and remains **OPEN and unmerged**. The subsequent checklist-only commit must pass its own CI before final sign-off. Keep the PR unmerged until the user expressly authorizes the merge. Issue #119's future production migration and calibration decisions remain open even after a research-only merge.
