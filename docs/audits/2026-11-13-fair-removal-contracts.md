# PR #87 — Contract preservation audit (initial findings)

**State:** Investigating; not a release sign-off.  
**Reference:** `old-fair` at `0f51689807680cc07aeac80e4da6a542e92cd696` (pre-cleanup 0.16.1 state).  
**Subject:** `main` at `9928ff3f03e44625aa92d66d9e35d6d050f4133a` (merged 0.17.0).  
**Hotfix base:** `main` above.  `old-fair` and `main` are not rewritten.

## Ground rules

Accepted ADRs are authoritative. No relaxation, replacement, or modification of
existing regression tests is implicit in this audit. A changed test must be
reviewed individually against its original and its governing ADR. Historical
FAIR-only tests and frozen resources remain accessible in `old-fair`.

## Repository-wide inventory

- **9 existing Python modules under `moeabench/` changed:**
  `__init__.py`, `core/experiment.py`, `core/save.py`,
  `environment_verification.py`, `metrics/evaluator.py`,
  `mops/base_mop.py`, `system.py`, `view/__init__.py`,
  `view/topo.py`.
- **12 existing tests changed:** `tests/conftest.py`,
  `tests/integration/test_compare_pipeline.py`,
  `tests/integration/test_full_walkthrough_smoke.py`,
  `tests/integration/test_metrics_pipeline.py`,
  `tests/integration/test_view_dispatch_pipeline.py`,
  `tests/test_stability_basic.py`, `tests/test_stability_smoke.py`,
  `tests/unit/test_api_breakage_expected.py`,
  `tests/unit/test_api_surface.py`,
  `tests/unit/test_dtlz_hypersphere_sampling.py`,
  `tests/unit/test_hypervolume_reference.py`,
  `tests/unit/test_report_contract.py`.
- **18 test assets deleted** (including 16 Python scripts and two JSON frozen
  FAIR calibration targets). Historical assets remain in `old-fair`.
- Additional removed content includes FAIR diagnostics, datasets, examples,
  calibration scripts, and notebooks. They are part of the full PR #87 diff.

## Confirmed contract defects

1. **HV absolute:** `metrics/evaluator.py` now rejects calls without `ref`
   and uses the external reference as denominator. This contradicts the
   MOP-derived Ground Truth interpretation and the separate `ref` geometry
   specified by ADR 0019 / 0035 / 0047. High severity.
2. **Regression test inversion:** `test_absolute_hv_requires_explicit_reference`
   and `test_absolute_hv_uses_supplied_reference` were added to ratify that
   unintended behavior. Three integration tests were altered to supply
   `ref=canonical_gt` to `scale='abs'`. High severity.
3. **Silent UI behavior:** `view.topology(markers=True)` was left as an
   accepted parameter without its FAIR implementation; `view.density(domain='clinic')`
   fell through to an unrelated view. These require explicit failure.

## Other changes — classification and remaining evidence

| Code element | Current assessment | Evidence/decision outstanding |
|---|---|---|
| `__init__.py` | FAIR-only surface removal | Verify remaining non-FAIR exports |
| `core/experiment.py` | FAIR-only `diagnose=True` removal | Verify `diagnose=False` output / seeds unchanged |
| `core/save.py` | Removed FAIR baseline hash | Check legacy compatibility; ADR 0032, 0050 |
| `environment_verification.py` | Resource relocation | Verify certification and packaged JSON; ADR 0050 |
| `metrics/evaluator.py` | **Confirmed regression** | Restore MOP GT and benchmark-specific density before merge |
| `mops/base_mop.py` | Removed clinical calibration | ADR 0035 mentions calibration: explicit decision required |
| `system.py` | Version metadata | Check version/citation consistency |
| `view/__init__.py` | Removed clinical views | Reject unsupported `domain` explicitly |
| `view/topo.py` | Removed clinical markers | Reject `markers=True` explicitly |

## Removed tests: preliminary scope

- FAIR-only suites: `test_clinic_pipeline`, `test_example13_consistency`,
  `test_topology_marker_alignment`, `test_mop_calibration`,
  `test_fair_conformance_issue75`, `test_audit_pf_sources_issue75`,
  `test_dpf5_closeness_protocol_issue81`,
  `test_headway_anchors_issue75`,
  `test_headway_pointwise_issue75`,
  `test_issue81_ecdf_compression`,
  `test_issue81_full_calibration_q`,
  `test_outward_half_normal_blur_issue83`,
  `test_qscore_ecdf`,
  `test_stage_closeness_adr0037`, and DTLZ6 s_K baseline integrity.
- Mixed suites must retain independent invariants: `test_stability_basic`,
  `test_stability_smoke`, `test_api_breakage_expected`,
  `test_dtlz_hypersphere_sampling`, `test_hypervolume_reference`.
  `test_report_contract` lost only clinical report cases.
- **Not approved:** editing, rebaselining, or deleting any further regression
  tests, or weakening any frozen numerical tolerance.

## ADR conflict requiring a design decision

- ADR 0018 defines **2,000 DTLZ / 10,000 DPF** sampled reference points for
  the scientific benchmark calibration context. The tentative 500-point HV
  solution in `review/joss-1.0.0` is **not approved** as a replacement.
- ADR 0035 describes `mop.calibrate()` as required to resolve `H_abs`, while
  FAIR separation removed that method. The scientific requirement that the
  GT belongs to the MOP remains; how to retain the canonical GT independently
  must be resolved and documented **before** altering that contract.

## Independent validation evidence (2026-11-13)

- [Core equivalence workflow #43491835518](https://github.com/monacofj/moeabench/actions/runs/43491835518):
  **PASSED**. Compared both implementations in the same Python 3.12 runtime
  on identical input data and seeds, covering sampled DTLZ/DPF fronts,
  conventional HV raw/rel, GD/GD+, IGD/IGD+, EMD, and NSGA-II trajectories.
  This check does **not** establish full codebase equivalence; it deliberately
  omits the known-broken absolute HV semantics.
- [Public API and numerical validation #43491835529](https://github.com/monacofj/moeabench/actions/runs/43491835529):
  **FAILED, expected**. Three new contract tests demonstrate existing
  absolute-HV regressions:
  - `test_absolute_hv_without_ref_uses_mop_gt`: explicit `ref` wrongly required.
  - `test_absolute_hv_ref_only_selects_bounds`: computed ratio differs from
    MOP-ground-truth denominator (0.909835 vs 1.320291 in the controlled case).
  - `test_absolute_hv_accepts_multiple_normalization_reference_sources`:
    multiple sources wrongly rejected.
  The existing inverted `test_absolute_hv_requires_explicit_reference` test
  still **passes** in the same run, illustrating why prior green CI was
  insufficient.
- Original regression tests remain unmodified by this hotfix branch. The
  added contract tests are in a separate new file.

## Pending before any merge

1. Resolve ADR 0018 / 0035 requirements with explicit user authorization.
2. Restore HV regression expectations; preserve and check core golden tests.
3. Compare 0.16.1 vs hotfix on identical inputs, seeds, reference geometry,
   numeric metrics, persistence, and visual APIs.
4. Run core CI and isolated compatibility checks; record all failures.
5. Obtain review of the final diff; do not merge automatically.

## Continuation: independent verification at PR #88 head (2026-11-13)

- [Core equivalence #43492145465](https://github.com/monacofj/moeabench/actions/runs/43492145465) **PASSED** against the frozen pre-cleanup commit; this remains a *scoped* probe, not a declaration of full numerical equivalence.
- [JOSS cleanup validation #43492145533](https://github.com/monacofj/moeabench/actions/runs/43492145533) completed **278 passed, 3 failed, 33 deselected**, one warning in the unit batch. All three failures are the newly introduced H_abs contract guards. The pre-existing test named `test_absolute_hv_requires_explicit_reference` passed, despite embodying the unintended new contract. In other words, no other failure was observed in the executed unit subset, but the overall job is correctly red.
- Read-only review reconfirmed ADR 0019's `H_abs = H_sol / H_GT` distinction and ADR 0047's separation of reference geometry from scaling. ADR 0035 historically requires `mop.calibrate()`, and ADR 0018 prescribes 2,000/10,000-point reference density for DTLZ/DPF calibration. Neither requirement has been overridden or approved for alteration.
- **Safety gate:** Do not silently switch `H_abs` to a 500-point reference, treat `ref` as the GT, or modify/remove historical regression assertions to make CI green. Before implementation, obtain a precise design decision about canonical MOP-owned GT availability after FAIR removal. Existing tests and ADR files remain untouched in the hotfix.

## Continuation: MOP ownership fix and CI (2026-11-13)

- Commit [0f94c9d](https://github.com/monacofj/moeabench/commit/0f94c9df1b3f7969fc8a7de3c3df267f9697bd11) fixed `_mop_from_evaluated_data`: an `experiment` exposes delegated `pf`/`evaluation` methods and was incorrectly returned as the MOP before looking at its owned `.mop`. Resolution now prioritizes `.mop`.
- [Core equivalence #43514311644](https://github.com/monacofj/moeabench/actions/runs/43514311644): **PASSED**. Same scoped numerical equivalence check as before; absolute HV is not part of the equivalence comparison.
- [JOSS cleanup validation #43514311885](https://github.com/monacofj/moeabench/actions/runs/43514311885): **285 passed, 7 failed, 33 deselected**. Three new GT/HV integration guards now pass after the ownership fix.
- **Remaining failures:** three *new* `test_absolute_hv_contract_guard.py` checks use a mocked analytical `pf` but do not call `mop.calibrate()`; the restored contract requires calibrated GT. Four failures are in the pre-existing `test_hypervolume_reference.py`, whose pre-cleanup 0.16.1 tests provisioned a MOP plus frozen GT for `abs`, whereas 0.17.0 removed those fixtures and introduced tests requiring an explicit `ref` denominator. Their changed expectations cannot simply be treated as canonical.
- **Change control:** no pre-existing regression tests or ADRs were edited by this branch. Before repairing the remaining four tests, explicitly notify the maintainer of the exact intended test changes and get authorization. The three *new* guards also need appropriately prepared GT fixtures; preserve their semantic assertions. Do not weaken tolerances or assume a green CI is sufficient.
- Separate research issue [#89](https://github.com/monacofj/moeabench/issues/89) captures `H_abs > 1` numerical/interpretability concerns, potential impact on `rel`, and projection-on-GT ideas. It does not change the current ADR-mandated hotfix contract.
