# Pre-merge unit test, ADR and regression audit — PR #88

Date: 2026-11-13
Repository: monacofj/moeabench
Branch: hotfix/0.17.0-contract-recovery
Scope: read-only audit of current tests, historical tests in `old-fair`, selected operative ADRs and the local `python test.py --deep` log. **No tests, golden baselines, production code or ADRs were changed by this audit.**

## Evidence and limitations

- PR #87 temporarily removed FAIR and Q-score functionality; PR #88 restores FAIR-independent GT calibration and H_abs.
- User-provided full deep log: 325 collected; 292 unit passed, 14 integration passed, 18 stability passed, one stability test **skipped** (324 passed / 1 skipped / 0 failed).
- GitHub Actions at hotfix commit `a710e43`: JOSS validation and a **scoped** pre-FAIR core numerical comparison succeeded.
- This audit checks core scientific/public contracts implicated by the cleanup; it is not a proof that every historical ADR (including superseded FAIR decisions) has a test for every clause.

## Blockers / high priority

### A. Deep statistical regression suite is non-operative

`tests/test_stability_deep.py` uses `MOEABENCH_RUN_DEEP` and reads `tests/baselines_v0.8.0.csv`. The latter is absent in **both** `old-fair` and the hotfix. The empty `CONFIGS` is collected as `test_deep_statistical_quality[NOTSET]` and skipped. Even enabling the environment variable alone cannot restore the missing baseline. Thus the successful `--deep` run does **not** establish the advertised deep 30-repeat statistical-quality regression coverage. Decide whether to provide properly reviewed reference artifacts or redesign/retire this obsolete case. Do not regenerate target numbers merely to pass.

### B. Historical calibration regression tests are not collected

`tests/unit/verify_gt_protocols.py` contains three `test_protocol_*` functions but is not named `test_*.py`, so it is absent from the 325 collected tests. It covers default analytical calibration, `source_gt` from a CSV path, and empirical `source_search`. The new `test_gt_calibration_independent.py` covers analytical, in-memory `source_gt`, cache/reload, validation, density policy and legacy GT-only migration, but **not** the CSV-path or empirical-search paths. Add collected replacement tests before retiring the script. The current file writes to the working directory and contains cleanup only under its `__main__` entrypoint, so it should not simply be renamed verbatim.

### C. Accepted ADR 0048 and public API tests disagree

Accepted `docs/adr/0048-public-clinical-metrics-and-audit-context-parity.md` explicitly makes `mb.clinic` and direct clinical/quality metrics public. The current `tests/unit/test_api_breakage_expected.py` and `test_api_surface.py` assert `not hasattr(mb, "clinic")`. The FAIR removal may deliberately suspend that API, but the architectural exception/supersession has not been recorded in ADR 0048. **Resolve this decision with maintainer authorization before merge; do not silently change tests or ADR.**

### D. No demonstrated independent unit enforcement of conventional H_rel (ADR 0047)

`tests/unit/test_hypervolume_reference.py` exercises raw geometry thoroughly and compares scale-independent diagnostics; new tests defend MOP-owned `H_abs`. The broad denominator/trajectory tests under `tests/unit/test_ordinal_hypervolume.py` concern **ordinal** HV, not conventional `mb.metrics.hv`. Add direct conventional H_rel tests: best individual external reference (not pooled union), single shared **final** denominator for all generations, self-reference best final run, legal >1 values without clipping, zero/nonpositive denominator, and backend/seed parity. Do not conflate issue #89's future UX research with current ADR 0047.

### E. Frozen analytical-metric regression protection is reduced (ADRs 0031/0050, docs/test.md)

PR #87 removed `tests/calibration_reference_targets_v0.13.2.json`, `tests/test_golden_dtlz6.py`, and many FAIR-specific clinical tests/assets. It is appropriate to exclude FAIR/Q-score tests from a FAIR-free release, but the accepted scientific testing architecture calls for **both** numerical measurement regression and solver-front reproducibility. The retained `tests/test_stability_basic.py` and `tests/test_stability_smoke.py` protect sampled geometry / frozen fronts, not a broad set of frozen outputs for conventional HV/GD/GD+/IGD/IGD+/EMD across MOP families. PR #88's core-comparison probe checks selected outputs against pre-cleanup code, but does not replace permanent frozen regression targets. Inventory and preserve the FAIR-independent numerical components of the removed reference dataset, under reviewed fixed inputs and expected outputs.

## Medium-priority coverage gaps

1. **MOP GT persistence/sidecar failure cases:** `moeabench/mops/calibration.py` supports analytical, external CSV/array and empirical search sources; tests do not yet comprehensively exercise empirical search, invalid sidecar schema/JSON, write failures, `force` overwrite policy, package-relative default path under read-only installation, or GT state after save/load. ADRs 0011/0035 and the historical GT API merit contract guards.
2. **H_abs multi-MOP/multi-run validation:** Add negative tests for mixed MOPs, incompatible dimensions, differing GT calibration contexts, empty/zero-GT denominator, numerical backend parity and all data/container input variants. Current `_mop_from_evaluated_data` checks basic MOP name/M/type homogeneity in lists; there is no demonstrated guard for distinct calibrated GTs with the same MOP class/name/dimension.
3. **Persistence modes (ADR 0011):** `test_environment_verification.py` tests `config` persistence and provenance; `test_experiment.py` and integration smoke test basic save/load; no direct unit test was found for the distinct `all/config/data` semantics, omission of unrequested payloads, export manifest and CSV fidelity, or legacy archive compatibility.
4. **Float64 policy (ADR 0021):** Unit tests exercise several numerical shapes/geometries but do not establish a path-wide no-downcast invariant for input ingestion, MOEA trajectories, persistence, and metric intermediates. An explicit precision audit is mentioned in ADR 0021 but was not demonstrated by the collected unit suite.
5. **Analytical GT sampling drift:** `tests/test_stability_basic.py` intentionally replaced frozen DTLZ2/3/4 sample-to-sample geometric checks with replay determinism and hypersphere invariants; `tests/unit/test_dtlz_hypersphere_sampling.py` adds symmetry/moment/extrema checks. This is defensible as a geometric approach, but these tests cannot detect every distributional/sampling change. Decide on stable distribution-sensitive sentinels without naively restoring an invalid cross-sample Hausdorff threshold.
6. **API negative assertions:** `tests/unit/test_api_breakage_expected.py` duplicates some namespace absence assertions from `test_api_surface.py`, while newer GT-specific features lack an equally complete public-surface matrix. The latter file's absence assertions are meaningful only after resolving the ADR 0048 exception.
7. **Core comparison coverage:** `scripts/audit/core_probe.py` and the compatibility workflow are useful targeted sentinels, not an exhaustive behavioral equivalence assessment. Expand long-term frozen checks rather than implying a universal equivalence proof.

## Obsolete / misleading candidates (do not delete without review)

- `tests/unit/debug_dpf3_emd.py`: manual exploratory debugging script under the unit-test directory, not pytest-collected; distinguish from a formal regression guard and consider relocating/removing after retaining any independently useful assertion.
- `tests/unit/verify_gt_protocols.py`: valuable historical contract intent, but not collected and writes fixed files in the current directory. Replace carefully with safe pytest fixtures, then retire.
- `tests/test_stability_deep.py`: currently a non-operative skipped historical baseline path; retain only if its inputs and statistical acceptance criterion are made viable and reviewed.
- `tests/unit/test_metrics.py`: inserts the repository root into `sys.path` and contains outdated narrative comments; its IGD case asserts only nonnegative values and metric name. Keep its actual numerical HV check, but replace weak/stale assertions with specific independent references when authorized.
- `tests/unit/test_api_breakage_expected.py`: overlaps `test_api_surface.py`, and its `mb.clinic` assertion conflicts with the still-accepted ADR 0048. Resolve the architectural decision first.
- `tests/unit/test_absolute_hv_contract_guard.py` docstring says no pre-existing regression test was edited, now inaccurate after the separately authorized PR #88 edits. Documentation-only correction can wait for explicit review.
- `docs/test.md` / ADR 0050 refer to `moeabench/diagnostics/resources/certified_environment.json` for certification, whereas the runtime reads `moeabench/resources/certified_environment.json`; documentation needs reconciliation.
- ADR 0031 discusses a calibration-target regression tier that the temporary FAIR-free release no longer runs. Its current/superseded applicability needs explicit documentation.

## Coverage that is present and should be preserved

- DTLZ/DPF analytical Pareto geometry and sampling tests, including disconnected DTLZ7 and DPF2 sources.
- NSGA-II/NSGA-III/U-NSGA-III deterministic seed, wrapper, mutation and constraints checks (ADR 0045).
- Raw HV geometry, exact/Monte Carlo engine choices, bounding-box diagnostics and reports (ADRs 0020/0042/0043).
- New MOP-owned GT and H_abs guards (ADRs 0018/0019/0035), including explicit vs implicit geometry reference.
- Statistical compare aliases/results, reporting format, views, population algebra, persistence/provenance and environment version/status checks.
- Frozen front/GT stability fixtures in `tests/stability_reference_fronts.json` and `tests/stability_reference_gt.json`.

## Decision gate

**Do not merge PR #88 solely on the green 324/325 result.** Before merge: decide ADR 0048 suspension, recover collected GT calibration source coverage, add missing conventional H_rel ADR 0047 guards, and decide how to handle the disabled deep statistical test / missing baseline. Longer-term numerical baseline, persistence and float64 gaps can be prioritized separately only through an explicit risk decision. Any test modification requires prior warning; any change incompatible with an accepted ADR requires express authorization.
