# New API Implementation Plan (Alpha -> Beta)

## Objective

Deliver a clean, consistent beta API with reduced surface area:

1. `clinic.audit(...)` as the primary clinical entry point.
2. `stats.perf_compare(...)` and `stats.topo_compare(...)` as comparison umbrellas.
3. `view` namespace organized by chart type (`topology`, `density`, `history`, etc.).
4. `report(show=True, full=False)` as the single reporting contract.
5. Removal of `summary()` and legacy clinical endpoints.

---

## Phase 1: Freeze Beta Contract

Create `docs/beta_api_contract.md` as the source of truth:

1. Canonical namespaces (`clinic`, `stats`, `view`).
2. Allowed public functions.
3. Final signatures and return objects.
4. Expected errors for removed calls.

Done criteria:

- Contract approved before implementation proceeds.

---

## Phase 2: Public Namespace and Exports

Target files:

- `moeabench/__init__.py`
- `moeabench/diagnostics/__init__.py`
- `moeabench/stats/__init__.py`
- `moeabench/view/__init__.py`

Actions:

1. Expose `mb.clinic` as canonical.
2. Keep internals stable while changing public interface.
3. Remove public `fair_audit` and `q_audit`.
4. Publish final canonical names for stats/view.

Done criteria:

- `dir(mb)`, `dir(mb.stats)`, `dir(mb.view)`, `dir(mb.clinic)` match contract.

---

## Phase 3: Clinic (audit/report)

Primary file:

- `moeabench/diagnostics/auditor.py`

Actions:

1. Consolidate `audit(target, quality=True|False)`.
2. Keep `DiagnosticResult` as primary clinic return type.
3. Remove `summary()` completely.
4. Standardize `report(show=True, full=False, **kwargs)` across clinic/fair/quality results.

Done criteria:

- No `.summary()` calls in library Python code or `.py` examples.

---

## Phase 4: Unified Compare in Stats

Primary file:

- `moeabench/stats/tests.py` (or split into dedicated compare module)

Actions:

1. Implement `perf_compare(..., method='shift|match|win')`.
2. Implement `topo_compare(..., method='match|emd|anderson')`.
3. Unify `perf_compare` return into `PerfCompareResult`.
4. Decide final policy for short wrappers (`perf_shift`, `perf_match`, `perf_win`, `topo_match`): official or removed.

Done criteria:

- Same `report(show, full)` contract across compare methods.

---

## Phase 5: Topology/Attainment Naming

Actions:

1. Rename `topo_attainment` -> `attainment`.
2. Rename `topo_gap` -> `attainment_gap`.
3. Keep view semantics explicit: `bands` is primary for attainment, `topology` is valid generic visualization.

Done criteria:

- Names are aligned in code and docs without ambiguity.

---

## Phase 6: Final View API

Target files:

- `moeabench/view/*.py`
- `moeabench/view/__init__.py`

Actions:

1. Make `density` canonical for clinic/perf/topo (remove separate `distribution` endpoint from public API).
2. Keep low-level object type as the documented contract for each view.
3. Preserve internal polymorphism while not exposing it as primary contract.

Done criteria:

- Mapping table and code signatures match 1:1.

---

## Phase 7: Migration Test Gate

Test files:

1. `tests/unit/test_api_surface.py`
2. `tests/unit/test_report_contract.py`
3. `tests/unit/test_compare_result_contract.py`
4. `tests/unit/test_api_polymorphism.py`
5. `tests/unit/test_view_contracts.py`
6. `tests/unit/test_api_breakage_expected.py`

Focus:

1. Canonical API surface only.
2. Unified report contract.
3. Removed API remains removed.
4. Polymorphism works where intended.

Done criteria:

- Full migration suite passing and blocking accidental reintroduction of removed names.

---

## Phase 8: Documentation and Examples

Target files:

- `docs/reference.md`
- `docs/userguide.md`
- `misc/methods.md`
- Canonical `.py` examples

Actions:

1. Replace old naming/signatures.
2. Align examples to new API.
3. Add explicit "Breaking changes alpha -> beta" section.
4. Keep historical names only in migration notes when needed.

Done criteria:

- No active docs/examples pointing to removed API.

---

## Recommended Execution Order

1. Phase 1
2. Phase 2 + Phase 3
3. Phase 4 + Phase 5
4. Phase 6
5. Phase 7
6. Phase 8

---

## Beta Readiness Definition

1. Public API is reduced and stable per contract.
2. `report(full=False|True)` works across unified result objects.
3. `summary()` removed.
4. Surface/contract/breakage tests are green.
5. Documentation and examples are fully aligned.
