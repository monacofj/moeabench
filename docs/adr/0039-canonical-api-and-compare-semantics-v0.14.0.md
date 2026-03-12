# ADR 0039: Canonical API and Compare Semantics (v0.14.0)

**Status:** Accepted  
**Date:** 2026-03-11  
**Author:** Monaco F. J.  
**Supersedes (partially):** ADR 0014 (semantic taxonomy), ADR 0030 (clinical instrument naming surface), ADR 0032 (diagnostic entrypoint simplification details).  
**Drivers:** API Cohesion, Scientific Clarity, Dispatch Polymorphism, Alpha Clean Break.

---

## 1. Context

From v0.13.x to v0.14.0, the library converged on a cleaner API surface, but three structural inconsistencies remained:

1. **Compare method ambiguity**  
   In `perf_compare` and `topo_compare`, the `method` parameter mixed two ontologies:
   - semantic intent (`shift`, `match`, `win`),
   - statistical operator (`ks`, `mannwhitney`, `a12`, `anderson`, `emd`).
   This made contracts ambiguous and documentation hard to stabilize.

2. **Naming split between analysis and visualization**  
   Some APIs were already canonicalized (`view.topology`, `view.radar`), while docs/examples still leaked legacy names or domain-prefixed surfaces (`clinic_*`, `topo_*` on view side).

3. **Diagnostics public contract drift**  
   The intended high-level contract was a single public diagnostic entrypoint, but residual references to intermediate calls (`fair_audit`, `q_audit`) and duplicated short reports (`summary`) persisted in materials and mental model.

At the same time, moeabench must preserve its **argument polymorphism** principle: users may pass low-level objects or high-level containers, and API endpoints should extract/compute required internals automatically.

---

## 2. Decision

### 2.1 Canonical namespace grammar

The public API is organized by intent:

- `mb.metrics`: scalar/trajectory metric producers.
- `mb.stats`: statistical comparators and structural analyzers.
- `mb.clinic`: diagnostic synthesis layer (physical + quality interpretation).
- `mb.view`: chart-type-oriented visualization endpoints.

This is a hard alpha break; legacy public names are not part of canonical beta surface.

### 2.2 Diagnostics contract: one public entrypoint

Diagnostics are publicly orchestrated through:

```python
mb.clinic.audit(target, quality=True)
```

- `target` is the real low-level analytical subject (population-compatible object).
- `quality=True` controls whether quality scores are included in the output payload.
- Internal sub-computations are implementation details, not user-facing public API.

### 2.3 Reporting contract unification

Result objects standardize on:

```python
result.report(show=True, full=False)
```

- `summary()` is removed from canonical API.
- `report(...)` is the single narrative endpoint for terminal/notebook rendering.

### 2.4 Compare architecture: technical methods + semantic aliases

#### Performance comparison

`perf_compare(method=...)` accepts only technical statistical operators:

- `method='mannwhitney'`
- `method='ks'`
- `method='a12'`

Semantic aliases remain as readability shortcuts:

- `perf_shift(...)` -> `perf_compare(method='mannwhitney')`
- `perf_match(...)` -> `perf_compare(method='ks')`
- `perf_win(...)` -> `perf_compare(method='a12')`

#### Topological comparison

`topo_compare(method=...)` accepts only technical operators:

- `method='ks'`
- `method='anderson'`
- `method='emd'`

Semantic aliases are separated by meaning:

- `topo_match(...)` -> `topo_compare(method='ks')`
- `topo_tail(...)` -> `topo_compare(method='anderson')`
- `topo_shift(...)` -> `topo_compare(method='emd')`

This resolves the prior conflation where `ks` and `anderson` were both interpreted as the same semantic label.

### 2.5 Report semantics for topology compare

`DistMatchResult.report()` now exposes both layers explicitly:

- semantic alias (`topo_match` / `topo_tail` / `topo_shift`)
- technical method (`KS` / `ANDERSON` / `EMD`)

Example header pattern:

```
Distribution Match Report (topo_tail | ANDERSON)
```

This keeps reports scientifically explicit while preserving user-level intent.

### 2.6 View API canonicalization

`mb.view` is chart-oriented and domain-dispatched:

- `topology`, `bands`, `gap`, `density`, `history`, `spread`,
- `ranks`, `caste`, `tiers`, `ecdf`, `radar`.

Key rule: view endpoints visualize; if input is high-level polymorphic (`Experiment`, `DiagnosticResult`, etc.), they may resolve required internal data but do not define independent analysis semantics.

### 2.7 Visualization title defaults

All canonical `mb.view.*` methods follow:

- `title=None` => semantic, auto-generated title
- `title="..."` => explicit override

This removes unnecessary boilerplate in examples while preserving publication-level customization.

### 2.8 Operational defaults promoted to contract

- `mb.stats.topo_shift(...)` delegates to `topo_compare(method='emd')` and uses:
  - `threshold=mb.defaults.displacement_threshold` if omitted (`0.1` in v0.14.0).
- Topological density dispatch (`mb.view.density(..., domain='topo')` or inferred) uses:
  - `space='objs'` by default
  - `axes=None` by default (auto axis selection).
- `mb.view.topology(...)` GT contract is explicit:
  - `show_gt=True` + `gt=<array>`: uses provided GT.
  - `show_gt=True` + no `gt`: infers GT from experiment-like inputs (warns on no-inference or mixed GT sources).
  - `show_gt=False`: never displays GT and ignores `gt`.

---

## 3. Rationale and Intuition

### 3.1 Why technical names in `method`?

`method` is an operator selector, not a narrative label.  
When `method='ks'`, there is no ambiguity about what is computed, p-value meaning, or numerical assumptions.

Semantic names belong in aliases because aliases encode user intent, not implementation details.

### 3.2 Why keep semantic aliases at all?

Aliases preserve expressive readability in notebooks and examples:

- `perf_shift(...)` communicates "location evidence"
- `topo_tail(...)` communicates "tail-sensitive equivalence check"

This dual layer (technical core + semantic alias) provides rigor without sacrificing ergonomics.

### 3.3 Why separate `topo_match` and `topo_tail`?

`KS` and `Anderson-Darling` are not equivalent observational lenses:

- `KS`: max-CDF discrepancy (global shape discrepancy anchor).
- `Anderson`: tail-sensitive discrepancy.

Treating both as "match" hides a meaningful semantic distinction and weakens interpretability.

### 3.4 Why remove `summary()`?

Two report entrypoints (`summary` and `report`) produce drift over time and increase maintenance entropy.
One narrative contract simplifies API learning, docs, and test contracts.

---

## 4. Consequences

### Positive

- **Deterministic contracts**: `method` is now scientifically unambiguous.
- **Cleaner docs/examples**: one mapping between intent, call, and interpretation.
- **Better testability**: alias tests validate deterministic delegation.
- **Reduced cognitive overhead**: diagnostics and reporting have one public path each.

### Trade-offs

- **Breaking change**: code using legacy method names (`shift`, `match`, `win`) in `*_compare` must migrate.
- **Doc migration burden**: all references must stay synchronized to avoid user confusion.

---

## 5. Verification and Guardrails

This ADR is enforced by:

1. Unit tests for compare contract (`method` accepted values and return shape).
2. Unit tests for alias delegation (`perf_*`, `topo_*` -> technical methods).
3. Regression tier checks preserving calibration numeric consistency across version baselines.
4. Documentation alignment checks in release workflow.

---

## 6. Migration Notes (v0.14.0)

- `perf_compare(method='shift'|'match'|'win')` -> `method='mannwhitney'|'ks'|'a12'`
- `topo_compare(method='match')` -> `method='ks'`
- `topo_emd` -> `topo_shift`
- `topo_anderson` -> `topo_tail`
- `result.summary()` -> `result.report(show=True, full=False)`
- use `mb.clinic.audit(target, quality=True)` as canonical diagnostic entrypoint.
