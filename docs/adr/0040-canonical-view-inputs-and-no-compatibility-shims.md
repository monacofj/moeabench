<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0040: Canonical View Inputs and No Compatibility Shims

**Status:** Accepted  
**Date:** 2026-03-13  
**Author:** Monaco F. J.  
**Supersedes (partially):** ADR 0010 (rich result objects), ADR 0030 (clinical instrument architecture), ADR 0039 (canonical API surface).  
**Drivers:** API Purity, Programmatic Parity, No Recalculation, Alpha Clean Break.

---

## 1. Context

The v0.14 API converged on canonical namespaces (`metrics`, `stats`, `clinic`, `view`), but one inconsistency remained in the visualization layer:

1. `view` methods often accepted canonical objects, but still exposed compatibility shims such as `stats=` or recomputed hidden intermediate data.
2. Some charts displayed information that was not fully accessible from the corresponding canonical result object.
3. This weakened the architectural contract:
   - compute in `metrics` / `stats` / `clinic`,
   - inspect and report programmatically,
   - render in `view` without recomputation.

For alpha, compatibility shims are not an asset. They preserve ambiguity and keep old mental models alive.

---

## 2. Decision

### 2.1 Canonical rule

If a chart visualizes information that belongs to a canonical analytical result, that information must live in the result object itself.

`view` methods must accept those result objects directly as their primary canonical input.

### 2.2 Programmatic parity rule

Everything shown visually by a canonical `view` must be accessible programmatically from the corresponding canonical object.

In other words:

- `result.report()` exposes the narrative
- `result.<property>` exposes the data
- `view(result)` renders the same information

No chart may rely on hidden view-only semantics as its primary contract.

### 2.3 No compatibility shims

When a canonical object contract exists, compatibility arguments that preserve an older path are removed instead of deprecated.

For alpha:

- no `stats=...`
- no `stats_result=...`
- no duplicate “helper path” carrying the same meaning as the canonical object itself

### 2.4 Polymorphism remains, but only as a convenience fallback

Argument polymorphism remains a library value, but it is not the primary architectural model.

Meaning:

- `view(raw_experiment)` may still resolve to the needed result internally
- but canonical examples, docs, and mental model must use `view(canonical_result)`

The canonical object is the source of truth.

---

## 3. First Enforcement Scope

This ADR is first enforced on performance contrast/distribution views:

- `mb.view.spread(perf_shift_result)`
- `mb.view.density(perf_match_result)`

The `PerfCompareResult` object is enriched with the plot payload required by the chart:

- samples
- labels
- metric label
- generation index

As a result:

- `mb.stats.perf_shift(...)` produces both inferential and visual payload
- `mb.stats.perf_match(...)` produces both inferential and visual payload
- `mb.view.spread(...)` and `mb.view.density(...)` no longer accept `stats=` compatibility arguments

---

## 4. Consequences

### Positive

- cleaner architectural separation
- no redundant user-facing pathways for the same semantics
- easier examples: `stats -> report -> view(stats)`
- stronger guarantee that visual output is reproducible programmatically

### Trade-offs

- some older calls stop working immediately
- migration must proceed method by method instead of pretending global completion

---

## 5. Migration Policy

This decision is enforced incrementally by domain, but each migrated endpoint must follow the final rule completely.

That means partial migration is allowed across the library, but not inside an individual endpoint that already has a canonical result contract.

Example:

- acceptable: `perf.spread` is clean while `clinic.density` is still pending
- unacceptable: `perf.spread(result)` exists, but `perf.spread(..., stats=result)` remains as compatibility

---

## 6. Verification

For each migrated `view` endpoint, the following must be true:

1. The canonical object contains all visual data shown by the chart.
2. The canonical example uses `result.report()` before `view(result)`.
3. No compatibility shim remains for the same semantic payload.
4. Raw-input polymorphism, if preserved, resolves into the same canonical object semantics.

