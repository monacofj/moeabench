<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0041: Layer Ontology for Structural Analysis

**Status:** Accepted  
**Date:** 2026-03-13  
**Author:** Monaco F. J.  
**Supersedes (partially):** ADR 0008, ADR 0023.  
**Drivers:** Ontological Clarity, API Precision, Alpha Clean Break.

---

## 1. Context

The structural-analysis domain previously mixed two different roles under the same vocabulary:

- `strata` was used as the broad structural concept
- one specific visualization/result family also wanted a strong semantic name
- `caste` filled that slot, but it introduced avoidable metaphorical baggage

This produced an uneven public surface:

- `ranks`
- `caste`
- `tiers`

The three endpoints were structurally coherent, but not linguistically coherent.

---

## 2. Decision

We adopt **layer** as the base structural ontology.

### 2.1 Internal concept

The primitive structural computation is a **layer decomposition** of the population by non-domination depth.

This internal concept is represented by private layer machinery, not by a public user-facing endpoint.

### 2.2 Public analytical views

The canonical public views over that layer structure are:

- `mb.stats.ranks(...)` / `mb.view.ranks(...)`
- `mb.stats.strata(...)` / `mb.view.strata(...)`
- `mb.stats.tiers(...)` / `mb.view.tiers(...)`

### 2.3 Semantic roles

- **`ranks`**: frequency/depth structure
- **`strata`**: quality distribution within each dominance layer
- **`tiers`**: joint comparative occupancy across a merged layer system

### 2.4 No compatibility shims

Because the project is still in `v0.x.y`, this renaming is a clean break:

- `mb.stats.caste` is removed
- `mb.view.caste` is removed
- no compatibility aliases are preserved

---

## 3. Rationale

### 3.1 Why `layer`?

`layer` is direct, physical, and neutral. It describes the decomposition itself without forcing one specific analytical lens.

### 3.2 Why reuse `strata` as a public view?

Once `layer` becomes the underlying ontology, `strata` is free to become the name of the rank-wise distributional lens. This is a better semantic fit than `caste`, and avoids metaphorical baggage.

### 3.3 Why not keep both names?

Keeping both would preserve ambiguity and documentation overhead during alpha. A clean break produces a sharper mental model and a cleaner codebase.

---

## 4. Consequences

### Positive

- cleaner ontology for structural analysis
- better alignment between concept, result object, and visualization
- removal of a metaphorically loaded public name

### Trade-offs

- breaking rename for notebooks, examples, and downstream callers
- historical ADRs and docs must be updated to explain the vocabulary shift

---

## 5. Verification

This decision is considered correctly applied when:

1. public APIs expose `ranks`, `strata`, and `tiers`
2. no canonical docs present `caste` as a supported public endpoint
3. internal structural code uses layer terminology for the underlying primitive
