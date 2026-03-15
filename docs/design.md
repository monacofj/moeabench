<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench Design Constitution

This document defines the architectural constitution of MoeaBench: the principles, constraints, and implementation directives that all public APIs, internal abstractions, and future extensions are expected to follow.

It is intentionally **normative**. Its role is not to narrate the historical evolution of the library, nor to duplicate the exhaustive API contracts in **[reference.md](reference.md)**. Its role is to state what the architecture *must* be, what trade-offs are deliberately accepted, and what it means for a change to be **design-compliant**.

The detailed historical record of decisions lives in **[ADR/](adr/)**. This document summarizes only what is currently binding.

---

## **Summary**
1.  **[1. Scope and Status](#1-scope-and-status)**
2.  **[2. The Problem MoeaBench Solves](#2-the-problem-moeabench-solves)**
3.  **[3. Architectural Commitments](#3-architectural-commitments)**
4.  **[4. Public API Constitution](#4-public-api-constitution)**
5.  **[5. Data and Object Model](#5-data-and-object-model)**
6.  **[6. Polymorphism Policy](#6-polymorphism-policy)**
7.  **[7. Rich Results and Reporting](#7-rich-results-and-reporting)**
8.  **[8. Views and Visualization Boundaries](#8-views-and-visualization-boundaries)**
9.  **[9. Diagnostics, Calibration, and Reproducibility](#9-diagnostics-calibration-and-reproducibility)**
10. **[10. Performance and Numerical Integrity](#10-performance-and-numerical-integrity)**
11. **[11. Extensibility and Plugin Discipline](#11-extensibility-and-plugin-discipline)**
12. **[12. Breaking Changes and API Hygiene](#12-breaking-changes-and-api-hygiene)**
13. **[13. Compliance Checklist](#13-compliance-checklist)**

---

## **1. Scope and Status**

`design.md` is the highest-level normative document of the repository.

Its purpose is to define:

- the problems MoeaBench is designed to solve
- the abstractions the library privileges
- the trade-offs it deliberately accepts
- the internal architectural model
- the implementation directives contributors are expected to follow
- the criteria used to judge whether a change is architecturally compliant

The division of documentation is therefore:

- **`design.md`**: constitutional principles and binding design directives
- **`adr/*.md`**: decision records and rationale history
- **`reference.md`**: exhaustive technical contracts
- **`userguide.md`**: didactic tutorial for users
- **`api_sheet.md`**: thematic map of the canonical API

If a contributor is unsure whether a change is architecturally appropriate, this document takes precedence over convenience, familiarity, or compatibility habits inherited from older versions.

---

## **2. The Problem MoeaBench Solves**

MoeaBench is not primarily a solver library. It is an **analytical framework for scientific auditing of multi-objective evolutionary optimization workflows**.

This means the library exists to solve the following problems:

1.  Raw optimization outputs are difficult to interpret scientifically.
2.  Stochastic algorithms require distribution-aware analysis, not cherry-picked single runs.
3.  High-dimensional objective spaces make naive metric interpretation physically misleading.
4.  Researchers need a reproducible path from experiment configuration to result interpretation.
5.  Visualization must support diagnosis, not merely presentation.
6.  Advanced diagnostics require calibration, baselines, and scientific provenance as first-class concerns.

From this follows a central architectural claim:

> MoeaBench is an instrument of interpretation, auditing, and explanation built around optimization experiments, not merely a thin utility wrapper around solver execution.

---

## **3. Architectural Commitments**

The architecture of MoeaBench is governed by the following commitments.

### **3.1. Analysis First**

The architecture is analysis-first, not solver-first.

- Solvers and benchmark problems are important, but they exist primarily as sources of analyzable experimental data.
- The center of gravity of the library is the transformation of trajectories into interpretable scientific evidence.
- Any feature that makes execution easier but weakens analytical coherence is suspect.

### **3.2. Semantic Separation**

Different namespaces exist because they represent different semantic roles.

- `metrics` computes scalar or trajectory-valued measurements.
- `stats` compares, structures, or synthesizes analytical evidence.
- `clinic` provides diagnostic synthesis and advanced pathology-oriented methods.
- `view` renders visual perspectives only.
- `system` provides operational support.

These boundaries are intentional and must not be blurred casually.

### **3.3. Canonical Workflows Over Ad Hoc Shortcuts**

The library privileges a canonical analytical workflow:

```python
import moeabench as mb

res = mb.stats.strata(exp)
res.report()
mb.view.strata(res)
```

This pattern is not accidental ergonomics; it is part of the design.

### **3.4. Reproducibility as Architecture**

Reproducibility is not a post-processing concern. It is an architectural requirement.

- seed management
- persistence metadata
- environment information
- calibration provenance
- baseline provenance

All of these are part of the design surface of the library.

### **3.5. Scientific Honesty Over Convenience**

The library prefers correct and explainable scientific semantics over superficially convenient but misleading abstractions.

Examples:

- prefer calibrated comparisons over raw thresholds
- prefer explicit normalization semantics over opaque defaults
- prefer rich result objects over bare floats when interpretation matters
- prefer canonical naming clarity over retaining bad historical names

---

## **4. Public API Constitution**

The canonical public usage style is:

```python
import moeabench as mb
```

The public surface is organized under:

- `mb.experiment`
- `mb.Run`
- `mb.defaults`
- `mb.metrics`
- `mb.stats`
- `mb.view`
- `mb.clinic`
- `mb.system`
- `mb.mops`
- `mb.moeas`

The following rules are binding.

### **4.1. Namespace Roles Are Semantic**

Public namespaces must reflect architectural roles, not historical implementation leftovers.

### **4.2. One Canonical Name Per Public Concept**

When a public concept has a canonical name, redundant synonyms should be removed unless they provide genuine semantic value.

Accepted examples:

- `mb.metrics.hv` as a domain-standard short alias for `hypervolume`
- semantic compare aliases such as `perf_shift`, `topo_match`, `topo_shift`

Rejected examples:

- redundant constructor aliases with no semantic gain
- historical compatibility shims that perpetuate bad ontology

### **4.3. Surface Hygiene Matters**

Internal modules, compatibility leftovers, and helper symbols must not leak casually into public namespaces.

If a name is visible publicly, that visibility must be intentional.

---

## **5. Data and Object Model**

The core data model is:

```text
Experiment -> Run -> Population -> SmartArray / ndarray-like space
```

This hierarchy is a first-class architectural commitment.

### **5.1. Experiment**

`Experiment` is the global manager of a scientific protocol.

- it owns the MOP/MOEA pairing
- it owns multi-run execution
- it is the cloud-centric aggregation point
- it is the default unit of comparison in most scientific usage

### **5.2. Run**

`Run` is one stochastic trajectory.

- it is part of the public object model
- it is not the primary construction entry point
- it exists because the architecture explicitly models trajectory-level analysis

### **5.3. Population**

`Population` is the canonical snapshot abstraction.

- it represents a generation-level state of the search
- it is the correct abstraction for geometry, ranking, and snapshot-local diagnostics

### **5.4. SmartArray**

Raw numerical data is not architecturally neutral.

Whenever feasible, numerical arrays should carry lineage and semantic metadata sufficient to support:

- reporting
- labeling
- plotting
- traceability

This is why MoeaBench uses annotated numerical containers rather than treating every matrix as anonymous.

---

## **6. Polymorphism Policy**

Polymorphism is a deliberate part of the API, but it is **disciplined polymorphism**, not arbitrary overloading.

### **6.1. Canonical Inputs**

The canonical scientific inputs are:

- `Experiment`
- `Run`
- `Population`
- compatible raw arrays in selected lower-level contexts
- rich result objects for correlated views

### **6.2. Purpose of Polymorphism**

Polymorphism exists to reduce extractive boilerplate while preserving semantic correctness.

Users should not have to write:

```python
exp[0].pop(-1).objs
```

when the library can safely infer the intended analytical object.

### **6.3. Polymorphism Has Boundaries**

Polymorphism is compliant only when:

- the semantic interpretation is stable
- the inferred object is scientifically meaningful
- the result is consistent with the canonical workflow

Polymorphism is not compliant when it introduces ambiguity about:

- what is being measured
- which domain is active
- what the result object means
- what the canonical visual correlate should be

### **6.4. Views May Be Input-Polymorphic, Not Semantically Magical**

Views may accept high-level inputs and resolve the needed internal data, but they must not invent new analytical semantics that belong in `metrics`, `stats`, or `clinic`.

---

## **7. Rich Results and Reporting**

MoeaBench is committed to a **rich result architecture**.

This is one of the strongest constraints in the system.

### **7.1. Everything Important Should Report**

Important analytical objects should support `.report()`.

This is not cosmetic. It is the architectural expression of the library's commitment to:

- narrative diagnostics
- scientific interpretation
- metadata-bearing outputs
- reproducible inspection

### **7.2. Bare Scalars Are Insufficient for Core Analytical Workflows**

Whenever an analytical result has structure, semantics, or a visual correlate, returning a bare primitive is usually non-compliant.

The preferred design is:

- compute a rich result object
- expose programmatic fields
- support `.report()`
- allow correlated views to consume the result directly

### **7.3. Typical Rich Results**

Examples of the intended result architecture include:

- `MetricMatrix`
- `PerfCompareResult`
- `AttainmentSurface`
- `AttainmentDiff`
- `RankCompareResult`
- `StrataCompareResult`
- `TierResult`
- `DiagnosticResult`
- advanced public result types such as `FairResult`

---

## **8. Views and Visualization Boundaries**

The `view` namespace is organized by **chart type**, not by hidden analytical semantics.

This is a binding design principle.

### **8.1. Views Show**

Views exist to render.

They may:

- dispatch by domain when the contract is explicit and stable
- accept rich result objects directly
- infer labels, metadata, and titles from structured inputs

They must not:

- replace proper analytical endpoints
- encode domain logic that belongs in `stats` or `clinic`
- become compatibility dumping grounds for legacy naming

### **8.2. Chart-Type Naming Is Canonical**

The canonical view grammar is built around forms such as:

- `history`
- `spread`
- `density`
- `topology`
- `bands`
- `gap`
- `ranks`
- `strata`
- `tiers`
- `radar`
- `ecdf`

This grammar is part of the public identity of the library and should remain coherent.

### **8.3. Correlated View Principle**

Whenever a canonical result object exists, the corresponding view should prefer consuming that object directly.

This preserves:

- determinism of interpretation
- architectural separation
- reusability of analytical state

---

## **9. Diagnostics, Calibration, and Reproducibility**

Diagnostics are part of the scientific architecture, not an optional add-on.

### **9.1. Canonical Diagnostic Entry Point**

The canonical public diagnostic entry point is:

```python
mb.clinic.audit(target, quality=True)
```

This is the central diagnostic narrative for first-time users and the preferred entry path for general workflows.

### **9.2. Advanced Public Diagnostic Methods**

FAIR metrics, Q-score functions, point-wise helpers, and baseline-management methods may remain public, but they are architecturally secondary to `audit`.

They exist for:

- advanced analysis
- custom instruments
- pathology research
- fine-grained semantic overlays

### **9.3. Calibration Is First-Class Infrastructure**

Calibration is not an implementation detail.

The architecture assumes:

- a problem may require GT acquisition
- diagnostics depend on calibrated baselines
- baseline provenance matters scientifically
- contributors must treat calibration artifacts as part of the analytical infrastructure

Both:

- `mop.calibrate(...)`
- `mb.clinic.calibrate(...)`

are valid expressions of this design principle.

### **9.4. Reproducibility Metadata Is Mandatory Infrastructure**

The architecture requires explicit support for:

- deterministic seed chains
- environment metadata
- persistence metadata
- baseline provenance
- audit provenance

If a feature weakens reproducibility, it must justify that cost explicitly.

### **9.5. Scientific Baselines Are Normative Artifacts**

Calibration references, baseline datasets, and other frozen scientific artifacts are part of the architectural contract of the library.

They are not optional examples, disposable fixtures, or informal regression hints.

Architecturally, this means:

- canonical baselines define expected scientific behavior for calibrated workflows
- contributors must treat baseline updates as explicit scientific changes, not routine maintenance noise
- any change that alters certified outputs must explain why the baseline itself should move

### **9.6. Deterministic Reproduction Is the Default Testing Expectation**

For calibrated workflows, MoeaBench assumes deterministic seed discipline and reproducible numerical replay as the default expectation.

This means:

- the same certified configuration should reproduce the same analytical outcome across code iterations
- tests may treat small numerical drift as a meaningful regression rather than harmless noise
- stochastic methodology does not excuse architectural sloppiness in seeded validation pipelines

---

## **10. Performance and Numerical Integrity**

MoeaBench is committed to analytical rigor and computational seriousness.

### **10.1. Vectorization Is the Default**

Core analytical paths must prefer NumPy-vectorized implementations over naive Python loops.

This is both a performance directive and a design-style directive.

### **10.2. Numerical Integrity Over Superficial Simplicity**

The library prefers implementations that are:

- more explicit about precision
- more explicit about normalization
- more explicit about GT assumptions
- more explicit about stable semantics

even when that makes the implementation less superficially simple.

### **10.3. High-Precision Reproducibility Is a Design Requirement**

MoeaBench is designed for scientific contexts where differences at fine decimal scales are analytically meaningful.

Therefore:

- internal analytical pipelines must preserve precision deliberately
- certified regression targets may require exact or high-precision agreement rather than broad tolerance bands
- a contributor must not assume that small floating-point variation is acceptable unless that tolerance is explicitly justified by the scientific contract of that pathway

For certified regression pathways, MoeaBench adopts a normative default expectation of agreement down to **6 decimal places** (`abs=1e-6`) unless a stricter contract is explicitly defined for that pathway.

This limit is not arbitrary. It is grounded in the project's scientific reproducibility stack:

- `float64` is the mandatory internal numerical policy
- deterministic or frozen PRNG behavior is part of the reproducibility architecture
- analytical verification paths may impose stricter tolerances such as `atol=1e-12` and `rtol=1e-8` when validating mathematical invariants or certified reference geometry

Contributors must therefore interpret `6 decimal places` not as a display preference, but as the default certification floor for regression-grade reproducibility in calibrated analytical workflows.

### **10.4. High-Dimensional Honesty**

Any metric or diagnostic feature that behaves misleadingly in high-dimensional regimes must be treated with scientific caution.

Architecturally, this means:

- no naive threshold rhetoric
- no false equivalence between raw values across incompatible regimes
- strong preference for calibrated and normalized interpretation

---

## **11. Extensibility and Plugin Discipline**

MoeaBench follows a host-guest plugin architecture.

### **11.1. The Framework Is the Host**

The library provides:

- execution infrastructure
- data model
- metrics
- statistics
- diagnostics
- visualization
- persistence
- calibration hooks

### **11.2. Problems and Solvers Are Guests**

User-defined MOPs and MOEAs must integrate with the host architecture rather than bypassing it.

This means custom extensions should gain:

- the same reporting contracts
- the same data hierarchy
- the same calibration pathways where applicable
- the same analytical workflows

### **11.3. Plugin Compliance Matters**

A plugin is architecturally good not merely when it runs, but when it participates correctly in:

- metadata flow
- data extraction
- reporting
- analysis compatibility
- reproducibility

---

## **12. Breaking Changes and API Hygiene**

During `v0.x`, the architecture prefers coherence over compatibility baggage.

### **12.1. Breaking Changes Are Acceptable**

Breaking the API is acceptable when it:

- removes bad naming
- eliminates ambiguous ontology
- clarifies canonical flows
- strengthens namespace discipline
- reduces compatibility shims

### **12.2. Compatibility Shims Are Not Neutral**

Compatibility shims increase:

- cognitive load
- documentation burden
- namespace pollution
- ambiguity about what is truly canonical

They should therefore be treated as costs, not as automatically virtuous.

### **12.3. Canonical Naming Has Priority**

If a historical name is semantically wrong, metaphorically bad, or structurally confusing, it should be replaced rather than preserved out of habit.

---

## **13. Compliance Checklist**

A contribution is more likely to be design-compliant when the answer to the following questions is "yes."

### **Public API**

- Does it use the canonical namespace for its role?
- Does it avoid introducing redundant public synonyms?
- Does it preserve `import moeabench as mb` as the canonical usage style?

### **Semantics**

- Is the semantic boundary between `metrics`, `stats`, `clinic`, and `view` respected?
- Is any new polymorphism disciplined and unambiguous?
- Does the naming match the ontology of the domain?

### **Results**

- Does the feature return a rich result object when interpretation matters?
- Does that result support `.report()` when appropriate?
- Is there a coherent correlated view if the result is visual?

### **Views**

- Is the new visualization truly a view, rather than hidden analysis logic?
- Is the view named by chart type, not by accidental implementation history?

### **Diagnostics and Reproducibility**

- Does the change preserve calibration and baseline integrity?
- Does it preserve or improve reproducibility metadata?
- Does it keep diagnostic synthesis centered on the canonical architecture?
- Does it preserve deterministic or high-precision reproducibility where the scientific workflow expects it?
- If numerical outputs changed, is there an explicit justification for changing certified baselines or regression targets?

### **Implementation Quality**

- Is the implementation vectorized where it should be?
- Does it avoid leaking internal helpers into the public surface?
- Does it align with the currently accepted ADR state?

If a change fails these questions, the burden of justification lies with the change, not with the constitution.

---

## **For Contributors**

New contributors should read this document before implementing non-trivial changes.

Recommended reading order:

1.  **`design.md`** — constitutional principles and compliance rules
2.  **[api_sheet.md](api_sheet.md)** — thematic map of the public API
3.  **[reference.md](reference.md)** — exact technical contracts
4.  Relevant **[ADRs](adr/README.md)** — decision rationale for the area being touched
