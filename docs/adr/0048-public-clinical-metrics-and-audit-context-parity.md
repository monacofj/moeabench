<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0048: Public Clinical Metrics and Audit Context Parity

## Status

Accepted

## Date

2026-08-27

## Context

ADR 0039 simplified diagnostics around `mb.clinic.audit(...)` and described the
intermediate clinical computations as implementation details. The public API has
since evolved: users may legitimately request one physical FAIR metric or one
Q-score directly, while `audit()` remains the high-level orchestration and
synthesis entrypoint.

That broader public surface creates a scientific consistency requirement. A
metric must not change merely because it was requested individually instead of
as one component of an audit. GT selection, resolution, population size,
baseline context, and other canonical references must therefore resolve in the
same way under equivalent/default calls.

## Decision

The public clinical API consists of both the aggregate audit and the individual
metric endpoints.

Physical metrics are public:

```python
mb.clinic.headway(...)
mb.clinic.closeness(...)
mb.clinic.coverage(...)
mb.clinic.gap(...)
mb.clinic.regularity(...)
mb.clinic.balance(...)
```

Their corresponding `q_*` functions are also public, alongside:

```python
mb.clinic.audit(...)
```

The following parity contract applies:

- Individual clinical calls and the corresponding components calculated by
  `audit()` use the same canonical diagnostic context when given equivalent
  inputs and no conflicting explicit overrides.
- Context resolution includes, as applicable, the evaluated final population or
  front, initial population for longitudinal metrics, Ground Truth, problem
  identity, effective population size `K`, resolution scale `s_k`, baseline
  source, uniform reference set, centroids, and occupancy reference.
- `audit()` may resolve and expose the exact context through
  `DiagnosticResult.diagnostic_context`; this context is part of the audit's
  reproducibility payload rather than hidden view-only state.
- Explicit arguments such as a user-supplied GT, `s_k`, baseline source, or
  other supported overrides may intentionally define a different context. The
  parity guarantee applies to equivalent contexts, not to calls with different
  explicit inputs.
- `audit(quality=True)` computes FAIR metrics, Q-scores, and synthesized quality
  interpretation. `audit(quality=False)` stops after the FAIR layer and does not
  fabricate or infer Q-scores.
- `audit()` remains the preferred entrypoint when a complete diagnosis is
  wanted; it is not the only public way to obtain clinical measurements.

Argument polymorphism remains part of this contract. In particular, an
`Experiment` resolves its aggregate final front through the experiment's
`front()` semantics, while a `Run` resolves that run's front and a `Population`
uses the supplied snapshot.

## Consequences

- Direct metric calls are first-class, documented, and testable public API.
- Results from direct calls and from `audit()` can be compared without a hidden
  change of calibration context.
- The audit becomes an orchestrator and synthesizer rather than an exclusive
  gateway to the clinical layer.
- Public clinical views can consume canonical result objects without defining
  independent diagnostic semantics, consistent with ADR 0040.
- ADR 0039 remains authoritative for namespace grammar, comparison semantics,
  and removal of `summary()`, but its "one public diagnostic entrypoint"
  restriction is superseded by this ADR.

## Relationship to Earlier Decisions

- [ADR 0026](0026-clinical-metrology.md) established the metrological clinical
  model.
- [ADR 0028](0028-refined-clinical-diagnostics-v0.9.1.md) refined the physical
  and quality layers.
- [ADR 0039](0039-canonical-api-and-compare-semantics-v0.14.0.md) established
  the canonical namespace and former audit-only public surface.
- [ADR 0040](0040-canonical-view-inputs-and-no-compatibility-shims.md) requires
  programmatic parity between canonical analytical results and views.
