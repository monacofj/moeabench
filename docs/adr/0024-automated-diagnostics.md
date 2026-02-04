# 24. Automated Diagnostics (Algorithmic Pathology)

Date: 2026-02-04

## Status

Accepted

## Context

Users often struggle to interpret conflicting performance metrics (e.g., Low GD but High IGD).  
A raw number does not convey the *meaning* of the behavior (e.g., "The algorithm has collapsed to a single point").  
There is a need for an expert system layer that translates numerical signatures into scientific diagnoses.

## Decision

We will implement a `MoeaBench.diagnostics` module (Algorithmic Pathology) that:
1.  **Audits** metrics (IGD, GD, EMD, H_rel).
2.  **Classifies** behavior into standardized pathologies (`DiagnosticStatus`):
    - `DIVERSITY_COLLAPSE`: Good Convergence / Poor Coverage.
    - `TOPOLOGICAL_DISTORTION`: High EMD.
    - `SUPER_SATURATION`: H_rel > 1.0.
3.  **Explains** the finding via a generated `rationale()` string.

## Consequences

### Positive
- **Didactic Value**: Teaches users how to interpret metrics.
- **Automation**: Allows programmatic pipeline gating (e.g., "Fail CI if Diversity Collapse").

### Negative
- **Complexity**: Adds a logic layer that must be maintained as metrics evolve.
- **Overhead**: Requires metric computation (IGD, EMD) which can be expensive if enabled by default (hence `diagnose=False` default).
