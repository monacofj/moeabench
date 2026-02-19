# ADR 0032: Enhanced Persistence Metadata and Diagnostic Simplification

## Context
As MoeaBench matures into a scientific research tool, the ability to reproduce and audit results over time becomes critical. Previous persistence versions (Schema v1) relied on binary serialization (joblib) without explicit library or environment metadata. Furthermore, the "Diagnostic Profile" (Industry, Research) was based on arbitrary absolute thresholds that were superseded by the relative, baseline-anchored Q-Score system introduced in v0.9.

## Decision
1.  **Enhanced Persistence (Schema v2)**: The persistence layer now generates a structured `metadata.json` and a human-readable `README.md` inside every experiment ZIP.
2.  **Explicit Metadata**: We now track library versioning, Python version, platform, and high-resolution hashes of the `baselines_v4.json` data package used during the run.
3.  **Scientific Headers**: Users can now assign `authors`, `license` (SPDX standardized), and `year` to an experiment. These are embedded as SPDX headers in the ZIP's README.
4.  **Diagnostic Simplification**: Formally removed the `DiagnosticProfile` Enum and the `profile` parameter from `mb.diagnostics.audit()`. The system now operates on a universal Q-Score regime ($Q \in [0, 1]$) with fixed semantic thresholds.

## Consequences
- **Positive**: Long-term auditability. A result collected today can be verified against its original library environment and baseline version.
- **Positive**: Simplified API. Users no longer need to choose a "profile" to get a diagnostic verdict.
- **Positive**: Scientific compliance. Saved data is now self-documenting with proper authorship and licensing identifiers.
- **Neutral**: Binary files saved with Schema v2 are slightly larger due to the `metadata.json` and `README.md` files.
- **Warning**: Code that explicitly passed a `DiagnosticProfile` to `audit()` will now trigger an error (as the parameter was removed) and should be updated to remove the argument.
