# ADR 0033: Flexible Reference Management for Longitudinal Auditing

## Context
Scientific research often requires comparing new results against historical benchmarks or alternative references. MoeaBench's diagnostic suite previously relied on a hardcoded internal baseline file (`baselines_v4.json`). While `register_baselines()` allowed adding new problems, there was no safe way to temporarily *swap* the entire reference system for legacy comparison without affecting the global state of the library.

## Decision
1.  **Contextual Baseline Switching**: Introduced `mb.diagnostics.baselines.use_baselines(source)`. This context manager allows a user to run a block of code (e.g., an audit) against a specific historical baseline file, automatically restoring the default system state upon exit.
2.  **Polymorphic Ground Truth Loading**: The `audit()` function and all individual metric functions (`q_headway`, `coverage`, etc.) now accept file paths in the `ground_truth` argument.
3.  **Automatic Format Detection**: The system now supports loading references from `.npy`, `.npz` (searching for 'F' or the first array), and `.csv` (standard header/delimiter) files automatically.

## Consequences
- **Positive**: Enables "Time-Travel" Auditing. Researchers can verify if an algorithm's Q-Score improved relative to last year's baselines.
- **Positive**: Workflow efficiency. No need to manually load NumPy arrays before calling `audit()`; simply passing the path is now sufficient.
- **Positive**: Registry Safety. `reset_baselines()` provides a way to ensure a clean state during batch processing or unit testing.
- **Warning**: When using `use_baselines()`, all metrics computed inside the block will refer to the provided source. If that source is incomplete (missing the specific problem), an `UndefinedBaselineError` will be raised as per the Fail-Closed policy.
