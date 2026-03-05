# ADR 0037: Universal Reporting Interface (URI)

## Status
Proposed (v0.13.1)

## Context
A core design pillar of MoeaBench is **Technical Storytelling** (as defined in `docs/design.md`). Previous versions limited narrative reporting (`.report()`) to analytical result objects like `Experiment`, `MetricMatrix`, and `StatsResult`. 

However, researchers frequently interact with intermediate structural objects:
- `Run`: A single stochastic trajectory.
- `Population`: A point-in-time snapshot of the manifold.
- `BaseMop`: The problem definition and its calibration status.
- `BaseMoea`: The optimization engine and its hyperparameter state.

In these objects, raw data (NumPy matrices) often lacks context. A `Run` object doesn't narrate its seed or duration, and a `BaseMop` doesn't immediately signal whether its clinical baselines are active.

## Decision
We expand the **Universal Reporting Contract** to include all core structural objects in the library. 

1.  **Base Mixin**: All reportable objects must inherit from the `Reportable` mixin (defined in `moeabench/core/base.py`).
2.  **Implementation**: Each class must implement a `.report()` method that returns a human-readable string.
3.  **Environment Awareness**: Reports must support both `plain text` (for CLI/Logs) and `Markdown` (for Jupyter/IPython) through the `markdown` keyword argument.
4.  **Global Entry Point**: The `mb` object itself will implement `.report()` to provide a high-level summary of the research environment.

## Classes Impacted
- `_MB` (Global Wrapper)
- `Run` (Core)
- `Population` (Core)
- `JoinedPopulation` (Core)
- `SmartArray` (Core Data)
- `BaseMop` (MOPs Registry)
- `BaseMoea` (Algorithms Registry)

## Consequences
- **Improved Interpretability**: Researchers can query any object for its "Story" without knowing internal attribute names.
- **Auditing Clarity**: Objects explain their own provenance (e.g., a `SmartArray` reports its source and generation).
- **Consistency**: The user experience is unified: if it's an object in `moeabench`, it has a `.report()`.
- **Minimal Overhead**: The mixin approach ensures that these methods remain decoupled from calculation logic.
