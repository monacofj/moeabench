# ADR 0011: Selective Persistence (Save/Load)

## Status
Accepted

## Context
The MoeaBench library (v0.3.0) transition to a trajectory-based architecture, where an `experiment` contains multiple `Run` objects with full generational history, created a need for a more robust persistence system. The legacy "capture-based" saving system was incompatible with the new data model and lacked the flexibility required for scientific workflows where data volume and configuration reproducibility are distinct concerns.

Additionally, internal components like DEAP algorithm wrappers introduced non-serializable attributes (e.g., `base.Toolbox`), which caused failures during standard object pickling.

## Decision
We have implemented a Selective Persistence system that allows users to save and load experiments with functional granularity.

1.  **Object-Level Serialization**: We use `joblib` for high-performance serialization of the entire `experiment` object. This preserves the internal state, including all NumPy-based trajectories.
2.  **Selective Modes**: We introduced the `mode` argument (`all`, `config`, `data`) to decouple execution results from algorithmic configurations:
    *   `mode='config'`: Persists only the "DNA" (metadata and parameters) of the experiment.
    *   `mode='data'`: Persists the trajectories (`runs`) and consolidated results.
3.  **Cross-Tool Interoperability**: The persistence format is now a ZIP archive containing not just the binary object but also standardized CSV files (`result.csv`, `pof.csv`) and a human-readable manifest (`problem.txt`).
4.  **Pickle Guarding**: We implemented `__getstate__` in the `BaseMoea` class to automatically sanitize objects before serialization, specifically clearing non-serializable functional attributes like DEAP's `toolbox`.

## Consequences
- **Positive**: Researchers can now share small configuration files (KB) or full result trajectories (MB/GB) depending on the need.
- **Positive**: The library is now robust against pickling errors from third-party algorithm engines.
- **Positive**: Data is now accessible outside Python via the included CSV files in the ZIP archive.
- **Negative**: The `load(mode='data')` requires the destination object to be pre-configured with a compatible MOP to maintain analytical integrity.
- **Neutral**: The system requires `joblib` as a core dependency for efficient NumPy serialization.
