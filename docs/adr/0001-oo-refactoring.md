# ADR 0001: Object-Oriented Refactoring and Data Hierarchy

**Status**: Accepted  
**Date**: 2026-01-14

## Context
The legacy MoeaBench codebase was a monolithic and procedural collection of scripts. It suffered from several critical architectural flaws:
1.  **Inconsistent Naming**: Files were prefixed with cryptic codes (e.g., `I_*.py`, `H_*.py`), which obscured the project structure and made discoverability nearly impossible.
2.  **Global State Dependency**: Data flow relied on implicit global variables and a brittle file-based caching system, leading to "spooky action at a distance" where changing one parameter could break distant modules.
3.  **Procedural Bloat**: Experiments were defined by modifying large, flat scripts rather than interacting with a clean API, making it difficult to automate large-scale benchmarks.
4.  **Opaque Data Structures**: Populations and results were passed around as raw dictionaries or lists of lists, making it hard to track what metadata (labels, seeds, generations) belonged to which data point.

## Decision
We decided to dismantle the legacy procedural scripts and rebuild MoeaBench as a structured Python package based on a clear **Object-Oriented Data Hierarchy**.

### 1. The Hierarchical Model
We established a chain of command for data:
- **`Experiment`**: The high-level orchestrator. It holds the MOP (problem), the MOEA (algorithm), and the history of results.
- **`Run`**: Represents a single optimization trajectory with a unique seed. It stores the generational history of a single execution.
- **`Population`**: A snapshot of the search at a specific generation, encapsulating both objective and decision space data.
- **`SmartArray`**: A customized NumPy ndarray subclass that carries metadata (labels, axis labels, sources) through mathematical operations.

### 2. Package Modularity
We moved away from the "flat directory" model to logical sub-packages:
- `MoeaBench.core`: The fundamental data structures and execution engine.
- `MoeaBench.mops`: Standard and custom multi-objective problems.
- `MoeaBench.moeas`: The execution wrappers for algorithm engines.
- `MoeaBench.stats` & `MoeaBench.metrics`: Specialized tools for analysis.

### 3. Modern Tooling
- **Type Hinting**: Implemented comprehensive Python type hints (`List`, `Optional`, `Any`) across the API to allow for static analysis and superior IDE autocomplete.
- **Vetorization First**: Mandatory use of NumPy for all core data manipulation to avoid the speed penalties of Python loops.

## Consequences
- **Positive**: The library is now highly discoverable. A user can type `exp.` and see exactly what methods are available for analysis.
- **Positive**: Data integrity is preserved across the stack. A `SmartArray` from a specific run knows its own name and label, facilitating automatic plot generation.
- **Positive**: Testing is simplified as components (like MOPs or Metrics) are now isolated and can be unit-tested without loading the entire framework.
- **Negative**: The refactor constitutes a 100% breaking change for any scripts relying on the legacy `I_` or `H_` files.
- **Neutral**: Increased architectural complexity (multiple classes) over the original "one file does all" script model.
