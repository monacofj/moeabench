<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Ocean Palette**: Implemented a new custom categorical color palette for all visualizations. The 9-color sequence (Indigo → Emerald → Plum → Jade → Bordeaux → Deep Teal → Orange → Red → Yellow) provides a premium, high-contrast visual identity.
- **Centralized Style System**: Infrastructure for global theme management added in `MoeaBench/view/style.py`, ensuring consistent aesthetics between Matplotlib and Plotly backends.
- **Selective Persistence (Save/Load)**: Implemented a new, robust persistence system for experiments.
    - Added `mode` argument (`all`, `config`, `data`) to `exp.save()` and `exp.load()`.
    - ZIP-based archive format containing standardized CSVs (`result.csv`), metadata (`problem.txt`), and serialized trajectories (`Moeabench.joblib`).
    - Integrated with `joblib` for high-performance NumPy-aware serialization.

### Changed
- **Automatic Branding**: The new "Ocean" visual identity is now automatically applied upon importing `MoeaBench.view`.
- **Documentation Audit**: Completed a thorough audit of all guides (`userguide.md`, `reference.md`, `design.md`), renaming the benchmarks technical guide to `docs/mops.md` and standardizing all references to the namespaced API.
- **Reference Provenance**: Added formal DOI and arXiv links for DTLZ and DPF references across all documentation guides.

### Fixed
- **Notebook Robustness**: Fixed a critical literal newline encoding issue in `examples/example_08.ipynb` that caused syntax errors in certain cloud execution environments (Google Colab/GitHub).
- **Pickling Stability**: Resolved a critical `PicklingError` affecting DEAP-based algorithms. Implemented `__getstate__` in `BaseMoea` to automatically strip non-serializable attributes (like the `toolbox`) before saving, ensuring stability across standard and selective save modes.
- **Save/Load restoration**: Restored the broken persistence functionality, removing legacy dependencies on internal file modules and the defunct `CACHE` system.

## [0.3.0] - 2026-01-15

### Changed
- **Break The World (Alpha Policy)**: Refactored the core architecture to enforce strict namespaced API usage.
- **Tier Nomenclature Refactor**: Standardized all competitive analysis terminology on **"Tier"**.
    - Replaced `arena` with `tier` and `domplot` with `tierplot`.
    - Updated reporting to use **"Dominance Ratio"** and **"Displacement Depth"** as primary metrics.
    - **Tabula Rasa**: Removed all legacy aliases (`mb.arena`, `mb.domplot`) and backward compatibility notes ("formerly known as") across the entire codebase and documentation.
- **Advanced Tier Plot**: Improved `tierplot` to display **absolute population counts** on the y-axis, providing a dual view of dominance proportions and rank density.
- **Documentation Overhaul**: All documentation guides (`userguide.md`, `reference.md`, `design.md`, `mops.md`), scripts, and notebooks have been updated to exclusively use the new namespaced API and Tier nomenclature.

### Added
- **Metric Alias**: Added `hv` alias in `mb.metrics` (`mb.metrics.hv`) to streamline the most common metric call.

## [0.2.0] - 2026-01-15

### Added
- **Smart Hypervolume**: Implementation of automatic Monte Carlo approximation for high-dimensional problems ($M > 6$), with vectorized NumPy computation.
- **Rich Stats Results**: New structured output for statistical tests (`mb.stats`) featuring lazy evaluation of effect sizes (A12, structural EMD) and narrative reporting via `.report()`.
- **Mirror Parity Architecture**: Unified plotting functions (`spaceplot`, `timeplot`) with environment-aware `mode='auto'` (uses Plotly in notebooks and Matplotlib locally).
- **Environment Awareness**: Automated configuration for Google Colab and interactive backends.

### Changed
- **Standardized Examples**: Standardized all notebook examples with direct `pip install` commands and removed redundant `mb_path` imports.
- **Repository Migration**: Moved official repository to `https://github.com/monacofj/moeabench`.
- **Zen Cleanup**: Simplified all example scripts by removing redundant `mode='static'` manual overrides.

### Fixed
- **Robustness**: Implemented fallback guard mechanisms for metrics (IGD/GD) when analytical fronts are missing from custom MOPs.
- **Plotting Bug**: Fixed `TypeError` in `mb.timeplot`/`plot_matrix` when handling custom titles and restored missing mean calculation logic.
- **Packaging**: Corrected `pyproject.toml` to use automatic sub-package discovery, fixing `ModuleNotFoundError` in cloud environments.
- **Interactive Experience**: Improved Plotly hover tooltips to show exact coordinates and configured reliable renderers for Jupyter/Colab.

### Added
- **New Directory Structure**: `core/`, `metrics/`, `plotting/` packages established.
- **Performance**: Vectorized Non-Dominated Sort implementation (NumPy broadcasting) in `core/run.py`.
- **Type Hints**: Comprehensive Python typing added to all Core and Benchmark components.
- Static plotting support for `timeplot` and `spaceplot` via `mode='static'` argument.
- `verify_api.py` regression test to validate API functionality.
- New `example_01.py` demonstrating the updated API usage.
- **Statistical Analysis**: Created `mb.stats` module featuring **"Smart Stats"** API, allowing direct comparison of `Experiment` and `MetricMatrix` objects. Supported tests: **Mann-Whitney U**, **Vargha-Delaney A12**, and **Kolmogorov-Smirnov (KS)**.
- **Analytical Optimals**: Implemented `exp.optimal()`, `exp.optimal_front()`, and `exp.optimal_set()` for theoretical result sampling on standard MOPs.
- `example_06.py` demonstrating statistical comparison and Pareto testing.
- Comprehensive `docs/reference.md` (API) and `docs/userguide.md` (How-to) created.
- **Improved Metrics**: `MetricMatrix` now features `.gens()` and `.runs()` robust selectors.
- **System Module**: Added `mb.system` for environmental awareness (`version`, `check_dependencies`).

### Changed
- **Terminology**: Renamed `benchmarks` to **`mops`** and `BaseBenchmark` to **`BaseMop`** globally for technical accuracy and brevity.
- **Major API Refactoring**: Restructured `MoeaBench` to follow a new object-oriented design (e.g., `mb.experiment`, `mb.moeas`, `mb.mops`).
- **Clean Namespace**: Standardized filenames to `snake_case` and removed 20+ legacy interface files (`I_*.py`, `H_*.py`) and the `CACHE` system.
- **Dependency Hygiene**: Pruned unused packages (`ordered_set`, `deepdiff`, `PyGithub`) from `requirements.txt`.
- `example_01.py` updated to use the new `MoeaBench` API and `DTLZ2` benchmark.
- `README.md` Quick Start example updated to match the new API.

### Removed
- **Parallel Execution Support**: All support for parallel execution (`multiprocessing`, `concurrent.futures`) has been removed from the library core (`experiment.run`, `attainment_diff`).
- **Rationale**: The overhead of Python's process-based parallelism (especially `spawn` mode required for stability) often exceeded the computational gains for typical MOEA runs. On standard hardware (e.g., 8GB RAM), the memory pressure from multiple worker processes proved unstable. Serial execution is now prioritized for robustness and simplicity.

### Fixed
- **CRITICAL**: Fixed `BaseMoeaWrapper` ignoring random seeds (`seed=...`), ensuring independent runs are now mathematically unique.
- **Plotting**: Fixed static 3D plots to use standard Matplotlib color cycle (`prop_cycle`), preventing identical colors for multiple runs.
- **API**: Restored missing metric shortcuts (`mb.hv`, `mb.igd`, `mb.gd`, `mb.gdplus`, `mb.igdplus`).
- Fixed plotting backends to correctly propagate the `mode` argument.
- Fixed typo "Vizualize" in `example_01.py` comments.
- **Fixed Parallel Reporting**: Resolved issue where workers failed to report progress due to missing local instantiation.

## [0.1.0] - 2025-01-10

### Added
- Initial import of project files from `fervat40`'s repository.
