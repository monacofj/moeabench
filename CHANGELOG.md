<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.6] - 2026-01-31

### Added
- **Consolidated Interactive Dashboard**: Replaced legacy Markdown reports with a unified HTML Certification Dashboard featuring interactive topography, statistical tables, and stabilization efficiency analysis.
- **Metric Terminology Refinement**: Transitioned from "Max Theoretical HV" to **"Sampled Reference HV"** to accurately reflect the discrete nature of Ground Truth reference sets.
- **DPF High-Density Policy**: Increased Ground Truth density to **10,000 points** for 1D degenerate curves (DPF family) to eliminate Hypervolume discretization artifacts.
- **Discretization Rigor Documentation**: Codified the interpretation of "Performance Saturation" (HV > 100%) in scientific reports.
- **DPF Family Scientific Rectification**: Implemented structured analytical sampling in `ps()` for the DPF family, ensuring perfectly geometric reference clouds.
- **DTLZ8 Analytical Solver**: Implemented a guided analytical manifold solver for the DTLZ8 problem, ensuring strictly feasible theoretical ground truth for any $M$.
- **Scientific Audit Suite (`Legacy2_optimal`)**: Implemented a comprehensive confrontation suite to validate v0.7.5 ground truth against high-precision legacy data.
- **Baseline Calibration Suite**: Introduced systematic quantitative benchmarking (`calibrate_baselines.py`) to establish definitive v0.7.6 IGD/KS thresholds.

### Changed
- **DPF Projection Restoration**: Restored squared projections in DPF2 and DPF4 to satisfy spherical parities, resolving previous geometric discrepancies.
- **Gallery Standardization**: Updated all benchmark audit notebooks to use NSGA-III as the default verification engine with optimized population/generation parameters.
- **DTLZ4 Geometric Rectification**: Replaced uniform decision-space sampling with uniform angular ($\Theta$) sampling to resolve spatial sparsity and "floating points" caused by extreme mapping bias ($x^{100}$).
- **Textbook-Style Reporting**: Introduced a new didactic narrative format for scientific reports (`docs/legacy2_optimal_report.md`) prioritizing technical depth over bulleted summaries.
- **Repository Curation**: Purged legacy v1 audit artifacts and redundant PoC notebooks; unhidden `wish.md` for better developer visibility.

### Fixed
- **Progress Bar Restoration**: Fixed missing progress updates in the `NSGA2deap` (DEAP) engine to ensure real-time feedback in notebooks.
- **DTLZ9 Parity**: Fixed per-objective sampling logic to strictly satisfy spherical parities in high dimensions.

## [0.7.5] - 2026-01-22

### Added
- **Data Export API (`mb.system.export_*`)**: Implemented new utilities to export experiment and population data directly to CSV files. 
  - `export_objectives()`: Extracts and saves the Pareto front (Experiment) or objectives (Population).
  - `export_variables()`: Extracts and saves the Pareto set (Experiment) or decision variables (Population).
- **Smart Naming & Resolution**: Added automatic data resolution and intelligent naming conventions that leverage the experiment's metadata for default filenames.
- **CSV Support**: Integrated `pandas`-based export with column headers (`f1, f2...`, `x1, x2...`), falling back to robust NumPy CSVs when pandas is unavailable.

## [0.7.4] - 2026-01-22

### Changed
- **Axis Nomenclature Standardization**: Unified all user-facing reports and visualizations to follow a **1-indexed** axis nomenclature (e.g., "Axis 1", "f1"). 
- **Fixed Inconsistency**: Updated `DistMatchResult.report()` in the statistics engine to match the visualization layer's convention, ensuring a consistent didactic experience across the entire framework.

## [0.7.3] - 2026-01-21

### Fixed
- **Synchronized Optimal Sampling**: Refactored `exp.optimal()` to ensure a strict row-by-row correspondence between the Pareto Set ($X$) and the Pareto Front ($F$).
- **Theoretical Manifold Filtering**: Implemented automatic non-dominance filtering in `optimal()` to handle benchmarks where $g=min$ amonts to a super-set of the Pareto front (e.g., DTLZ7).
- **Corrected DTLZ9 Manifold**: Updated the DTLZ9 analytical Pareto Set sampling to correctly represent the $(M-1)$-dimensional hypersphere manifold for $M > 2$ objectives.

## [0.7.2] - 2026-01-21

### Added
- **Many-Objective Support**: Removed artificial constraints on the maximum number of objectives ($M$) and variables ($N$) across the framework.
- **Didactic Error Messages**: Replaced generic `ValueError` or `IndexError` in DTLZ problems with detailed, textbook-style explanations of theoretical variable requirements (position vs. distance variables), including references to foundational literature.
- **Performance Guards**: Implemented `UserWarning` indicators for high-complexity operations (e.g., exact Hypervolume for $M > 8$), suggesting more efficient alternatives for many-objective scenarios.

### Changed
- **DPF Unlimited**: Removed the $M \le N$ constraint from the DPF (Degenerate Pareto Front) problem family, enabling high-dimensional objective mapping as originally intended.
- **Robust Metrics**: Updated the `normalize` function in the metrics evaluator to dynamically handle any number of objectives, preventing dimension fallback mismatches.

## [0.7.1] - 2026-01-20

### Added
- **Permanent Aliases**: Promoted `mb.spaceplot`, `mb.timeplot`, and `mb.rankplot` to permanent status due to their stability and widespread use in the literature.
- **Full Example Migration**: Updated all official examples (`examples/*.py` and `examples/*.ipynb`) to use the new **Scientific Domains** taxonomy as the standard nomenclature.
- **Policy Formalization**: Established the "Soft Deprecation" vs "Permanent Alias" tiers in the documentation.
- **Taxonomy Refinement**: Renamed `strat_hierarchy` to `strat_caste` for better conceptual alignment with population geology.
- **Custom Stop Criteria**: Restored and standardized the ability to inject custom termination logic (`exp.stop` or `exp.run(stop=lambda...)`) into any algorithm (Legacy/Pymoo), receiving the algorithm instance as context.

### Changed
- **Version Bump**: Updated library version to `0.7.1`.
- **Documentation Cleanup**: Purged all legacy breadcrumbs ("Successor to...") from the User Guide to focus exclusively on the Scientific Taxonomy.

## [0.7.0] - 2026-01-20

### Added
- **Scientific Domains Taxonomy**: Established a three-factor analytical framework for all visualizations and statistics:
  - **Topography (`topo_`)**: Focus on "Where solutions are". New views: `topo_shape` (successor to `spaceplot`), `topo_bands` (Search Corridors/EAF), `topo_gap` (Topologic Gap/EAF Difference), and `topo_density` (Spatial Density/KDE).
  - **Performance (`perf_`)**: Focus on "How well algorithms perform". New views: `perf_history` (successor to `timeplot`), `perf_spread` (Performance Contrast/A12 Boxplots), and `perf_density` (Quality Distribution/KDE).
  - **Stratification (`strat_`)**: Focus on "How population is organized". New views: `strat_ranks`, `strat_caste`, and `strat_tiers`.
- **Full Word Statistics API**: Renamed statistical functions for clarity and taxonomic consistency:

| Legacy Name | New Name | Analytical Domain |
| :--- | :--- | :--- |
| `perf_prob` | `perf_probability` | Performance (A12) |
| `perf_dist` | `perf_distribution` | Performance (KS) |
| `topo_dist` | `topo_distribution` | Topography (Match) |
| `topo_attain` | `topo_attainment` | Topography (EAF) |

### Changed
- **Modular Architecture**: Visualization layer refactored into domain-specific modules (`MoeaBench/view/topo.py`, `perf.py`, `strat.py`).
- **Stability Policy (Legacy Support)**: Established a two-tier support system:
  - **Permanent Aliases**: `spaceplot`, `timeplot`, and `rankplot` are maintained as first-class citizens.
  - **Soft-Deprecated**: Other legacy functions (`casteplot`, `tierplot`, `topo_dist`, `perf_prob`, etc.) are maintained for compatibility but will be restricted in future major releases.

### Fixed
- Fixed internal `AttributeError` in `HypothesisTestResult` reporting when calling renamed statistical methods.

## [0.6.3] - 2026-01-20
### Added
- Robust input validation guards for DPF problems to prevent mathematical inconsistencies.
- Population size validation for DEAP-based MOEAs (selTournamentDCD guard).
- Standardized `**kwargs` support across all benchmark constructors for improved extensibility.
- New regression test suite (`test/unit/test_consistency_regression.py`).

### Fixed
- Resolved numerical drifts in `DTLZ6` benchmark.
- Fixed `TypeError` collisions in custom MOP constructors.

## [0.6.2] - 2026-01-20

### Added
- **Integrated Testing Infrastructure**:
    - `test/test.py`: Unified CLI orchestrator with granular control (`--unit`, `--scripts`, `--notebooks`).
    - `test/nb_runner.py`: Headless Jupyter execution engine for validating notebooks in CI/CD.
    - `test/unit/`: Dedicated directory for unit tests, including initial system checks (`test_system.py`).
    - **Self-Healing**: Automated dependency checking and kernel registration prompts for notebook environment setup.

### Fixed
- **Notebook Regression**: Fixed legacy function call `topo_attain_diff` in `examples/example_07.ipynb` (renamed to `topo_gap`), restoring pass/fail integrity.

## [0.6.1] - 2026-01-20

### Changed
- **Taxonomy Alignment**: Migrated `attainment.py` to `topo_attain.py` to match the Topologic Domain nomenclature.
- **Refinement**: General internal visual and naming adjustments in the visualization layer.

## [0.6.0] - 2026-01-19

### Added
- **Final Semantic Taxonomy**: Standardized researcher-centric nomenclature for all analysis tools:
    - Performance Domain (`perf_*`): `perf_evidence` (Mann-Whitney U), `perf_prob` (Vargha-Delaney A12), `perf_dist` (Kolmogorov-Smirnov).
    - Topologic Domain (`topo_*`): `topo_dist` (Multi-axial KS/Anderson/EMD matching), `topo_attain` (Empirical Attainment Functions), `topo_gap` (EAF Differences).
- **API Symmetry**: Unified naming between statistics and visualization (e.g., `mb.view.topo_dist`).
- **Explicit Methodology**: All documentation and internal docstrings now explicitly state the underlying statistical methods (Mann-Whitney, KS, A12, EAF).
- **Scientific Distribution Plot (`mb.view.topo_dist`)**: High-quality academic visualization for probability densities (KDE), integrated with statistical matching results. Supports grid and independent layouts.
- **Example Version Reporting**: Integrated discreet library version printing (`Version: 0.6.0`) in the start of all 10 examples to improve reproducibility.

### Changed
- **Clean Slate Policy**: Removed all legacy aliases and deprecated shortcuts (formerly `mann_whitney`, `a12`, `ks_test`, `dist_match`, `attainment`, `distplot`). v0.6.0 establishes a clean break focused on semantic clarity.
- **Documentation**: Major rewrite of `userguide.md` and `reference.md` to establish the new taxonomy and remove all legacy cross-references.

### Fixed
- **Benchmark Robustness (DTLZ8/9)**:
    - Resolved critical $M=2$ crash in `DTLZ8` and improved input validation.
    - Implemented analytical Pareto Set sampling (`ps()`) for `DTLZ9` to support statistical metrics.
    - Added informative diagnostic messages for `DTLZ8` topological limitations.

## [0.5.0] - 2026-01-19

### Added
- **Topological Analysis (`mb.stats.topo_dist`)**: Implementation of a new multi-axial statistical engine to verify convergence equivalence. Supports multi-sample KS, Anderson-Darling, and Wasserstein (EMD).
- **Rich Result (`DistMatchResult`)**: Comprehensive reporting object for dimensional analysis, profiling exact axes where algorithms diverge.
- **Educational Examples**: Created `examples/example_10.py` and `.ipynb` demonstrating multimodality detection through decision-space matching.
- **ADR 0013**: Formalization of the "Ocean" centralized visual identity.

### Changed
- **Pedagogical Refinement**: Major rewrite of the `User Guide` (Sections 6-8) adopting a technical textbook narrative and the "Scale of Abstraction" framework for data selection.
- **Documentation Accuracy**: Exhaustive audit of selector equivalents (e.g., `exp.pop().objs`) and property documentation in the guide and technical reference.

### Fixed
- **Markdown Integrity**: Resolved multiple rendering failures in the User Guide due to unclosed code blocks in the Statistics and Persistence sections.

## [0.4.1] - 2026-01-18

### Fixed
- **DPF Mathematical Restoration**: Corrected a fundamental error in the Degenerate Pareto Front (DPF) benchmark engine. Chaotic weights are now **static and unsorted** (stored in `__init__`), restoring the characteristic "stepped" front geometry in DPF1-5.
- **Tabula Rasa v2 (Code Cleanup)**: Permanently removed the legacy code graveyard, including `problem_benchmark/`, `kernel_moea/`, and multiple orphan factory files (`problems.py`, `I_problems.py`, etc.).
- **Architecture Integrity**: Refactored `mops` and `moeas` module initializers to use explicit, modern imports, eliminating unsafe dynamic loading loops (`os.walk`).
- **Algorithm Reorganization**: Migrated essential `pymoo` kernels directly into `MoeaBench/moeas/` as internal helpers (`_*.py`), streamlining the internal execution layer.

## [0.4.0] - 2026-01-18

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
- **Parallel Execution Support**: All support for parallel execution (`multiprocessing`, `concurrent.futures`) has been removed from the library core (`experiment.run`, `topo_gap`).
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
