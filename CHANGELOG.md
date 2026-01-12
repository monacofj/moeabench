<!--
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **New Directory Structure**: `core/`, `metrics/`, `plotting/` packages established.
- **Performance**: Vectorized Non-Dominated Sort implementation (NumPy broadcasting) in `core/run.py`.
- **Type Hints**: Comprehensive Python typing added to all Core and Benchmark components.
- Static plotting support for `timeplot` and `spaceplot` via `mode='static'` argument.
- `verify_api.py` regression test to validate API functionality.
- New `example-01.py` demonstrating the updated API usage.
- **Flexible Parameters**: Implemented `**kwargs` propagation across all MOEAs (`NSGA3`, `MOEAD`) and Benchmarks (`DTLZ`, `DPF`), allowing advanced parameter tuning.
- **Documentation**: Comprehensive `docs/reference.md` (API) and `docs/userguide.md` (How-to) created.
- **Examples**: Added `example-05.py` (Flexible Parameters) and `example-custom.py` (Custom Extensions).

### Changed
- **Major API Refactoring**: Restructured `MoeaBench` to follow a new object-oriented design (e.g., `mb.experiment`, `mb.moeas`, `mb.benchmarks`).
- **Clean Namespace**: Standardized filenames to `snake_case` and removed 20+ legacy interface files (`I_*.py`, `H_*.py`) and the `CACHE` system.
- **Dependency Hygiene**: Pruned unused packages (`ordered_set`, `deepdiff`, `PyGithub`) from `requirements.txt`.
- `example-01.py` updated to use the new `MoeaBench` API and `DTLZ2` benchmark.
- `README.md` Quick Start example updated to match the new API.

### Fixed
- **CRITICAL**: Fixed `BaseMoeaWrapper` ignoring random seeds (`seed=...`), ensuring independent runs are now mathematically unique.
- **Plotting**: Fixed static 3D plots to use standard Matplotlib color cycle (`prop_cycle`), preventing identical colors for multiple runs.
- **API**: Restored missing metric shortcuts (`mb.hv`, `mb.igd`, `mb.gd`, `mb.gdplus`, `mb.igdplus`).
- Fixed plotting backends to correctly propagate the `mode` argument.
- Fixed typo "Vizualize" in `example-01.py` comments.

## [0.1.0] - 2025-01-10

### Added
- Initial import of project files from `fervat40`'s repository.
