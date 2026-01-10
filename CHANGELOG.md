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
- Static plotting support for `timeplot` and `spaceplot` via `mode='static'` argument.
- `verify_api.py` regression test to validate API functionality.
- New `example-01.py` demonstrating the updated API usage.

### Changed
- **Major API Refactoring**: Restructured `MoeaBench` to follow a new object-oriented design (e.g., `mb.experiment`, `mb.moeas`, `mb.benchmarks`).
- `example-01.py` updated to use the new `MoeaBench` API and `DTLZ2` benchmark.
- `README.md` Quick Start example updated to match the new API.

### Fixed
- Fixed plotting backends to correctly propagate the `mode` argument.
- Fixed typo "Vizualize" in `example-01.py` comments.

## [0.1.0] - 2025-01-10

### Added
- Initial import of project files from `fervat40`'s repository.
