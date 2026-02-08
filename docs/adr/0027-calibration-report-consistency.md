<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0027: Calibration Report ↔ Clinical Payload Contract

## Status
Proposed

## Context
The interactive calibration dashboard must remain a single source of truth. Historically,
the HTML matrix displayed `status/verdict` badges derived from the diagnostics module
but the payload used in `tests/calibration/generate_visual_report.py` did not satisfy the
expected contract: it injected raw `gd/igd/emd` numbers, causing the auditor to default to
`inf` efficiencies and `SEARCH_FAILURE` or repeated `IDEAL_FRONT` badges, depending on
how the baseline data was structured.

Meanwhile, the `clinic` module publishes efficiency ratios (`igd_eff`, `emd_eff_uniform`)
and `gd_p95` that represent true clinical context. The calibration report also mixes data
from the 30-run CSVs and the `run00` snapshots, so the contract must be explicit.

## Decision

1.  The HTML generator now builds the audit payload from the computed `gd_p95`, ratio-based
    efficiencies, and the current `run00` snapshot, ensuring the auditor receives consistent inputs.
2.  The matrix renders `IGD_eff`/`EMD_eff_uniform` values returned by `audit()` and annotates
    the origin of the payload (run00) versus the baseline aggregate (30 runs).

This decision keeps the diagnosis deterministic and prevents future regressions where the
badges "always pass" or "always fail" regardless of the underlying geometry.

## Consequences

*   **Positive**: The calibration HTML now faithfully mirrors the `auditor`'s logic (status + verdict).
*   **Positive**: Future tuning can reason about a static contract (`gd_p95`, `igd_eff`, `emd_eff_uniform`).
*   **Effort**: Any future refactor of `audit()` must document the payload again to keep the report consistent.

