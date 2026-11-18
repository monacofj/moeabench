<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0048: Temporary Removal of FAIR from the 0.x Release Line

## Status

Accepted — supersedes the former public clinical API decision for the FAIR-free release.

## Context

The FAIR physical metrics, their Q-scores, and the aggregate clinical audit
remain active research and are not ready to be part of the public release.
The FAIR-free cleanup temporarily removed them to allow the independent,
mature core of MoeaBench to be published and reviewed.

## Decision

For the FAIR-free 0.x release line and its 1.0.0 preparation:

- The `mb.clinic` public API, individual FAIR metrics, Q-scores, and
  `audit()` are unavailable. A public caller must not receive a silently
  substituted or fabricated clinical result.
- Conventional indicators in `mb.metrics` and their established
  `raw`/`rel`/`abs` semantics remain available and independent of FAIR.
- `mop.calibrate()`, analytical or empirical Ground Truth and calibration
  persistence are core MOP features, independent of FAIR.
- Reintroduction of FAIR requires a new reviewed architectural decision and
  regression evidence. This ADR does not select or prescribe a future FAIR API.
- Historical FAIR code and decisions can remain in the separate research
  branch; public documentation for the FAIR-free release must describe only
  the supported functionality.

## Consequences

The FAIR-free release exposes conventional MOEA/MOP experiments, metrics,
statistics, visualization and persistence without claiming that experimental
FAIR measurements are already certified. Historical revisions of this ADR
remain available in Git history; they are not the active public API contract.
