<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0049: Longitudinal Headway Definition

## Status

Accepted

## Date

2026-08-27

## Context

ADR 0037 redefined `HEADWAY` as a dimensionless search-progress quantity, but
specified its denominator as a random-search baseline. The current implementation
instead measures progress relative to the actual initial state of the evaluated
search. The User Guide and API Reference already describe this longitudinal
behavior.

A diagnostic intended to measure search progress should compare the final state
with the search state from which that same run actually started. Using the
initial population also avoids introducing a separate stochastic/random reference
into the physical definition of the metric.

## Decision

The canonical physical `HEADWAY` metric is the fraction of the initial
95th-percentile convergence error that remains at the evaluated final state:

\[
\mathrm{HEADWAY}
=
\frac{GD_{95}(P_{final}\to GT)}
     {GD_{95}(P_{initial}\to GT)}.
\]

The following behavior is normative:

- `P_final` is the evaluated final front/population.
- `P_initial` is generation 0 for an `Experiment` or `Run`. For lower-level
  inputs, callers may provide equivalent longitudinal context through
  `initial_data`.
- `GT` is resolved through the ordinary clinical diagnostic context.
- The default result is dimensionless and lower is better.
- `0.0` means the initial convergence error has been completely removed.
- `1.0` means the final 95th-percentile convergence error equals the initial
  error; values above `1.0` are valid and indicate that the final state is
  farther from GT than the initial state under this statistic.
- The metric is not clipped.
- If initial longitudinal context is unavailable, the default HEADWAY value is
  undefined (`NaN`) rather than being replaced by a random baseline.
- If the initial error is numerically zero, the implementation returns `0.0`
  when the final error is also zero and `1.0` when the final error is positive,
  avoiding division by an effectively zero denominator.
- `headway(..., raw=True)` is a separate physical view: it reports the final
  `GD95(P_final -> GT)` in resolution-scaled units (`s_k`) rather than the
  longitudinal ratio.

The quality layer (`q_headway`) remains a calibrated interpretation of the
physical HEADWAY value and does not redefine this physical measurement.

## Consequences

- HEADWAY measures progress of the actual search trajectory rather than progress
  relative to a synthetic random-search denominator.
- Two runs with equal final convergence can have different HEADWAY values if
  they began at different distances from GT; this is intentional because the
  metric is longitudinal.
- A raw array without an initial state cannot support the default longitudinal
  interpretation unless `initial_data` is supplied.
- `HEADWAY` remains distinct from `CLOSENESS`: closeness measures final proximity
  in resolution terms, while HEADWAY measures the fraction of initial search
  error remaining.
- The HEADWAY definition in Section 3 of ADR 0037 is historical and is
  superseded by this ADR. ADR 0037 remains valid for the Plausible Q1 closeness
  correction and the decision to make HEADWAY dimensionless.

## Relationship to Earlier Decisions

- [ADR 0029](0029-headway-nomenclature.md) introduced the HEADWAY name.
- [ADR 0036](0036-half-normal-closeness-v0.12.0.md) belongs to the earlier
  clinical physical-layer evolution.
- [ADR 0037](0037-plausible-q1-and-search-drive-headway-v0.13.1.md) made
  HEADWAY dimensionless but used the now-superseded random-baseline denominator.
- [ADR 0048](0048-public-clinical-metrics-and-audit-context-parity.md) defines
  public clinical-call and audit-context parity; both paths therefore use this
  same longitudinal HEADWAY semantics.
