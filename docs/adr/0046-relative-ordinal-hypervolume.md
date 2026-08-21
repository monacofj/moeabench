<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0046: Relative Scaling for Ordinal Hypervolume

## Status

Accepted

## Context

ADR 0044 introduced raw Ordinal Hypervolume (OHV) and deliberately left
relative variants outside its initial scope. Raw OHV is auditable ordinal-cell
volume, but separate values are easier to compare when expressed against one
observed performance baseline. That scaling must not change the ordinal lattice,
reference point, backend, or raw geometry diagnostics.

Normalizing against the pooled union of reference fronts would create an
artificial super-front that no run attained. Normalizing independently by
generation would move the scale over time. Clipping at one would also erase
legitimate evidence that a historical generation exceeded the best final
reference performance.

## Decision

- `ordinal_hypervolume(..., scale="raw")` retains the original mathematics and
  remains the default. `scale` is keyword-only to preserve existing positional
  calls.
- `scale="rel"` divides the complete raw matrix by one fixed denominator: the
  greatest raw OHV among the individual final reference fronts in the common
  ordinal lattice.
- Each run contributes one individual final-front candidate. Runs are never
  pooled before their candidate OHVs are evaluated.
- With no external `ref`, the final fronts of the evaluated runs are the
  candidates. With external references, only their individual final fronts are
  candidates.
- The denominator uses the same ordinal reference point, exact or Monte Carlo
  backend, sample count, and Monte Carlo seed as the evaluated trajectory.
  Reusing the seed provides common random numbers. Auxiliary denominator work
  does not advance generation progress.
- A non-positive denominator raises `ValueError`. Relative values are not
  clipped and may exceed one.
- Raw ordinal geometry is preserved before scaling. In particular,
  `raw_ohv_fraction_of_ordinal_box` remains independent of reporting scale.
- `scale="abs"` is unsupported because Ground Truth sampling would require a
  separate scientific definition of the ordinal lattice.

## Consequences

- `1.0` means equality with the best individual final performance in the shared
  context, not a mathematical ceiling or complete ordinal-box coverage.
- Separate calls using the same `ref` share both their ordinal ruler and
  relative denominator.
- `gens` still selects only returned trajectory rows. Even `gens=0` retains a
  fully defined reference context and denominator.
- Reports distinguish relative scaling from `Raw OHV/OBox`, where the latter is
  raw OHV divided by total ordinal-box volume.
- Monte Carlo denominators are reproducible estimates and are reported as such.
- Absolute, calibrated, weighted, adaptive-rank, and mixed-direction OHV remain
  outside the public API.
