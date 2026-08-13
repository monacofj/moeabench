<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0043: Hypervolume Engine Selection and Reproducible Monte Carlo

## Status

Accepted

## Context

The automatic Hypervolume policy switched to Monte Carlo above 6 objectives. A seven-objective history with 300 generations therefore drew 100,000 samples independently for every generation and checked each sample against every front point. This made metric evaluation substantially slower than the optimization itself, even though pymoo 0.6.2 provides a fast exact backend for this supported dimensional range.

The approximation also used NumPy's process-global random generator, so repeated calls could differ, and progress was updated only after an entire run history had finished. Explicit reference lists silently ignored experiments that had not been executed, making the actual normalization context differ from the requested one.

## Decision

- `mode='auto'` selects exact Hypervolume for up to 8 objectives and Monte Carlo above 8.
- `mode='exact'` always selects exact evaluation and warns above 8 objectives.
- `mode='fast'` always selects Monte Carlo.
- pymoo 0.6.2 is the minimum supported version.
- Monte Carlo uses a dedicated seeded `numpy.random.Generator`. `mc_seed=None` resolves to `mb.defaults.seed`.
- A common seeded random-weight sequence is reused across the generation history, and approximation delegates to moocore's native `DZ2019-MC` backend.
- Both engines notify the metric progress bar after each generation.
- Every supplied external reference item must contain at least one evaluated front.
- The result records backend provenance and Monte Carlo parameters in `MetricMatrix.diagnostics`.

## Consequences

- Seven- and eight-objective histories use the faster exact implementation by default and avoid unnecessary sampling error.
- Monte Carlo results are repeatable for equal data, settings, and seeds; common random numbers also reduce sampling noise in generation-to-generation comparisons.
- Approximate evaluation avoids the former Python loop over every front point and exposes visible progress for long histories.
- References accurately reflect the user's explicit list; incomplete experiments fail early with an actionable error.
- Users who deliberately prefer approximation at 8 or fewer objectives can still request `mode='fast'`.
