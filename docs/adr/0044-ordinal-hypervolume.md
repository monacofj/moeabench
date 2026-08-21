<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0044: Ordinal Hypervolume

## Status

Accepted

## Context

Conventional Hypervolume (HV) measures dominated geometric volume. That is
useful when objective-space distances are meaningful, but it also means that a
large numeric gap in one objective contributes more volume than a small gap.
For some convergence studies the desired information is instead ordinal: which
reference levels have been reached or passed, independently of the physical
distance between those levels.

Applying ranks independently at every generation is not a valid solution. The
coordinate system would move as the population changes, so an apparent temporal
improvement could be caused by rescaling rather than search progress. Likewise,
feeding ranks to the existing Hypervolume evaluator would still trigger its
ideal/nadir normalization and 1.1 bounding geometry, changing the proposed
metric a second time.

## Decision

MoeaBench introduces **Ordinal Hypervolume (OHV)** as a distinct public metric:

```python
mb.metrics.ordinal_hypervolume(...)
mb.metrics.ohv(...)
```

The alias is the same callable. OHV is not a mode or scale of
`mb.metrics.hypervolume`, whose numerical and reporting behavior remains
unchanged.

For every objective $j$, all final fronts in the selected reference context are
pooled and reduced to sorted distinct values $L_j$. With $K_j=|L_j|$, an
evaluated value is transformed by its left-insertion rank:

$$\rho_j(x)=|\{v\in L_j:v<x\}|.$$

Reference levels occupy $0$ through $K_j-1$; the OHV volume boundary is exactly
one ordinal unit farther at $(K_1,\ldots,K_M)`. Raw ordinary Hypervolume is then
calculated in this integer lattice without ideal/nadir normalization or a 1.1
endpoint. Equal reference values share a coordinate. Values outside the
reference range map to 0 or $K_j$.

The coordinate system is fixed for the entire returned trajectory. `gens`
selects evaluated and returned generations but never limits the final fronts
used to build the ruler. Without `ref`, all final fronts of the evaluated input
provide one self-reference shared across runs. With `ref`, only its final fronts
provide the ruler. A list such as `ref=[exp1, exp2]` therefore establishes one
common lattice for separate algorithm calls; performance views already inject
this list when comparing raw experiments.

OHV reuses the HV engine-selection policy: exact through 8 objectives under
`auto`, Monte Carlo above 8, and Monte Carlo whenever `fast` is requested. Both
new evaluators receive ordinal coordinates directly. The Monte Carlo seed is
reused across generations. Sampling arguments are validated only when sampling
is actually selected.

OHV v1 assumes minimization in every objective. Non-finite reference or selected
evaluated values are errors. A zero-span objective is valid because it forms one
level. Empty evaluated generations dominate zero volume. Relative, absolute,
calibrated, weighted, adaptive-rank, and mixed-direction variants are outside
this decision.

The result is a normal `MetricMatrix`, but `diagnostics['metric_kind']` identifies
OHV explicitly. Its report presents ordinal levels, ordinal reference point,
ordinal box volume, backend, and the diagnostic final `OHV/OBox` fraction. It
must not present conventional nbox, bbox, range-expansion, or `HV/BBox` fields.
The actual sorted levels are retained alongside their counts so the ruler is
scientifically auditable.

## Consequences

- Strictly increasing transformations of each objective leave OHV unchanged
  when applied consistently to evaluated and reference fronts.
- OHV preserves weak Pareto ordering, although distinct values in one reference
  interval can intentionally collapse to a tie.
- Adding or removing intermediate recorded generations cannot alter the ruler.
- Separate OHV values are comparable only when they use the same ordinal
  reference context; multi-experiment performance views provide that context.
- Raw OHV has the concrete interpretation of dominated ordinal unit-cell volume.
- OHV complements rather than replaces HV: distance-sensitive and
  ordering-sensitive convergence can be inspected side by side.
- Relative post-processing was subsequently specified by
  [ADR 0046](0046-relative-ordinal-hypervolume.md) without changing this raw
  ordinal geometry.
