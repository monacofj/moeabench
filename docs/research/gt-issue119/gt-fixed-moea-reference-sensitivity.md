<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Fixed-MOEA-Front GT Reference Sensitivity (#119)

**Status:** research experiment, not ADR 0056 performance certification.
**Code:** `scripts/audit/gt_fixed_moea_sensitivity_issue119.py`.
**Data artifact:** `gt-fixed-moea-sensitivity.json` from the dedicated
GitHub Actions workflow when completed.

## Question

Holding the final objective populations of real stochastic MOEA
executions completely fixed, what changes when only the finite
ground-truth reference is replaced by a different resolution?
Can such a reference change reverse the observed ordering of
two algorithms under the public IGD, IGD+ or HV metrics?

This explicitly differs from comparing multiple stochastic
solver executions or different software versions. No fresh
run is performed when a reference is switched.

## Frozen experimental design

- Problems: DTLZ2, DTLZ7, DTLZ8, DPF5, all with M=3
  (DPF5 uses D=2).
- Algorithms: the project's NSGA2 and SPEA2 adapters.
- Independent integer run seeds: 11941 and 11942; each
  problem/algorithm/seed triple is executed **once**.
- Search budget: 32 population members for 12 generations.
- Each observed final non-dominated objective matrix is retained
  unchanged across **both** references, n=128 and n=512.
- Reference generation: `mop.pf(n_points=n)`; objective and
  reference-array SHA-256 digests, seed, generation count and
  cardinality are captured.
- Indicator semantics: `mb.metrics.igd`,
  `mb.metrics.igdplus`, and exact `mb.metrics.hv`
  `scale="raw"` with an explicit external reference for its
  normalization bounds. Separate raw, unnormalized IGD and
  positive-part IGD+ are diagnostic, not substitutes for the
  published public interface.

A changed HV value under `scale="raw"` may reflect a changed
reference normalization box even though the evaluated MOEA front
is identical; `scale="abs"` would require separately calibrated
GT and is **not** examined in this research. We never silently
generate or persist a calibrated reference.

For each problem and run seed, compare NSGA2 against SPEA2
*within each metric*, recording the algorithm-score difference
for each reference resolution. A strict sign change between
the two scores is an **observed numerical ordering reversal**,
not a statistically significant performance rank reversal.
No win/loss or global superiority inference is made from
two seeds or from a 12-generation pilot.

## Acceptance and limitations

The dedicated workflow requires 16 fixed final fronts, 24
metric/seed/problem pairwise comparisons, complete finite
scores and valid front/reference SHA-256 identities.
All inputs and outcomes are archived in a JSON artifact.
Positive unit tests exercise the public indicators and
one real, small NSGA2 run. Full unit and numerical stability
checks also run on the PR head.

This experiment does **not** authorize modifying any old GT,
regression baseline, performance baseline or numerical certificate.
It cannot certify the 64-seed/ADR-0056 release performance design,
continuous-front covering, or a canonical mixed-dimensional
reference measure.

**Execution results must be appended after successful final-head
CI.** The existence of this document or workflow alone is
not a completed scientific check.
