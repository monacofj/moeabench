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

## Executed pilot results (2026-11-21)

The dedicated workflow succeeded on commit `ff16bb6`, after fixing
a *new synthetic test fixture* whose HV external reference had
zero extent along one objective. The public HV contract was
preserved; the test reference now spans every objective.

All 16 MOEA runs completed, and all **24** algorithm/seed/metric
pairwise comparisons were evaluated under both GT resolutions.
There were **zero strict sign reversals** in this small pilot.
For all eight problem/seed pairs, the reported NSGA-II minus SPEA2
IGD and IGD+ differences changed numerically when the reference
budget increased from 128 to 512, even though the same frozen
algorithm fronts were evaluated. Examples:

| Problem | Seed | Metric | Difference at 128 | Difference at 512 |
| --- | --- | --- | ---: | ---: |
| DTLZ2 | 11941 | public IGD+ | 0.076640 | 0.093021 |
| DTLZ7 | 11941 | public IGD+ | -0.125893 | -0.116249 |
| DTLZ8 | 11942 | public IGD | 0.012135 | 0.009009 |
| DPF5 | 11942 | public IGD+ | 0.018595 | 0.020807 |

For this selection, HV *pairwise differences* were unchanged
at both resolutions, because the evaluated fronts were frozen and
the external reference extrema of each tested finite GT
coincided. In DTLZ7 both evaluated HVs were zero in these short
runs, so a zero pairwise difference cannot establish that HV
distinguishes the algorithms reliably.

The small budgets (32 individuals, 12 generations, two seeds),
three-objective cases and absence of older-finite-reference
comparison limit the scope. The findings do **not** constitute
statistical performance stability or ADR 0056 release
certification. Extending seeds, evaluation budgets and dimensions
requires a separate plan; no automatic reference migration is
authorized.

The complete JSON artifact and job logs, including objective
hashes and every individual indicator value, remain attached to
the dedicated GitHub Actions run.
