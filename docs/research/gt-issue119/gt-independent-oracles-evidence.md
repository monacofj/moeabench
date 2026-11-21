<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Independent front-oracle finite-probe audit (#119)

**Status:** read-only reproducible research; not a global GT or release
certificate. PR #153 extends the post-fix quantitative work archived
in `gt-postfix-quantitative-evidence.md`.

## Mathematical basis

The probes do **not** call the production `ps()`, `pf()` or private
GT sampling helpers.

- **DTLZ8:** separately parameterized true-Pareto central line
  `f_i=(1-f_M)/4` for `i<M`, `1/3<=f_M<=1`;
  and `M-1` patches with `0<=f_M<1/3`,
  `f_k=u`, `f_j=1-2f_M-u` for `j!=k,M`,
  where `(1-f_M)/4<=u<=(1-2f_M)/2`.
- **DPF3:** independently constructed inverted essential spherical
  objectives `Y=1-h(theta)` with angular exponent 100, followed by
  the published ordered piecewise min/max projection.
- **DPF4:** the same inverted spherical essential construction with
  linear angles, squared after the ordered piecewise projection.
- **DPF5:** independent unit spherical objective directions for P
  and degenerate D, with the P interval `x_1<1/3` and D interval
  `x_1>=1/3`, including the `M-D+1` repeated essential coordinate.

The DPF3/DPF4 logistic projection thresholds are regenerated from
the published recurrence (`x_0=0.1`, `r=3.8`), rather than imported
from the production chaos pool. Additional independent decision-space
probes crosscheck the DPF3–DPF5 objective evaluators against these
equations; this is separate from the GT-nearest-distance experiment.

## Initial executed finite-probe evidence

The first completed workflow, for M=3/5, n=128/512 and three
independent seed offsets, passed **63 positive tests** and
**48 scenario comparisons**. Each scenario reports its components
separately and exports the source PF SHA-256. All 16 named
problem/dimension/component groups showed lower worst-across-seed
finite maximum gap at budget 512 than at 128.

Representative *worst over the three seeds* raw objective-space gaps:

| Problem | M | Component | n=128 | n=512 |
|---|---:|---|---:|---:|
| DTLZ8 | 3 | line | 0.008603 | 0.002089 |
| DTLZ8 | 3 | patch 0 | 0.048752 | 0.024128 |
| DTLZ8 | 5 | patch 0 | 0.080045 | 0.048477 |
| DPF3 | 5 | front | 0.103244 | 0.049298 |
| DPF4 | 5 | front | 0.314782 | 0.091268 |
| DPF5 | 5 | P | 0.448826 | 0.286130 |
| DPF5 | 5 | D | 0.194592 | 0.074074 |

A subsequent expanded workflow includes the special native M=4
DTLZ8 path and high-dimensional M=10. Its final results must be
examined independently; the table above applies only to the
first executed M=3/5 study.

## Scientific limitations

The finite nearest-point distance from independently generated
probes to the production GT is an *empirical distance*, not a
certified continuous Hausdorff bound. Every line/patch/P/D component
is reported separately. No universal weighting is inferred for
the 1-D and 2-D pieces. No test of these deterministic Pareto
probes alone demonstrates stochastic MOEA performance or indicator
rank invariance; GT-HV sensitivity and ADR 0056 release
certification remain distinct.

The dedicated research workflow exports
`gt-independent-geometry.json`. No existing regression test,
production evaluator, accepted ADR, calibration budget, frozen GT
or performance baseline is modified.
