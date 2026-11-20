<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3 published global Pareto set/front characterization — issue #119

**Status: mathematical proof for the originally published DPF3 equations with strictly increasing eta, not validation of the current runtime or its finite-precision reference sampler.** Primary source: Zhen et al., *Multiobjective Test Problems with Degenerate Pareto Fronts*, §IV-C, equations (15)–(16), and §IV-F, equation (21) (https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf). The published eta parameters are the **increasingly sorted** logistic-sequence values; current `BaseDPF` uses sequence order for DPF3, conflicting with that premise when the sequence is not already increasing.

## Proof of global Pareto membership

Assume all objectives minimized, feasible decision box `[0,1]^N`, `M>D>=2`, `N>=D`, strictly ordered constants `0<eta_1<...<eta_(M-D)<1`, and published `g(x)=sum_{i=D}^N(x_i-0.5)^2>=0`.

Let `h(theta)` denote the D-dimensional standard positive-orthant hyperspherical vector (nonnegative, norm exactly 1), with `theta_j=(pi/2)x_j^100` for the first D−1 decisions. Then the **essential objectives** are

```text
gamma(x) = (1+g(x)) [1 - h(theta(x))].
```

The first D−1 published objectives are `gamma_1,...,gamma_(D-1)`. Put `t=gamma_D`. The remaining U+1 objectives, where `U=M-D`, are the ordered saturating projections

```text
p_0(t)   = min(t, eta_1),
p_i(t)   = min(max(t, eta_i), eta_(i+1)),  i=1,...,U-1,
p_U(t)   = max(t, eta_U).
```

**Projection conservation lemma:** for any real t, with strictly ordered thresholds, the projections telescope:

```text
sum_{i=0}^U p_i(t) = t + sum_{j=1}^U eta_j.
```

The identity follows by partitioning the real line into intervals separated by the eta thresholds; in each interval one and only one projection changes linearly with t, all others are constant. Consequently the full projection vector is coordinatewise nondecreasing, and **if all projected components at t_a are <= their counterparts at t_b then t_a<=t_b**, as seen by summing all coordinates. This makes the projection an order-preserving **and order-reflecting** embedding of the scalar essential objective.

**All g=0 decisions are Pareto-optimal:** consider a decision y with g(y)=0, hence `gamma(y)=1-h_y`. Any feasible x allegedly strictly dominating F(y) would have its first D−1 essential coordinates <= gamma(y) and, from the projection conservation lemma, its final essential coordinate also <= gamma_D(y). Thus `gamma(x)<=gamma(y)` componentwise and at least one inequality is strict (the projection cannot introduce strict inequality when both essential vectors are identical). Write `gamma(x)=(1+g_x)(1-h_x)`. Since g_x>=0 and 1-h_x>=0, `1-h_x<=gamma(x)<=1-h_y`, i.e. `h_x>=h_y` componentwise. But both h vectors are nonnegative and have Euclidean norm 1, so componentwise greater-or-equal forces exact equality. Then any strictly smaller gamma would contradict `gamma(x)>=(1-h_x)=(1-h_y)=gamma(y)`. Thus strict domination is impossible. This argument includes arbitrary cross-projection sectors and works for every D, not only D=2.

**Every g>0 decision is dominated by a g=0 witness:** preserve x's angular coordinates and replace all distance coordinates `x_D,...,x_N` with 0.5 to obtain a feasible y. Then `h_y=h_x`, `g_y=0`, and `gamma(y)=gamma(x)/(1+g_x)<=gamma(x)` componentwise, strictly for at least one component because not all coordinates of a unit vector h can equal one. The min/max projections are nondecreasing, so `F(y)<=F(x)`. To guarantee at least one strict objective inequality, note that if one of the first D−1 essential coordinates is strictly positive in y it grows strictly when g>0. Otherwise, `gamma_D(y)>0`, and the projection conservation lemma guarantees that an increase in `gamma_D` produces a strict increase in at least one projected coordinate. Therefore every g>0 decision is strictly dominated.

**Exact global result for the published DPF3:** `PS={x:g(x)=0}`, consisting of all first D−1 decision coordinates free in [0,1], with the remaining coordinates equal to 0.5. The exact objective-space `PF=F(PS)` is the complete continuous image of this PS under the *published sorted-threshold* projection.

## Important qualifications

1. **Current production source differs:** thresholds are not sorted for M−D>=2 where their original logistic order is nonmonotone. The order-reflection identity **does not generally hold** for unsorted thresholds. Current float64 `1−cos(theta)` also collapses mathematically positive small objectives; even with sorted thresholds, numerical ties can cause false sampled domination. Therefore this theorem does **not** certify current `DPF3.pf()` filtering.
2. **Representation and coverage:** `x^100` makes a uniform decision x grid highly nonuniform in objective angles and can produce values below float64 resolution. The theorem establishes *exact membership and complete mathematical parameterization*, not representable cardinality, geometric spacing, extremes or a sufficient calibrated GT. DPF3's finite sampler remains to be independently designed and certified.
3. **Not a production decision:** accepted ADR 0006, 0017, 0018, 0056, 0057 and all historical baseline/test protections remain in force. Any necessary existing-regression change or ADR-incompatible migration must be warned and approved by the maintainer before implementation.

**Independent corroboration:** `scripts/audit/gt_dpf3_global_membership_issue119.py` reconstructs the published sorted projection in memory, verifies the conservation identity, unit essential geometry and feasible same-angle g0 dominance witnesses. `tests/unit/test_gt_dpf3_global_membership_issue119.py` adds four new isolated tests; neither changes existing regression files, production evaluation or frozen GT. Numerical probes are **not** substitutes for the algebraic proof above.
