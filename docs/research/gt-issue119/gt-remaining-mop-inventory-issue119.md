<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Remaining Built-in MOP Ground Truth: Reproducible Audit Matrix (issue #119)

## Context and scientific purpose

A Ground Truth (GT) reference used for IGD, IGD+, or reference-based
hypervolume must represent the **actual global Pareto front** of the MOP,
not merely a convenient decision-space sample that happens to be
internally nondominated. For constrained, disconnected, degenerate or
mixed-dimensional problems, a GT can be feasible and reproducible yet
miss entire relevant regions. The reference budget must be reported as
**requested decision rows, returned PS/PF rows and effectively distinct
objective vectors**.

The mathematical corrections and independent scientific characterization
for DPF3 and DPF5 were completed in
[PR #139](https://github.com/monacofj/moeabench/pull/139) and
[PR #140](https://github.com/monacofj/moeabench/pull/140).
This report records the remaining built-in MOP scope of
[issue #119](https://github.com/monacofj/moeabench/issues/119), as
coordinated by [umbrella #122](https://github.com/monacofj/moeabench/issues/122).

**Scope of the new source audit:** DTLZ1–4, DTLZ7–9, and DPF1/2/4,
at selected M=3/5, requested budgets 128/512, additional intrinsic
D=3 for DPF at M=5, and a DTLZ8 CSV-size stress test at M=3/5/10.

**Independent exclusion:** original DTLZ5/6 full Pareto-front
completeness is owned by [#65](https://github.com/monacofj/moeabench/issues/65).
No approximate `g=0` subset is falsely certified as the complete
published PF by this report. DPF3/5 are already represented by the
completed independent proofs and correction tests.

## Validation protocol

The read-only script
[`scripts/audit/gt_remaining_mop_matrix_issue119.py`](../../../scripts/audit/gt_remaining_mop_matrix_issue119.py)
exercises the **actual** MOP `ps()`, `pf()`, and `evaluation()`
rather than generating a substitute benchmark. Every row records:

- Mathematical configuration M, intrinsic D for DPF, total N, and
  source class (native generation, inherited BaseDPF, or DTLZ8 CSV).
- Requested rows, actual PS and PF rows, exact float64 objective
  duplicates and remaining sample-internally dominated objective rows.
- Repeated determinism, decision bounds, forward evaluation replay and
  approximate generation time.
- Nearest-neighbor mean and coefficient of variation in objective
  space, **not** a global covering or surface-regularity certificate.
- Known necessary mathematical invariants: DTLZ1/DPF1 simplex sum 0.5,
  DTLZ2/3/4 positive unit-hypersphere norm and objective-axis distances,
  DTLZ7/DPF2 record-region occupancy, DTLZ8/9 constraint residuals,
  and DTLZ9 quarter-circle geometry.
- For DPF4, the actual chaotic projection-threshold order is reported
  separately from any claim about the **published** requirement.
  DPF3's previously established sorting rule must **not** be imported
  into DPF4 without checking the DPF4 source paper.
- A deliberate `global_Pareto_membership_proven_by_this_audit=false`
  and `continuous_PF_coverage_certified_by_this_audit=false` on
  every row. This is an audit of **necessary conditions**, not a
  theorem that every generator is complete.

The independent tests
[`test_gt_remaining_mop_matrix_issue119.py`](../../../tests/unit/test_gt_remaining_mop_matrix_issue119.py)
check the named mathematical identities, disconnected components,
deterministic replay and constrained feasibility with a stated 1e−12
diagnostic tolerance, without changing any existing regression test.

The dedicated CI records machine-readable JSON as
`gt119-remaining-mops.json`; reports must cite measurements from that
artifact before declaring numerical failures.

## DTLZ8: independent global Pareto-front characterization (M >= 3)

The DTLZ8 objective-space constraints are, for all earlier objectives
(i<M) and all distinct (i,j<M),

[
t+4f_ige1,qquad
2t+f_i+f_jge1,qquad t=f_Min[0,1].
]

The second family is equivalent to the published
`2*f_M + min_{i<j}(f_i+f_j) - 1 >= 0` constraint, evaluated by
`DTLZ8._calc_constraints`. Write
(alpha(t)=(1-t)/4) and (eta(t)=1-2t).

**When (tge1/3)**, the coordinatewise minimal feasible point
with that last objective is the **central curve**

[
(f_1,ldots,f_{M-1},t)
=(alpha(t),ldots,alpha(t),t).
]

Indeed (2alpha(t)geeta(t)) in this regime.
Every other feasible point at the same (t) is dominated by that
central point.

**When (t<1/3)**, the minimal feasible points form
(M-1) **arms** (with a shared inner endpoint). Choose a distinguished
index (k<M), and any (ain[alpha(t),eta(t)/2]).
The resulting objective point has

[
f_k=a,qquad
f_i=eta(t)-aquad(i
e k, i<M),qquad
f_M=t.
]

This is feasible because each earlier coordinate is at least
(alpha(t)) and every pair sum is at least (eta(t)).
For any feasible row with the same (t), let (a) denote its
smallest earlier coordinate. If (aleeta(t)/2), preserve
that coordinate and reduce all other earlier coordinates to
(eta(t)-a); otherwise reduce all earlier coordinates to
(eta(t)/2). This constructs a **feasible, componentwise
no-worse witness** on the stated arms. Whenever any coordinate is
strictly improved, the original row is **not Pareto-optimal**.

**Global nondominance and completeness:** an arm point has a
pair of earlier coordinates summing to exactly (eta(t)).
A candidate with (t'<t) must have that pair sum at least
(eta(t')>eta(t)), hence cannot dominate the arm point.
At the same (t), decreasing any arm coordinate violates
at least one tight pair constraint. A point with (t'>t)
cannot dominate because its last objective is worse.
For central points with (tge1/3), smaller (t') implies
(alpha(t')>alpha(t)), and same-(t) feasibility fixes their
minimal earlier coordinates. Thus **all stated central/arm points
are globally Pareto-optimal**, and the constructive same-(t)
witness proves that every feasible point outside their union is
dominated.

Since each objective is a block mean of decision variables in
([0,1]), any such feasible objective vector has a valid
decision-space preimage obtained by assigning its objective value
as a constant to all variables in its corresponding block, provided
(N) is divisible by (M) as in the standard DTLZ8 setting.

Consequently, this gives an **exact analytical PF construction for
the published constrained model** at (Mge3), independent of
the incumbent CSV or fallback sampling algorithm. It does **not**
on its own certify **finite coverage, cardinality, float64 equality
constraints or surface-area regularity** of any existing CSV reference.
The source audit tests a constructive domination witness for those
independently generated objective samples.

## Existing established evidence (not new measurements)

The project's earlier, much broader
[`docs/gt-validation-protocol.md`](../../gt-validation-protocol.md)
already studied **DTLZ1–4** in depth using real MOP evaluations and
independent surface references. It compared native sampling with
decision-preimage-validated FPS, endpoint controls, projected Voronoi
relaxation, exact two-objective curves, continuous covering certificates
and higher-dimensional M=10/15 tradeoffs. The evidence demonstrates
substantial gains in several simplex settings, but **not** a universally
superior sampling algorithm. Neither the experimental candidates nor
their incomplete float64 error enclosures are approved automatic GT
defaults.

For **DTLZ7 and DPF2**, the existing helper
`_dtlz7_pareto_regions.py` restricts decision parameters to two
disconnected strict Pareto-record intervals of
`a(x)=x(1+sin(3πx))`. In multiple dimensions there are
`2^(M−1)` DTLZ7 components (or `2^(D−1)` for the corresponding
DPF2 base). Existing tests verify the component cases and the
monotone redundant projection. This establishes sound membership of
the generated parameter domain; **finite conditional coverage within
each component** remains a separate question.

For **DTLZ8**, the current implementation checks for bundled static
CSV decision matrices when M is 3, 5 or 10 and returns the **entire
available file** if the requested point count is at least its row
count. Consequently, a sufficiently large nominal `n_points`
request can return **fewer rows than requested**. For other M, a
guided fallback constructs central-curve and branching-arm samples;
its complete continuous Pareto membership and coverage have **not**
been established by simply checking the inequality constraints.
This audit separately labels source policy and actual returned count.

For **DTLZ9**, the current PS inverse maps a quarter-circle objective
locus to constant decision blocks through the tenth power.
Floating-point feasibility at the equality boundary must be measured
with an explicit tolerance; a strict infeasible flag at approximately
machine epsilon does not establish a real constraint violation.
The prior audit reported 1e−16-scale residuals under strict float
comparisons at M=3 and M=5, with none above 1e−12.

**DPF1** and **DPF4** currently inherit the same BaseDPF
decision-parameter generation, but they have different mathematical
objective maps (simplex-plus-weighted projections versus
squared/min-max projections on inverted hyperspherical objectives).
Using the same code to sample their position parameters **does not**
establish equal objective-space regularity or coverage, particularly
when intrinsic D increases. This audit examines the returned
objective geometry independently and records DPF4 threshold ordering
without assuming the DPF3 publication's rule also holds for DPF4.

## Scientific acceptance matrix and interpretation

Every MOP must be evaluated separately for (i) global Pareto
membership, (ii) completeness of its parametrization, including
all branches and extremes, (iii) finite objective-space coverage,
(iv) regularity and uniqueness at effective budget, (v) numerical
fidelity, (vi) deterministic reproducibility and cost, and
(vii) effects on IGD/IGD+/GT-relative HV against fixed MOEA outputs.

A **passing identity, absence of internal dominance, or all-known
components appearing in one finite sample does not by itself certify
global membership or continuous covering**. Conversely, an observed
effective-cardinality shortfall is a directly reproducible contract
defect, without needing a global Hausdorff theorem.

Before modifying an existing source or baseline, use the
[ADR index](../../adr/README.md), in particular ADRs 0006, 0017,
0018, 0056 and 0057. The documented temporary correction to DPF3/5
does not authorize blanket retuning of other MOPs. Warn and obtain
explicit maintainer permission before any contradiction of an accepted
ADR or change to a historical regression test.

**Research artifact organization:** this is a specialist report,
not a laboratory diary inserted into permanent
[`docs/gt.md`](../../gt.md).
