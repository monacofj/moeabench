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
