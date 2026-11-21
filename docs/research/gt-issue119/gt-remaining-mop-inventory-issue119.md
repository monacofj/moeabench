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
request can return **fewer rows than requested**. Independently of
that cardinality defect, the **constructive global-Pareto witness**
above verifies whether the static CSV objectives are **actually
Pareto-optimal**: feasible points that are strictly improved by the
witness are mathematically dominated. For other M, a guided fallback
constructs central-curve and branching-arm samples; this audit
separately labels source policy, actual cardinality and correctness.
The complete mathematical characterization above is independent of
both existing implementations.

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
when intrinsic D increases.

Importantly, the **primary DPF publication** explicitly states that
chaotic threshold values `eta` for **both DPF3 and DPF4** are the
**increasingly sorted** logistic sequence (Zhen et al., *Multiobjective
Test Problems with Degenerate Pareto Fronts*, arXiv:1806.02706v2,
Section IV-F, printed page 7). The audited DPF4 source uses its
unsorted `BaseDPF._chaos_pool` when M-D>=2. An independent published
DPF4 reference therefore constructs the same essential inverse-
hyperspherical objectives, but applies the **sorted** published
min/max projections before the specified squaring. Any observed
difference is an equation error, not merely a subjective preference
about finite GT spacing. [Primary paper](https://arxiv.org/pdf/1806.02706).

## Reproduced source findings: independent CI runs (2026-11-21)

The read-only investigation recorded **49 MOP configurations**:
46 rows from the M=3/5 and n=128/512 matrix (including applicable
DPF D=3 at M=5) plus DTLZ8 n=4096 at M=3,5,10.
The first 33 independent invariant tests passed in
[CI #44891878592](https://github.com/monacofj/moeabench/actions/runs/44891878592).
The independent same-t DTLZ8 dominance checks subsequently passed
37 tests in [CI #44892213828](https://github.com/monacofj/moeabench/actions/runs/44892213828).
The published DPF4 equation comparison passed 38 tests in
[CI #44892537935](https://github.com/monacofj/moeabench/actions/runs/44892537935).

**Effective cardinality.** In the M=3/5, n=128/512 cases, every
audited MOP returned the requested number of numerically distinct
objectives, with no exact duplicate vectors. **DTLZ8 is an exception
at larger budgets** because its static CSVs have only 1,000 rows:

| DTLZ8 objectives | Requested | Returned | Distinct float64 objectives | Constructively dominated |
|---|---:|---:|---:|---:|
| M=3 | 128 | 128 | 128 | **128** |
| M=3 | 512 | 512 | 512 | **512** |
| M=3 | 4096 | 1000 | 1000 | **1000** |
| M=5 | 128 | 128 | 128 | **128** |
| M=5 | 512 | 512 | 512 | **512** |
| M=5 | 4096 | 1000 | 1000 | **1000** |
| M=10 | 4096 | 1000 | 1000 | **1000** |

The strict improvements are **macroscopic**, not float
rounding artifacts. The largest observed improvement in an
individual objective over the dominated CSV rows was approximately
**0.585595 (M=3), 0.856042 (M=5), and 0.747456 (M=10)**.
Each witness preserves the final objective, satisfies the original
DTLZ8 constraints, and is no worse in every other objective.
These data demonstrate a **global Pareto-membership defect in the
CSV references**, even when their *within-file* nondominance and
cardinality checks pass. The former frozen CSV must **not** be
treated as an accepted scientific GT of the published DTLZ8 front.

**DPF4 equation fidelity.** Comparing identical sampled
decisions under the current evaluator and an independently
reconstructed **published sorted-threshold projection** produced:

| DPF4 | Evaluated decisions | Disagreements above 1e-12 | Largest absolute component difference |
|---|---:|---:|---:|
| M=3, D=2 | 128 | 0 | 0 |
| M=5, D=2 | 128 | **128** | **0.509667** |
| M=5, D=2 | 512 | **512** | **0.509667** |
| M=5, D=3 | 128 | 0 | 0 |

The M=3,D=2 instance has only one threshold; the M=5,D=3
instance has two thresholds that happen to be generated in
increasing order, so neither exhibits an ordering discrepancy.
When `eta` is not ordered, **the implemented objectives differ
from the published DPF4 equations**. This is separate from any
question about the quality of the finite PS sampling.

**Other structural results:** the audited DTLZ1/DPF1 simplex,
DTLZ2/3/4 spherical invariants, DTLZ7/DPF2 record-region membership,
DTLZ8/9 feasibility within a 1e-12 diagnostic tolerance, DTLZ9
quarter-circle and repeated-coordinate structure, deterministic
replay and requested/returned counts passed the newly introduced
tests. Their success is a necessary mathematical consistency check,
**not** certification of continuous objective-space coverage.

## Exact-real DTLZ8 finite reference candidate (research only)

With the mathematical central curve and M-1 two-dimensional Pareto
arms established above, complete componentwise parameter grids can
provide an **analytical continuous objective-space covering bound**.
A read-only candidate in
[`scripts/audit/gt_dtlz8_analytic_candidate_issue119.py`](../../../scripts/audit/gt_dtlz8_analytic_candidate_issue119.py)
constructs valid block-constant decision preimages, retains a complete
grid on every component and refills duplicate arm endpoints
deterministically to preserve the user-requested count of distinct
objectives. **It is not installed in `DTLZ8.ps()`.**

Parameterize the central curve by `t in [1/3,1]`, and any arm
by `t in [0,1/3]`, `s in [0,1]`:

[
a(t,s)=rac{1-t}{4}+
       rac{s(1-3t)}4,qquad
b(t,s)=1-2t-a(t,s).
]

The arm has one objective equal to `a`, all other earlier
objectives equal to `b`, and final objective equal to `t`.

For a complete `k_c`-point central grid, the Euclidean
derivative of the center has norm
(sqrt{1+(M-1)/16}), and every continuous central point
is at most

[
R_c le rac{sqrt{1+(M-1)/16}}{3(k_c-1)}
]

from a grid sample. The arm derivatives satisfy

[
sup_{t,s}|partial F/partial t|_2
 = sqrt{(49M-81)/16},quad
sup_{t,s}|partial F/partial s|_2
 = sqrt{M-1}/4 .
]

The first equality follows from convexity of the squared derivative
norm in `s` and comparison at its endpoints for `M>=3`.
A complete `k_t` by `k_s` parameter grid in every arm then
satisfies, by the triangle inequality along parameter coordinate
directions,

[
R_a le
rac{sqrt{(49M-81)/16}}{6(k_t-1)}
+
rac{sqrt{M-1}}{8(k_s-1)}.
]

If **every** component grid is fully retained, the entire published
PF has **exact-real-arithmetic** Hausdorff covering upper bound
(max{R_c,R_a}), in unnormalized Euclidean objective space.
There are nominally
(k_c+(M-1)k_tk_s) grid rows before identifying shared
component boundaries. The candidate searches feasible integer
splits for the lowest analytic bound and replenishes duplicates.
These results **do not certify float64 interval rounding** or
a universal distribution law over mixed intrinsic dimensions.

## Reproduced analytical DTLZ8 candidate results

The independent [candidate CI workflow](https://github.com/monacofj/moeabench/actions/runs/44892943727)
constructed complete component grids and evaluated their block-constant
preimages through the **unchanged actual DTLZ8 evaluator**.
The DTLZ8 research candidate generated the exact number of
numerically distinct requested objectives for **M=3/5/10 and
n=128/512/4096**. No DTLZ8 production code was modified.

| M | Requested = returned = unique | Ideal real-arithmetic continuous cover upper | Largest gap among independent finite probes |
|---|---:|---:|---:|
| 3 | 128 | 0.072967 | 0.050505 |
| 3 | 512 | 0.033537 | 0.020639 |
| 3 | 4096 | **0.011138** | 0.010099 |
| 5 | 128 | 0.169219 | 0.083167 |
| 5 | 512 | 0.072296 | 0.053237 |
| 5 | 4096 | **0.023762** | 0.012068 |
| 10 | 128 | 0.468385 | 0.200989 |
| 10 | 512 | 0.178016 | 0.087376 |
| 10 | 4096 | **0.055843** | 0.027418 |

The greatest objective reconstruction discrepancy observed in these
cases is **2.23e-16**, attributable to float64 block averaging.
The geometric upper bounds are **derived for exact real arithmetic**;
the finite-probe maxima do not prove them and the numerical
implementation does not have a rigorous directed IEEE interval
enclosure. The candidate is a **technically tested replacement design**
that can be reviewed before changing a production MOP or canonical
reference under the accepted ADRs.

## Architectural contradiction and explicit approval gate

The currently **accepted ADR 0016** describes DTLZ8 as migrated to
a *guided analytical manifold reconstruction* and expressly
contrasts it with frozen heuristic reference data. The also
**accepted ADR 0017**, written subsequently, describes
*NSGA-III-based frozen high-fidelity numerical reference CSVs* for
DTLZ8/9. The current DTLZ8 implementation prioritizes bundled
static CSV references for M=3/5/10. The two historical
architectural descriptions are inconsistent, and now the static
DTLZ8 references have been **independently shown to be dominated**.

**A production change replacing the frozen DTLZ8 data with the
proved analytical generator requires an explicit maintainer
decision reconciling ADRs 0016 and 0017**, including the scientific
status of the previous CSVs and any historical regression impact.
The DPF4 correction also requires review of accepted historical
DTLZ/DPF GT calibration and performance reports (e.g. ADR 0018)
before changing the objective evaluator, sample generator,
or references.

This PR performs **only research and documentation**, with new
independent tests. It neither supersedes an accepted ADR nor
changes production, canonical GT, old regression fixtures,
calibration sidecars, or the default `calibrate()` budget.

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
