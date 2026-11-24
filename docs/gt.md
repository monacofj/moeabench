<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Ground Truth in MoeaBench: Mathematical Foundations, Reference Generation, and Quality Assurance

This document specifies how **Ground Truth (GT)** is defined, constructed, calibrated, persisted, validated, and used by MoeaBench indicators. It distinguishes mathematical Pareto truth, finite computational representations, and certified metric reference volumes. It starts from multiobjective optimization fundamentals; explains the DPF3/DPF5 geometric audits; and states the implemented scalar absolute-Hypervolume reference contract, with research evidence kept separately under `docs/research/`. **The accepted ADRs are normative when a decision conflicts with historical investigation text.**

The DPF3/DPF5 findings were established by independent mathematical arguments and reproducible experiments in [issue #119](https://github.com/monacofj/moeabench/issues/119) and [PR #139](https://github.com/monacofj/moeabench/pull/139). **The proofs concern the originally published mathematical models.** A theorem about a model does not imply that every software implementation reproduces its equations correctly. Implementation defects below refer to the audited version and must not be mistaken for permanent mathematical properties of those benchmark problems.

## 1. Why Ground Truth Matters

MoeaBench organizes experiments involving **Multiobjective Optimization Problems (MOPs)** and **Multiobjective Evolutionary Algorithms (MOEAs)**. A single-objective problem seeks to optimize one function. In a multiobjective problem, several objectives must be optimized simultaneously, even though their minima may be mutually incompatible.

Consider minimizing the cost, energy consumption, and completion time of a design. A cheap design may be slow; a fast one may consume more energy. Many feasible designs are available, but a **Pareto-optimal** solution is one for which no other feasible solution improves at least one objective without worsening another.

Evaluating an MOEA by the number of returned solutions is insufficient. We also need to know whether its results approach optimality, whether they cover the meaningful trade-offs, whether extremes are represented, and whether two algorithms are compared against the same scientific reference. A cloud of apparently regular points may still cover only a narrow part of the actual optimal front.

A **Ground Truth (GT)** is the scientific reference used for these comparisons. When a problem admits a mathematically known optimal set, GT should derive from that mathematical model, rather than being declared correct merely because a solver was run for many generations.

Three distinct concepts must be kept separate:

- **Mathematical truth:** the exact Pareto set (PS) and Pareto front (PF), defined over all feasible decisions.
- **Computational representation:** a finite sample intended to represent the continuous PF with documented quality.
- **Reference artifact:** saved, versioned data used by an experiment, metric, or certification, with traceable provenance.

A small sample can contain only globally optimal points yet be inadequate for coverage assessment. A dense sample can follow the wrong geometry. A historically reproducible reference can contain values that were later shown to be mathematically incorrect. These cases require different interpretations.

## 2. Mathematical Definitions and Terminology

### 2.1. Multiobjective functions and dominance

Let \(N\) be the number of decision variables and \(M\) the number of objectives. A multiobjective evaluation is

\[
F:X\subseteq\mathbb R^N\longrightarrow\mathbb R^M,\qquad
F(x)=(f_1(x),f_2(x),\ldots,f_M(x)),
\]

where \(X\) denotes the feasible decisions. The DPF problems discussed here minimize every objective.

A decision \(a\) **dominates** \(b\) if

\[
f_i(a)\le f_i(b)\quad\text{for every }i,
\qquad
f_j(a)<f_j(b)\quad\text{for at least one }j.
\]

The word *nondominated* must always identify the **domain of comparison**. If no two points in a sample of 100 dominate one another, the points are nondominated **within that sample**. A feasible point outside the sample may still dominate some of them. Thus, applying nondominated filtering to a finite candidate set does **not** prove global Pareto optimality.

### 2.2. Pareto set and Pareto front

The **Pareto set (PS)** consists of feasible decisions not dominated by any other feasible decision; the **Pareto front (PF)** is its image in objective space:

\[
PS=\{x\in X:\nexists y\in X\text{ that dominates }x\},
\qquad PF=F(PS).
\]

PS lives in \(N\)-dimensional decision space; PF lives in \(M\)-dimensional objective space. Different decisions may map to the **same** objective vector, particularly under degeneracy or redundancy.

Within the DPF family, \(D\) identifies the number of **essential objectives** in the formulation; additional \(M-D\) objectives can be redundant projections. Configuration parameter \(K\) also influences the number of variables. \(D\) is not the total number of decision variables and must not be confused with \(N\).

### 2.3. Intrinsic dimension, degeneracy, and components

A front with \(M\) objective coordinates need not have intrinsic dimension \(M-1\). A **degenerate** PF lies on a lower-dimensional manifold. A **mixed-dimensional** PF can comprise components of different intrinsic dimensions, such as a surface and a curve.

Consequently, assigning equal numbers of samples to a curve and a surface does not produce equal geometric coverage. Likewise, uniform decision-space sampling is not generally uniform in **objective space**. The evaluation map can distort distances, collapse coordinates, and create duplicate objective vectors.

### 2.4. Sampling, effective cardinality, and coverage

A finite reference requires several independent counts:

1. **Requested points:** the nominal input, often called \`n_points\`.
2. **Generated decisions:** the actual number of rows in the returned PS sample.
3. **Returned objectives:** the PF sample rows after evaluation and any subsequent filtering.
4. **Distinct numerical objectives:** the number of different objective vectors under a stated numerical equality rule and precision.
5. **Mathematically distinct objectives:** a stronger notion that cannot be established merely through exact comparison of floating-point arrays.

A directed geometric coverage error between a continuous PF \(P\) and a reference sample \(R\) is

\[
h(P,R)=\sup_{p\in P}\min_{r\in R}\|p-r\|_2.
\]

The **Hausdorff distance** is \(d_H(P,R)=\max\{h(P,R),h(R,P)\}\). If every point of \(R\) lies exactly on \(P\), then \(h(R,P)=0\). Proving a bound on \(h(P,R)\) is substantially stronger than measuring nearest-neighbor distances at finitely many probes. A maximum over finite random probes is **not** a certified bound on the entire continuous front.

Coverage must also account for **extreme points**, interfaces between components, disconnected regions, density, and spacing, rather than relying only on a mean or median distance.

### 2.5. Mathematical, numerical, and statistical fidelity

A scientifically defensible GT requires several independent properties:

- **Equation correctness:** implementation follows the published mathematical model for all supported dimensions, boundaries, and branches.
- **Global membership:** reference points are globally Pareto-optimal, not just nondominated within a candidate set.
- **Mathematical completeness:** the parameterization reaches every relevant part of the PF.
- **Finite coverage:** point number and distribution adequately represent the continuous PF.
- **Numerical fidelity:** the chosen arithmetic can represent the points, including inverse transformations and rounding.
- **Computational cost:** time and memory are appropriate for the purpose of reference generation.
- **Reproducibility and provenance:** configuration, seeds, mathematical version, source revision, numerical environment, and reference identity are recorded.

These requirements are not substitutes for each other. A spherical identity is a necessary mathematical condition for some problems, but by itself does not prove global Pareto membership. Conversely, a proof of the continuous PF does not automatically guarantee an adequate finite sample.

## 3. MOP Ownership and API Architecture

### 3.1. Reference-generation strategy belongs to each MOP

**Each MOP and configuration must own its scientifically justified GT-generation strategy.** The problem knows its objective functions, constraints, intrinsic dimensions, geometry, extremes, and disconnected or mixed-dimensional components. A general-purpose sampler must not be imposed on different MOPs merely to reduce implementation effort.

This **does not forbid shared infrastructure**. Vectorized evaluation, numerical kernels, deterministic random generation, validation, serialization, seed allocation, and geometrical operations that are mathematically equivalent may be shared. The criterion for reuse is **demonstrated scientific equivalence**, not code reuse as an end in itself. A shared hyperspherical kernel can serve distinct problems with genuinely matching geometry; it must not replace, without a separate proof, a power-100 angular parameterization or a PF containing components of different dimensions.

This division of responsibility follows [ADR 0006](adr/0006-benchmarks-reimplementation.md) (native analytical PS/PF at MOP level) and the architecture accepted in principle by [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md).

### 3.2. What the public operations mean

**\`evaluation(X)\`** maps a matrix of decision vectors to objective vectors. An error here changes the **optimization problem itself** and can influence every result obtained by an MOEA.

**\`ps(n_points)\`** produces decision-space samples intended to lie on PS. A requested count does not automatically imply an equal number of unique objective vectors.

**\`pf(n_points)\`** produces an objective-space reference. In **\`BaseMop\`**, the default implementation evaluates **\`ps()\`**. It therefore inherits sampling defects and mathematical evaluation defects. Individual MOPs can override \`pf()\`; the audited DPF3 implementation additionally filtered by nondominance, reducing its effective cardinality and breaking any assumption that row \(i\) of PS always corresponds to row \(i\) of the returned PF.

**\`calibrate()\`** is the **explicit** operation for obtaining and persisting a GT. Ground-truth calibration predates and is independent of the discontinued FAIR metric research; it remains part of MoeaBench's scientific API.

**\`calibrated_gt\`** exposes the instance's calibrated objective reference, returning a defensive copy.

The audited **\`moeabench/mops/calibration.py\`** implementation can load an existing sidecar, accept an external **\`source_gt\`**, use empirical **\`source_search\`**, or generate a reference through **\`pf(n_points)\`**. Saved data includes a **\`gt_reference\`** payload and basic metadata in JSON schema **\`mop_gt_v1\`**. Explicit **\`force\`** controls replacement of a persisted artifact.

### 3.3. Visualization, evaluation, and calibration have different purposes

A lightweight PF used in interactive visualization need not have the same density as a scientific reference used by metrics. Display-oriented reduction may preserve important extrema and components without becoming the metric's GT. Ordinary **\`ps()\`** or **\`pf()\`** calls must not implicitly create persistent files or disk caches.

Historical calibration defaults use **2,000 points for DTLZ** and **10,000 for DPF** when no count is supplied. Those are historical operational defaults (ADR 0018), **not mathematical sufficiency theorems**. [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md) accepts in principle that the calibration budget should depend automatically on MOP geometry, target accuracy, and bounded cost, without yet approving a universal numerical rule. The GT's quality and cost must be characterized for the individual MOP.

Preservation of the **exact GT actually used** when exporting an experiment falls under the experiment's persistence contract. Export must not silently generate a new reference.

### 3.4. Calibration provenance and applicability to indicator references

A GT sidecar using schema `mop_gt_v1` stores the problem identity
(`problem_id`), objective count (`mop_dimension`), `gt_reference`,
timestamp and `gt_source`. Current source kinds are `analytical`
(generated by the MOP's own `pf()`), `external` (supplied using
`source_gt`) and `empirical_search` (obtained from an optimization
search). Loading a sidecar restores the recorded source provenance
without recomputing the GT; absent or unrecognized provenance is
**not** proof of analytical sampling. An instance exposes the calibrated
objective matrix through `mop.calibrated_gt`.

A label of `analytical` alone does **not** certify that a sampled GT
is exhaustive or that its Hypervolume equals the continuous optimum.
It identifies its generation route; separate mathematical evidence is
required for any stronger claim. Similarly, finite reference cardinality,
deterministic replay, and a small change after refinement are
quality diagnostics, not a proof of complete coverage.

The default density remains 2,000 points for DTLZ and 10,000 for
DPF when `calibrate()` is called without an explicit point budget.
[ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md)
accepts the direction of geometry-aware, purpose-specific calibration
but does not yet authorize an automatic replacement numerical budget.
Neither `ps()` nor `pf()` implicitly persists a sidecar. This contract
is independent of deferred FAIR research.

### 3.5. Status of further MOP-specific GT geometry

The principles in Sections 3.1–3.4 apply beyond the detailed DPF3/DPF5
case studies below. The current original **DTLZ5 and DTLZ6**
implementations use `moeabench/mops/_dtlz56_front.py` to sample
the full-front geometry for `M>=4`, including Pareto-optimal regions
with `g>0`, instead of incorrectly assuming that `g=0` alone
always describes the complete original front. For `M<=3`, they
retain their analytically supported historical degenerate
sampling strategy. These are **MOP-specific PF/PS facts** and
do not certify the continuous HV denominator; DTLZ5/6 must not
inherit DTLZ2's spherical exact-volume formula merely because
they share spherical expressions in parts of the evaluator.

For other special geometries, including DPF4's own ordered
projection thresholds, DPF5's distinct P/D components and
DTLZ8's constrained line-and-patch front, the mathematical
definition and configuration-specific coverage requirements
take precedence over a generic sampler. Any unverified
sampler candidate remains a **research candidate**, not an
automatic certified GT, until its component-wise/global
membership, constraints, effective cardinality and numerical
quality are independently established. See
[ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md)
and [issue #119](https://github.com/monacofj/moeabench/issues/119)
for the adopted scope and outstanding requirements.

## 3.6. Original DTLZ5/DTLZ6: Mathematical Ground Truth and Global Dominance Proofs

This section is the **current technical reference** for the original
DTLZ5 and DTLZ6 Pareto geometry and its independent audit. It consolidates
the mathematical results investigated in [issue #65](https://github.com/monacofj/moeabench/issues/65)
and the incremental verification work in
[issue #175](https://github.com/monacofj/moeabench/issues/175) and
[superissue #176](https://github.com/monacofj/moeabench/issues/176).
Experimental chronology and exploratory counterexamples remain in
[the issue #65 research record](research/dtlz5-dtlz6-original-front-issue65.md).
**All proofs below concern the original mathematical problems over real
numbers.** The software's floating-point checks have narrower claims.

### 3.6.1. Original feasible objective image (exact parameterization)

Let \(M\ge 2\), \(K=N-M+1\ge 1\), \(x\in[0,1]^N\), and minimize all
\(M\) objectives. The original distance functions are

\[
g_5(x)=\sum_{j=M}^{N}(x_j-\tfrac12)^2,\qquad
g_6(x)=\sum_{j=M}^{N}x_j^{0.1}.
\]

Consequently their **exact attainable ranges** are

\[
g\in[0,g_{\max}],\qquad
g_{\max}=
\begin{cases}
K/4 &\text{(DTLZ5)},\\
K   &\text{(DTLZ6)}.
\end{cases}
\]

Every value in its respective interval is attainable: set each
distance variable to \(1/2+\sqrt{g/K}\) for DTLZ5, or
\((g/K)^{10}\) for DTLZ6. For the position variables, set

\[
\theta_1=\frac{\pi x_1}{2},\qquad
\theta_i=\frac{\pi(1+2gx_i)}{4(1+g)}
\quad (2\le i\le M-1).
\]

Writing \(H_M(\theta)\) for the standard **nonnegative unit**
hyperspherical vector, the objectives are

\[
F(x)=(1+g)H_M(\theta),\quad
F(x)\ge 0,\quad
\|F(x)\|_2=1+g.
\tag{DTLZ56-1}
\]

For each fixed \(g\), the exact independently attainable angular
intervals are

\[
\theta_1\in[0,\pi/2],\qquad
\theta_i\in[a(g),b(g)]\ (i\ge2),
\]

\[
a(g)=\frac{\pi}{4(1+g)},\qquad
b(g)=\frac{\pi(1+2g)}{4(1+g)}
=\frac{\pi}{2}-a(g).
\tag{DTLZ56-2}
\]

At \(g=0\), the inner angles \(\theta_i\) for \(i\ge2\) are all
\(\pi/4\), leaving a degenerate **one-dimensional curve** controlled
by \(\theta_1\). For \(M\ge4\), that curve is **not** the complete
Pareto front of the original problems: globally optimal regions
with positive \(g\) also exist. All statements and software checks
below preserve that distinction. The feasible image parameterization
itself is **not** a characterization of which feasible points are Pareto-optimal.

### 3.6.2. Theorem 1 — least feasible radial distance in each direction

For any feasible nonnegative objective vector \(y\ne0\), consider
**a feasible angular representation** of its objective direction,
with inner angles \(\theta_2,\ldots,\theta_{M-1}\). Such a
representation is unique away from hyperspherical coordinate
singularities; when leading coordinate groups vanish, multiple
angle tuples can represent the same direction.
Define

\[
q(\theta)=\max_{2\le i\le M-1}
\left|\frac{4\theta_i}{\pi}-1\right|,
\qquad
g_{\min}(\theta)=\frac{q(\theta)}{1-q(\theta)},
\tag{DTLZ56-3}
\]

with \(q=0\) when \(M=2\). Feasible points necessarily have \(q<1\).
The angular constraints (DTLZ56-2) are equivalent to

\[
\left|\frac{4\theta_i}{\pi}-1\right|
\le \frac{g}{1+g},\qquad i=2,\ldots,M-1.
\]

**Theorem.** For a **specified feasible angular tuple**
\(\theta\), \(g_{\min}(\theta)\) is the smallest attainable
\(g\) realizing that tuple. A feasible point with
\(g>g_{\min}(\theta)\) is strictly Pareto-dominated by another
feasible point on **the same objective-space ray**. At a
hyperspherical coordinate singularity, the least feasible \(g\)
of the ray itself may be even smaller, because the same ray can
admit several angular tuples.

**Proof.** Taking the maximum of the angular inequalities yields
\(q\le g/(1+g)\), equivalent to \(g\ge q/(1-q)\). At
\(g=g_{\min}\), each angular value is still inside its admissible
interval; the independent position variables can realize those
angles and the distance variables can realize \(g_{\min}\), which
also lies below the original feasible candidate's \(g\le g_{\max}\).
With identical angles, the vectors differ only in the common
scale \(1+g\). Reducing \(g\) decreases every positive objective
without increasing any other objective, and at least one coordinate
strictly decreases because the vector has positive norm. Hence the
larger-\(g\) point is dominated. \(\square\)

**Corollary.** Every globally Pareto-optimal vector must satisfy
\(g=g_{\min}(\theta)\) for **every** feasible angular realization
of that vector. If \(g>0\), at least one inner angle in each
such realization lies at an endpoint of (DTLZ56-2), equivalently
at least one position variable \(x_i\), \(2\le i\le M-1\), equals
zero or one. A degenerate ray that also admits a \(g=0\)
realization cannot contain a Pareto-optimal \(g>0\) point. **This is a necessary condition, not a sufficient
global-optimality proof**: different boundary directions may
dominate one another.

**Coordinate-singularity refinement.** The inverse angular
representation is nonunique whenever an objective and all
preceding coordinates in its hyperspherical subvector
vanish simultaneously. In that case `atan2(0, 0)`
is a library convention, **not** a mathematically
determined angle. Since the original angular intervals
always contain `pi/4`, an undetermined inner angle
may be chosen as `pi/4` in a feasible representation;
its contribution to the minimum required radial
`g` is therefore zero. This choice is independent
for every undetermined angle and realizes the
minimum feasible `g` over representations of
the same objective ray.

For example, the original endpoint
`(0, ..., 0, 1)` is feasible and Pareto-optimal
at `g=0` (take `theta_1=pi/2`); forcing its
indeterminate inner angles to the numerical
value `atan2(0, 0)=0` incorrectly rejects it.
The independent inverse-geometry auditor
explicitly handles only **exactly zero**
numerator-and-subradius pairs in this way.
Near-zero but determined angles still require
the ordinary mathematical feasibility checks;
this is not a blanket numerical tolerance
that makes an impossible direction feasible.
A positive-`g` vector on the same last-axis
ray remains provably radially dominated by
its unit-radius representative.

The independent implementation
[`_dtlz56_independent_audit.py`](../moeabench/mops/_dtlz56_independent_audit.py)
uses objective norms, inverse angles and (DTLZ56-3). It detects
infeasible angular/radial images and points with provable same-ray
dominance. Numerical residual checks are subject to the declared
tolerance; the algebraic theorem does not inherit their rounding error.

### 3.6.3. Theorem 2 — globally Pareto-optimal unit-radius points

**Theorem.** Every **feasible** objective vector satisfying \(g=0\)
is globally Pareto-optimal.

**Proof.** By (DTLZ56-1), every feasible objective vector \(z\)
has norm \(\|z\|_2=1+g(z)\ge1\). Assume that a feasible \(z\)
dominates a feasible \(y\) with \(g(y)=0\). Since all coordinates
are nonnegative, \(0\le z_i\le y_i\) for every \(i\), with at least
one strict inequality, implies
\(\sum_i z_i^2<\sum_i y_i^2=1\). This contradicts
\(\|z\|_2\ge1\). Therefore no feasible dominator exists.
\(\square\)

**Scope:** this establishes membership of **individual**
\(g=0\) points. It does **not** certify an entire finite GT
consisting only of such points when the original PF contains
positive-\(g\) regions. In particular, the original \(M\ge4\)
full-front assessment is not automatically `pass` merely
because these points are valid.

### 3.6.4. Theorem 3 — exact global Pareto criterion using only scalar g

Let \(y\) be any feasible objective vector and let
\(g_y=\|y\|_2-1\). Define the feasible objective image at radius
parameter \(g\) by

\[
\mathcal F(g)=\{(1+g)H_M(\theta):
\theta_1\in[0,\pi/2],\
\theta_i\in[a(g),b(g)]\ (i\ge2)\}.
\]

**Theorem (necessary and sufficient).**

\[
\boxed{
y\in PF
\iff
\nexists\,g\in[0,g_y),\
z\in\mathcal F(g)\ \text{with }z_i\le y_i
\text{ for all }i.
}
\tag{DTLZ56-4}
\]

**Proof, forward direction.** If such \(g\) and \(z\) exist,
then \(g<g_y\) gives
\(\|z\|_2=1+g<1+g_y=\|y\|_2\). Therefore \(z\ne y\).
Together with \(z\le y\), this implies strict Pareto
dominance of \(y\), contradicting \(y\in PF\).

**Proof, reverse direction.** If \(y\notin PF\), there exists
a feasible dominator \(z\le y\) with at least one strict
coordinate inequality. Nonnegativity implies
\(\|z\|_2<\|y\|_2\). By (DTLZ56-1),
\(g(z)=\|z\|_2-1<g_y\), establishing precisely the point
excluded by the right-hand side. \(\square\)

This theorem is **exact for every supported \(M,N\)** under
the original DTLZ5/6 equations. It reduces a global comparison
over decision vectors to **one continuous scalar \(g\) plus an
exact fixed-\(g\) feasibility predicate**. It neither presumes
convexity nor mistakes the minimal-radius necessary condition
for a proof of Pareto efficiency.

A particularly important consequence is the **open endpoint**
\(g<g_y\). Inspecting only \(g\le g_y-\delta\) for any
\(\delta>0\) leaves the interval \((g_y-\delta,g_y)\)
unresolved. An arbitrarily small improvement may still
dominate the candidate. Conversely, for \(g_y=0\) the search
domain is empty, recovering Theorem 2.

### 3.6.5. Theorem 4 — exact feasibility predicate at a fixed g

Fix a feasible \(g\) and a nonnegative objective-space cap
\(y=(y_1,\ldots,y_M)\). The predicate

\[
D(g;y)\ \Longleftrightarrow\
\exists\,z\in\mathcal F(g): z_i\le y_i\quad\forall i
\tag{DTLZ56-5}
\]

has a deterministic **\(O(M)\)** decision procedure in exact
real arithmetic, without discretizing any angle.

**Construction.** Start with residual hyperspherical radius
\(R=1+g\). The last objective is \(z_M=R\sin\theta_1\), where
\(\theta_1\in[0,\pi/2]\). Allocate the largest permissible
value \(z_M=\min(y_M,R)\); the remaining radius becomes
\(R\leftarrow\sqrt{R^2-z_M^2}\). For the subsequent objectives
\(j=M-1,M-2,\ldots,2\), each relevant angle lies in
\([a(g),b(g)]\). At current radius \(R\), feasibility
requires

\[
R\sin a(g)\le y_j.
\tag{DTLZ56-6}
\]

If (DTLZ56-6) fails, return infeasible. Otherwise allocate

\[
z_j=\min\big(y_j,R\sin b(g)\big),\qquad
R\leftarrow\sqrt{R^2-z_j^2}.
\tag{DTLZ56-7}
\]

After processing those coordinates, return feasible exactly when
\(R\le y_1\).

**Why greediness is exact.** At each step every allowed choice
is a nonnegative contribution bounded by both its objective cap
and the angular interval. The largest contribution leaves the
smallest nonnegative residual radius. For every later coordinate,
a smaller residual weakly decreases its minimum permissible
contribution and does not make the remaining objective caps
harder to satisfy. Thus, if the greedy choice fails to satisfy
a remaining cap, any smaller preceding allocation would fail
as well. Conversely, each allocation is realizable by an
allowed angle, so a successful recursion constructs a feasible
vector in \(\mathcal F(g)\). \(\square\)

For \(M=2\), there are no constrained inner angles and the
intermediate loop is empty. This predicate is independently
represented in
[`scripts/audit/dtlz56_fixed_g_dominance_oracle.py`](../scripts/audit/dtlz56_fixed_g_dominance_oracle.py).
Floating-point implementations require slack near angle/radius
boundaries; passing finite tests is not equivalent to directed-rounded
certification of (DTLZ56-5).

### 3.6.5a. Corollary — squared-energy fixed-g feasibility recurrence

The fixed-\(g\) predicate of Theorem 4 admits an **equivalent
exact-real recurrence without intermediate square roots**. This
reformulation is useful when constructing independently verified
numerical enclosures, but does not itself certify floating-point
computations.

Put \(r=1+g\), \(a=\pi/(4r)\), and
\(\beta(g)=\sin^2(a)\). The upper constrained angular limit
is \(b=\pi/2-a\), so \(\cos^2(b)=\beta\).
If \(y_M\ge r\), choose \(\theta_1=\pi/2\);
the last-axis point fits every nonnegative cap.

Otherwise, after greedily allocating \(z_M=y_M\), define
the **squared residual** \(q=r^2-y_M^2>0\).
Process \(j=M-1,M-2,\ldots,2\). The necessary
minimum contribution \(z_j=\sqrt q\sin a\)
fits the cap precisely when

\[
q\beta(g)\le y_j^2.
\tag{DTLZ56-17}
\]

If it does not fit, the candidate caps are infeasible at
this \(g\). If it fits, maximize the feasible contribution:
\(z_j=\min(y_j,\sqrt q\sin b)\). The squared residual
after this allocation is exactly

\[
q_{\mathrm{new}}
=q-\min(y_j^2,q\sin^2 b)
=\max\bigl(q-y_j^2,\ q\beta(g)\bigr).
\tag{DTLZ56-18}
\]

After the inner objectives have been processed,
the remaining first objective fits if and only if
\(q\le y_1^2\). For \(M=2\), the inner loop is
empty.

**Proof.** The minimum and maximum angle bounds,
monotonicity of sine on \([0,\pi/2]\), and
Theorem 4's optimal greedy allocation establish
(DTLZ56-17). Squaring the nonnegative admissible
contributions and applying the Pythagorean residual
identity establishes (DTLZ56-18). Each step is
equivalent to Theorem 4's exact-real construction,
so the final first-coordinate condition is also
necessary and sufficient. \(\square\)

This exposes a fixed-\(g\) decision predicate built
from nonnegative arithmetic, comparisons, one
\(\sin^2\) expression and \(\max\). In particular,
the original repeated floating-point square-root
recursion is **not necessary** for the fixed-\(g\)
decision. A future validated interval evaluator can
enclose the sine term and all squared quantities,
while explicitly marking uncertain comparisons.

The independent research implementation
[`scripts/audit/dtlz56_energy_recurrence.py`](../scripts/audit/dtlz56_energy_recurrence.py)
returns only `numerically_feasible` or
`numerically_infeasible` **observations** and
records `scientific_status=unverified`. New tests
compare it to an independently written original
angle/length construction at \(M=2,3,4,5,10\),
including singular-axis boundaries. These are
ordinary floating-point consistency tests, not
directed-rounding proofs.

**Conservative continuous-interval extension.** For any
closed interval \([L,U]\subseteq[0,g_{\max}]\),
the functions \(r(g)^2=(1+g)^2\) and
\(\beta(g)=\sin^2(\pi/(4(1+g)))\) are respectively
increasing and decreasing. For every \(g\in[L,U]\),

\[
q_0(g)\ge q_0^-=
\max\{0,(1+L)^2-y_M^2\},\qquad
\beta(g)\ge\beta^-=
\sin^2\!\left(\frac{\pi}{4(1+U)}\right).
\tag{DTLZ56-19}
\]

The maps \(q\beta\) and
\(\max(q\beta,q-y_j^2)\) are nondecreasing
in nonnegative \(q,\beta\). Consequently the
fixed-\(g\) recurrence admits a conservative
interval **lower bound**, using
\(q_0^-\) and \(\beta^-\) throughout. If
\(q^-\beta^->y_j^2\) at an inner objective,
or the final \(q^->y_1^2\), every
\(g\in[L,U]\) is infeasible. Otherwise the
interval is **unresolved**: the optimistic
lower bound cannot prove that a feasible
dominator exists. This implication holds in
exact real arithmetic, including degenerate
intervals and coordinate singularities.

The independent
`observe_interval_energy_exclusion()` helper
in the same module implements this conservative
recurrence. Its `numerically_excluded` outcome
is **not a certified exclusion**, because
ordinary floating-point evaluation has not yet
established outward error enclosures. In
particular, for a candidate at \(g_y>0\),
checking \([0,g_y-\delta]\) cannot resolve
the open interval \((g_y-\delta,g_y)\);
even applying the bound to \([0,g_y]\)
typically remains unresolved because it
contains the original feasible candidate.

**Exact-rational interval exclusion (research implementation).**
The new independent helper
[`scripts/audit/dtlz56_rational_interval.py`](../scripts/audit/dtlz56_rational_interval.py)
evaluates the conservative exclusion implication
(DTLZ56-19) with exact Python rational arithmetic.
It does **not** call hardware sine or use heuristic
floating-point slacks. Instead, it constructs
provable rational upper and lower bounds on
\(\pi\), using Machin's identity

\[
\pi=16\arctan(1/5)-4\arctan(1/239)
\tag{DTLZ56-20}
\]

and alternating rational \(\arctan\) series.
For \(0\le t\le\pi/4<1\), an even number
of terms of the alternating sine Taylor
series gives a rigorous lower bound on
\(\sin t\). Combining these bounds with
the monotonicity of sine on this domain
gives an **exact-rational lower bound**
on \(\beta(g)\) throughout \([L,U]\).

The lower squared-residual recurrence
then uses only exact rational addition,
multiplication, subtraction, \(\max\)
and strictly positive comparisons. If
a lower bound proves \(q^-\beta^->y_j^2\),
or the final \(q^->y_1^2\), **no feasible
original DTLZ5/6 objective vector with
any \(g\in[L,U]\) satisfies the supplied
caps**. If strict separation cannot be
proved, the method returns `unresolved`,
even if an ordinary floating-point
implementation happens to report infeasibility.

Floating-point **inputs** supplied to
this helper are first converted to their
exact binary-rational values. The
conclusion therefore concerns those
precisely specified objective caps
and interval endpoints. It is independent
of the host machine's BLAS, libm and
floating-point trigonometric behavior.

The machine-readable result states
`rigorously_excluded` or
`unresolved`, the rational interval,
the failed objective and the exact
positive rational separation (numerator and denominator encoded
as hexadecimal integers to avoid Python's decimal-digit conversion
limit). It
retains `scientific_certification=unverified`:
a sound partial interval exclusion is
not an independent proof that the
candidate \(y\) is a feasible original
GT point, nor a complete exclusion
of \([0,g_y)\), nor a demonstration of
finite GT coverage or indicator fitness.
Source- and result-bound acceptance
under ADR 0059 remains required.

**Crucial remaining difficulty:** a test at one
fixed \(g\), even if rigorously evaluated, cannot
prove \(\neg D(g;y)\) for every
\(0\le g<g_y\). Full positive-\(g\)
Pareto certification for \(M\ge4\) still needs
a mathematically exhaustive treatment of the
continuous scalar domain and its open endpoint.
Neither this corollary nor the new helper changes
public `assess_gt()` status, numerical baselines
or any accepted certification gate.

### 3.6.6. Theorem 5 — sound continuous-interval *relaxation*

For any \(0\le L\le U\le g_{\max}\), the angular domain is
**nested**: \([a(g),b(g)]\subseteq[a(U),b(U)]\)
whenever \(g\in[L,U]\). Consider a hypothetical
\(z\in\mathcal F(g)\) for some \(g\in[L,U]\). Radially
contracting \(z\) to

\[
z'=\frac{1+L}{1+g}z
\]

preserves every angular coordinate and moves to radius
\(1+L\), which is no greater in any nonnegative coordinate.
The contracted vector belongs to the **optimistically
relaxed** spherical patch of radius \(1+L\) with inner
angles in \([a(U),b(U)]\). Therefore

\[
\left(
\text{no point of the relaxed patch lies below }y
\right)
\Longrightarrow
\neg D(g;y)\quad\forall g\in[L,U].
\tag{DTLZ56-8}
\]

**Proof.** If a true fixed-\(g\) feasible vector \(z\le y\)
existed in the interval, the radial contraction would be
a member of the relaxed patch with \(z'\le z\le y\).
This contradicts its infeasibility. \(\square\)

This is an **exact real-arithmetic implication** that justifies
covering a continuous \(g\)-interval through branch-and-bound
rather than checking an arbitrary grid of \(g\) values. The
relaxed patch may admit vectors that are not feasible at any
actual \(g\in[L,U]\); in that case the branch-and-bound may
need more subdivisions or remain unresolved. *Optimistic
relaxation cannot, by itself, prove the existence of a true
dominator.* Any positive witness must instead be reconstructed
and checked against (DTLZ56-1)–(DTLZ56-2).

### 3.6.7. Theorem 6 — exact dominance by the entire g=0 curve

For \(M\ge2\), write the analytical \(g=0\) Pareto curve as

\[
C(t)=\big(a_1\cos t,\ldots,a_{M-1}\cos t,\sin t\big),
\quad t\in[0,\pi/2],
\]

where \(a_1=2^{-(M-2)/2}\) and
\(a_j=2^{-(M-j)/2}\) for \(2\le j\le M-1\).
For an arbitrary **nonnegative** objective vector \(y\),
the curve contains a weakly dominating vector if and only if

\[
y_j\ge a_j\sqrt{1-\min(y_M,1)^2}
\quad\text{for all }j=1,\ldots,M-1.
\tag{DTLZ56-9}
\]

**Proof.** The last coordinate constraint
\(\sin t\le y_M\) bounds the feasible angle by
\(t\le\arcsin(\min(y_M,1))\). Every other coordinate
of \(C(t)\) is a nonnegative multiple of \(\cos t\),
which decreases as \(t\) increases in the prescribed
interval. Thus the largest admissible angle minimizes
**all** those coordinates simultaneously, producing
(DTLZ56-9). If \(y_M\ge1\), the endpoint \(t=\pi/2\)
has zero first \(M-1\) coordinates and the same
formula applies. \(\square\)

This test is **necessary and sufficient for weak
dominance by the complete \(g=0\) curve**, not for
global Pareto membership against all positive-\(g\)
feasible points. A point outside the curve's
dominance region may still be dominated by a
positive-\(g\) point. This distinction was essential
to the diagnosis in issue #65; it prevents mistaking
a dense or continuously checked \(g=0\) reference
for the full original PF.

**Implemented independent check.** The module
[`_dtlz56_g0_curve.py`](../moeabench/mops/_dtlz56_g0_curve.py)
constructs the complete-curve candidate with
`t=asin(min(y_M,1))` directly from the analytical
coefficients in (DTLZ56-9), without sampling `pf()`, `ps()`,
or calling the original evaluator or production scalarizer.
If the reconstructed curve vector lies strictly inside the
candidate's first (M-1) objective caps by a declared numerical
safety margin, and the candidate has positive radial (g),
it reports a provisional **dominance witness**. If a cap
fails the complete-curve inequality by more than tolerance,
it reports `no_curve_dominator`; boundary cases remain
`unverified`. These are finite-arithmetic outcomes,
not directed-rounded interval certificates.

The structured `mop.assess_gt()` result includes
`curve_dominance` and its numerical witness count and rows
for original DTLZ5/6; `mop.report()` includes the same
evidence for an existing calibrated GT. **Its scientific
status stays `unverified`** until the numerical result has
passed the ADR 0059 environment and output gates. In
particular, finding no point on the `g=0` curve that
dominates (y) does **not** establish global efficiency
against the positive-(g) feasible image.

### 3.6.8. Theorem 7 — global Chebyshev scalarization and Pareto tie-breaking

Let \(w\in\mathbb R_{>0}^M\) be a positive vector
of objective weights. Consider the **global**
weighted Chebyshev problem over the entire original
feasible decision domain:

\[
t^\star(w)=
\min_{x\in[0,1]^N}\max_{1\le i\le M}\frac{f_i(x)}{w_i}.
\tag{DTLZ56-10}
\]

**Coverage of strictly positive Pareto vectors.**
For any globally Pareto-optimal \(y\in\mathbb R_{>0}^M\),
choose \(w=y\). Its own feasible realization gives
\(t^\star(w)\le1\). If the global optimum were strictly
less than one, some feasible vector would improve
**every** objective relative to \(y\), contradicting
Pareto optimality. Hence \(t^\star(y)=1\), and \(y\)
is itself a global minimizer. Consequently the union
of global minimizers over positive weights includes
every strictly positive Pareto vector, without any
convexity assumption.

**Safe tie-breaking.** Select a global minimizer of
(DTLZ56-10) having the **smallest \(g\)** among all
global minimizers with the same optimal value.
It is Pareto-optimal. Otherwise a feasible
dominator would have Chebyshev value no greater
than the global optimum, hence would also be a
global minimizer; by Theorem 3, its strictly
smaller norm would give strictly smaller \(g\),
contradicting the tie-break. \(\square\)

The *global* qualifier is essential: a locally
minimized scalarization value or a solver that
stops at a finite numerical tolerance does not
automatically certify its objective vector as
Pareto-optimal. Zero-weight or zero-objective
boundary cases are **not** covered by this
strictly positive weight construction; such
points require direct boundary checks or a
separate limiting argument.

The production full-front sampler corrected in
[#65](https://github.com/monacofj/moeabench/issues/65)
uses bounded scalarization to **construct finite
candidate references**, while the independent
global criterion of Theorem 3 and its audits
in #175 are used to **check** objective-space
dominance without trusting the same production
scalarizer as an oracle. The mathematical
coverage theorem does not guarantee that any
fixed collection of weights or any finite point
budget covers every relevant component at an
acceptable resolution.

### 3.6.9. What has and has not been certified numerically

Three **separate layers** implement these results:

| Layer | Implemented independent code | What is established | What remains unresolved |
|---|---|---|---|
| Feasible-image and same-ray audit | `moeabench/mops/_dtlz56_independent_audit.py` | Feasible angular/radial constraints; minimal-\(g\) lower bound; pointwise \(g=0\) proof; rejection of radially dominated candidates | Global efficiency of positive-\(g\) minimal-radius candidates |
| Objective-coordinate margin exclusion | `moeabench/mops/_dtlz56_dominance_bounds.py` | Numerically excludes a dominator that improves any coordinate by at least a selected \(\delta_f>0\), across all \(g\) intervals examined | Improvements below \(\delta_f\); formal directed-rounding guarantees |
| Radial-gap global audit | `moeabench/mops/_dtlz56_global_radius.py` | Searches the continuous **closed** subinterval \(0\le g\le g_y-\delta_g\); records a reconstructed numerical witness, optimistic interval exclusion, or unresolved budget | The open interval \(g_y-\delta_g<g<g_y\); directed-rounding and full source-bound environment verification |

The **global mathematical criterion** (DTLZ56-4) is exact.
**Its current executable implementation is not an exact
zero-gap oracle.** In particular, an interval discarded using
ordinary IEEE floating-point arithmetic with an outward
heuristic slack is **not** a formal interval-arithmetic
certificate; comparisons near a boundary can depend on
rounding, transcendental functions, BLAS/libm and platform.
The current audit appropriately preserves
`global_dominance.status = "unverified"` and full-original
DTLZ5/6 `validity.status = "unverified"` for \(M\ge4\),
even when all checked finite-gap intervals were numerically
excluded. A reconstructed dominance witness is actionable
evidence that requires independent feasibility verification,
not automatic promotion to a portable certificate.

Users may request these additional read-only measurements explicitly:

```python
mop = mb.mops.DTLZ5(M=4)
candidate = mop.assess_gt(
    n_points=32, probes=128,
    dominance_margin=0.01,          # coordinate improvement >= 0.01
    dominance_max_intervals=4096,
    dominance_g_gap=1e-4,           # radial improvement >= 0.0001
    dominance_g_max_intervals=8192,
)
print(candidate.validity.to_dict())
print(candidate.dominance.to_dict())
print(candidate.global_dominance.to_dict())
```

Both optional audits also appear in `mop.report(...)`
when explicitly requested for an already calibrated GT.
Neither changes the MOP, GT, calibration sidecar or
historical regression baseline. They are disabled by default
because their cost scales with points and interval subdivisions.

The next **mathematical** task in [#175](https://github.com/monacofj/moeabench/issues/175)
is a *zero-gap* decision method for (DTLZ56-4), including
precise treatment of the open boundary \(g\uparrow g_y\).
The next **numerical** task is to implement its conclusions
with independently checked, directed-rounded interval
bounds (or another demonstrably rigorous decision procedure)
and certify the execution environment and resultant artifacts
under [ADR 0059](adr/0059-gt-quality-measurement-and-acceptance.md).
Only then can the front membership of positive-\(g\) points
be promoted beyond `unverified`; **finite-sample coverage,
regularity, component completeness and indicator sensitivity
still require their separate certification gates**.

### 3.6.10. Independent decision-space witnesses and limits

A GT is generally saved as a cloud of **objective** vectors;
those coordinates alone cannot establish that any **feasible
decision** actually produced them. For original DTLZ5/6,
`assess_gt()` now accepts an explicitly aligned
`decision_reference`, in addition to the objective
`reference`:

```python
mop = mb.mops.DTLZ5(M=4)
decisions = ...  # exact rows whose objective evaluations formed the GT
objectives = ... # matching rows, in the same order
a = mop.assess_gt(
    reference=objectives,
    decision_reference=decisions,
)
print(a.feasibility.to_dict())
# With an existing, explicitly calibrated objective GT:
mop.report(decision_reference=decisions)
```

The independently implemented
[`_dtlz56_decision_witness.py`](../moeabench/mops/_dtlz56_decision_witness.py)
checks every supplied decision against the original
\(x\in[0,1]^N\) bounds, independently computes the exact
DTLZ5 or DTLZ6 distance function and angular transform in
(DTLZ56-1)–(DTLZ56-2), reconstructs the objective vector, and
compares it row-for-row with the claimed GT using the declared
numerical tolerance. It does **not** invoke the production
`evaluation()`, `ps()`, `pf()`, or scalarizer.

A `feasibility.status="pass"` certifies the **stated
decision/objective correspondence within the numerical
tolerance** for the original DTLZ5/6 equations, **not**
the global Pareto efficiency of those points, which is
governed independently by (DTLZ56-4). Incorrect row
alignment, missing decisions, bounds violations and
objective mismatches must not be hidden by merely
rechecking finite objective-space nondominance.

The witness is optional, read-only and has to be
**supplied explicitly**. For on-demand
`n_points` samples, decision correspondence is
not assumed automatically: use an explicit
`reference` and matching decisions instead.
The reporting method similarly never regenerates a
PS or changes an existing calibration sidecar.
Other MOPs may have their declared decision bounds
checked, but without a verified independent
equation/constraint oracle, their general
`feasibility` remains `unverified`.
Independent replay, constrained decision equations
for other families and platform-certified numerical
comparison remain separate #175 obligations.

### 3.6.11. Theorem 8 — sufficient zero-gap local dominance by a feasible differential path

The exact global condition (DTLZ56-4) has an open endpoint
\(g\uparrow g_y\), which cannot be certified by increasing
the resolution of an ordinary positive-gap search. There is,
however, a separate **constructive sufficient condition for
non-Pareto-optimality** that works without prespecifying any
positive radial improvement.

Write the original feasible objective parameterization as
\(F(g,u)\), with \(u=(x_1,\ldots,x_{M-1})\in[0,1]^{M-1}\)
and \(g\in[0,g_{\max}]\), as in (DTLZ56-1)–(DTLZ56-2).
At a feasible candidate with \(g_0>0\) and position \(u_0\),
choose a one-sided path

\[
u(g)=u_0+(g-g_0)v,\qquad g\le g_0.
\tag{DTLZ56-11}
\]

For this path to remain in the original position bounds
for all sufficiently small reductions of \(g\), the
direction \(v\) must satisfy

\[
v_i\le0\quad\text{when }u_{0,i}=0,\qquad
v_i\ge0\quad\text{when }u_{0,i}=1,
\tag{DTLZ56-12}
\]

with no sign restriction when \(0<u_{0,i}<1\).
The distance variables can independently realize every
nearby \(g\ge0\) using the constructions in Section 3.6.1.

**Theorem (strict differential sufficiency).** If such a
feasible \(v\) exists with

\[
\left.
\frac{dF_j(g,u(g))}{dg}\right|_{g=g_0}
>0\quad\text{for every }j=1,\ldots,M,
\tag{DTLZ56-13}
\]

then the candidate is **not** globally Pareto-optimal:
there exists a feasible \(g'<g_0\), arbitrarily close
to \(g_0\), with

\[
F_j(g',u(g'))<F_j(g_0,u_0)\quad\forall j.
\]

**Proof.** For \(g_0>0\), the original spherical map and
its angle formulas are continuously differentiable in
a neighborhood of \((g_0,u_0)\). By (DTLZ56-12), the
path is feasible for sufficiently small positive
\(\varepsilon\) with \(g'=g_0-\varepsilon\ge0\).
By (DTLZ56-13) and continuity of every derivative,
each objective is strictly increasing with \(g\)
throughout some left-neighborhood of \(g_0\) along
the path. Consequently reducing \(g\) strictly
decreases **all** objectives, furnishing a feasible
dominator. \(\square\)

An equivalent **linear feasibility problem** follows
from differentiating the original equations:

\[
\frac{dF}{dg}=b(g_0,u_0)+A(g_0,u_0)v.
\]

Here \(b\) contains the radial derivative and
the fixed-position angular derivatives, while
\(A\) contains the angular Jacobian multiplied
by the derivatives with respect to \(v\).
One can maximize a common derivative floor \(t\)
subject to

\[
b+Av\ge t\mathbf 1,\qquad
v\text{ obeys (DTLZ56-12)}.
\tag{DTLZ56-14}
\]

A strictly positive **mathematically verified** feasible
solution \(t>0\) establishes (DTLZ56-13).
A zero, negative, solver-failed or merely unverified
numerical value **does not prove global Pareto
optimality**: other feasible paths, nonlocal dominators
or higher-order improvements can exist.

The independent implementation
[`_dtlz56_local_descent.py`](../moeabench/mops/_dtlz56_local_descent.py)
uses product differentiation of the original
hyperspherical equations (avoiding divisions by
zero at coordinate boundaries), a bounded-variable
linear program to search for \(v\), and **directly
reconstructs a finite feasible objective-and-decision
witness** at some smaller \(g\). It is available
on request through

```python
audit = mop.assess_gt(
    reference=objectives,
    decision_reference=decisions,
    local_descent=True,
)
print(audit.local_descent.to_dict())
# Also: mop.report(decision_reference=decisions, local_descent=True)
```

The extra optimization cost is **opt-in** and requires
aligned decision witnesses; objective coordinates alone
do not preserve the necessary original position
variables. The public `local_descent.status` is
`unverified` at the **portable scientific certificate**
level, even when a strictly improving numerical
witness was found: the derivative calculation,
LP and trigonometric reconstruction use ordinary
floating point and have not been certified through
the three ADR 0059 gates. Such a witness is useful
actionable evidence against that candidate's
membership. Absence of a witness is **never**
a global Pareto certificate.

**Scope relative to the remaining blocker.** This
theorem rigorously identifies some candidates for
which dominance exists at arbitrarily small positive
\(g\)-gaps. It does **not** establish the complementary
zero-gap global nondominance needed to certify
every positive-\(g\) true-front point. A complete
method must still prove exclusion of all smaller
feasible \(g\), including the entire open endpoint
of (DTLZ56-4), or provide an equivalent independently
proved characterization.

## 4. DPF3: Mathematical Definition and Chaotic Thresholds

### 4.1. Essential objectives

In published DPF3, the first \(D-1\) position variables define the angles

\[
\theta_j=\frac{\pi}{2}x_j^{100},\qquad j=1,\ldots,D-1,
\]

and the distance function is

\[
g(x)=\sum_{i=D}^{N}(x_i-0.5)^2\geq0.
\]

Let \(H_D(\theta)\) denote a nonnegative \(D\)-coordinate unit hyperspherical vector, so that \(\|H_D(\theta)\|_2=1\). The essential objectives are

\[
\gamma(x)=(1+g(x))(\mathbf1-H_D(\theta(x))).
\]

The first \(D-1\) published objectives are the corresponding first coordinates of \(\gamma\). Its final essential coordinate is represented through a sequence of saturating projections that produce redundant objectives.

The **power 100** is fundamental: a uniform grid in \(x_j\) is very far from uniform in \(\theta_j\), leaving substantial regions of the PF underrepresented.

### 4.2. What are the “chaotic” parameters \(\eta\)?

DPF3 uses \(U=M-D\) threshold values \(\eta_1,\ldots,\eta_U\), which determine the **interval boundaries** of its redundant-objective projections. They are generated by a deterministic sequence of a nonlinear **logistic map**, of the form

\[
z_{n+1}=3.8z_n(1-z_n).
\]

“Chaotic” describes the nonlinear, sensitive behavior of that dynamical system. The thresholds are **not** independently randomized on every objective evaluation or solver run. Once fixed, they form part of the problem's mathematical definition.

The paper requires the thresholds to be **used in ascending order**. For example, values generated in the order \(0.34,0.85,0.47\) must be used as \(0.34,0.47,0.85\).

### 4.3. How do those thresholds affect the objectives?

Let \(t=\gamma_D\) be the final essential coordinate. The \(U+1\) projected objectives are

\[
\begin{aligned}
p_0(t)&=\min(t,\eta_1),\\
p_i(t)&=\min\{\max(t,\eta_i),\eta_{i+1}\},
\quad i=1,\ldots,U-1,\\
p_U(t)&=\max(t,\eta_U).
\end{aligned}
\]

With **ordered thresholds**, each interval of \(t\) affects one projected coordinate while the others remain constant. The crucial conservation identity is

\[
\sum_{i=0}^{U}p_i(t)=t+\sum_{j=1}^{U}\eta_j.
\]

It also allows us to recover the **order** of the essential coordinate from the projection vector: if all projected coordinates at \(t_a\) are less than or equal to those at \(t_b\), then necessarily \(t_a\le t_b\).

With unsorted thresholds, intervals can be inverted or overlap, some projections can become artificially constant, and the identity need not hold. The resulting objective functions and PF geometry can differ from the paper and affect quality indicators. Unsorted thresholds **alone** do not prove that every generated point becomes dominated.

### 4.4. What was wrong in the audited implementation?

The audited DPF3 implementation received a chaos sequence from **\`BaseDPF\`** and used it without enforcing the strictly ascending order required by the DPF3 paper. When \(M-D\ge2\), this may change the objective values. For \(M=3,D=2\), only one threshold exists, so that case's cardinality defect cannot be blamed on unsorted thresholds.

Comparing **identical** \(g=0\) decisions through the incumbent evaluator and an independent reconstruction of the published equations yielded the following **maximum per-coordinate differences in the tested samples**:

| DPF3 | Maximum observed difference |
|---|---:|
| \(M=3,D=2\) | 0 |
| \(M=5,D=2\) | 0.3844009575 |
| \(M=6,D=3\) | 0.3844009575 |
| \(M=7,D=4\) | 0.3844009575 |

These are **sample maxima**, not universal error bounds. Sorting a shared base-class generator indiscriminately would also risk changing other DPFs. A production correction must apply the DPF3-specific mathematical requirement without unintentionally altering other benchmarks.

### 4.5. Global proof of the published DPF3 PS and PF

Take a feasible decision \(y\) with \(g(y)=0\). Suppose a different feasible \(x\) strictly dominates \(y\). The ordered-projection identity would force the essential vector \(\gamma(x)\) to dominate \(\gamma(y)\). Since \(g(x)\ge0\), that would require the nonnegative unit vector \(H_D(\theta(x))\) to be componentwise at least \(H_D(\theta(y))\), with a strict improvement compatible with dominance.

However, **two nonnegative vectors of unit Euclidean norm cannot satisfy a strict componentwise ordering**: increasing any coordinate without decreasing another increases the norm. Hence a \(g=0\) point cannot be strictly dominated.

Conversely, for every feasible \(x\) with \(g(x)>0\), keep its angular parameters and set all distance variables to \(0.5\). The resulting feasible point has \(g=0\), and its essential objectives improve componentwise, strictly in at least one coordinate; the ordered projections preserve strict improvement in at least one output.

Thus, for the **published mathematical model**,

\[
\boxed{PS_{DPF3}=\{x:g(x)=0\}},\qquad PF_{DPF3}=F(PS_{DPF3}).
\]

The proof holds across valid intrinsic dimensions under the published hypotheses. It certifies the **complete continuous mathematical parameterization**, not the correctness of every software evaluator. See the [full proof](research/gt-issue119/gt-dpf3-global-pareto-proof-issue119.md).

## 5. DPF3: Numerical Cancellation, Finite Samples, and Geometric Coverage

### 5.1. Catastrophic cancellation in \(1-\cos(\theta)\)

For \(D=2\), the first essential objective uses

\[
1-\cos(\theta),\qquad\theta=\frac{\pi}{2}x_1^{100}.
\]

At small positive \(\theta\), \(\cos(\theta)\) is extremely close to 1. Under **float64**, it can round to exactly 1, making the subtraction return **zero** even though the true mathematical value is positive. This is **catastrophic cancellation**.

The trigonometric identity

\[
1-\cos(\theta)=2\sin^2(\theta/2)
\]

avoids subtracting nearly identical numbers and is more stable. For extremely small values, however, **underflow** can still produce zero when the result is below the representable range. Replacing one kernel therefore does not automatically resolve discretization or other objective-precision problems.

A **160-decimal-digit reference calculation** for 512 decision points found 511 mathematically positive first-objective values (the point \(x_1=0\) is exactly zero). Of those positive values:

| float64 expression | Mathematically positive values rounded to zero |
|---|---:|
| Direct \(1-\cos(\theta)\) | 423 |
| Stable \(2\sin^2(\theta/2)\) | 12 |

Counting the **true** zero at \(x_1=0\), the respective totals of numerical zeros are **424** and **13**. At a requested sample size of 128, the direct expression lost 105 of 127 mathematically positive results; the stable form lost three. Reports must distinguish **positive values lost** from the **total number of numerical zeros**.

Experimentally replacing only the first-component kernel improved the number of sample-nondominated points, but did **not** correct the unsorted thresholds, every other precision issue, or the sample's geometric distribution.

### 5.2. Nondominated filtering is not a GT generator

The audited DPF3 code returned approximately:

| Requested count | Decisions from ps() | Objectives returned by pf() |
|---|---:|---:|
| 128 | 128 | 23 |
| 512 | 512 | 88 |

These counts occurred for the investigated \(D=2\) cases with \(M=3\) or \(M=5\). The \`pf()\` method filtered an internally evaluated sample using **sample-relative nondominance**, reducing cardinality. Besides not proving **global** Pareto optimality, filtering destroys any automatic position-by-position correspondence between PS decision rows and the returned PF objectives.

### 5.3. A certified angular reference for D=2

Rather than uniformly sampling the decision parameter \(x_1\), sample the **actual hyperspherical angle**:

\[
\theta_i=\frac{i}{n-1}\frac{\pi}{2},
\qquad
x_{1,i}=\left(\frac{2\theta_i}{\pi}\right)^{1/100},
\quad i=0,\ldots,n-1.
\]

For \(D=2\), the published PF is **1-Lipschitz** in \(\theta\). Its ordered projections cannot increase geometric distances beyond the corresponding scalar change. Every angle is at most **half a grid step** away from a selected value, giving the **exact-real-arithmetic** bound

\[
\boxed{d_H(PF,GT_n)\le\frac{\pi}{4(n-1)}}.
\]

Distances use **unnormalized Euclidean objective space**. The bound concerns the exact image of the angular grid under the **published** model, with every point already proved to lie on PF. For \(n=128\) it is approximately **0.00618424**; for \(n=512\), **0.00153698**.

The independent candidate produced **128 and 512 distinct objective vectors**, respectively, in the tested \(M=3,5,10\) configurations, unlike the 23 and 88 points returned by the audited incumbent \`pf()\`. This is a **mathematically supported, reproducible candidate construction**, not merely a visually smoother plot.

### 5.4. Extending the bound to arbitrary intrinsic dimension

The angular parameterization \(H_D\) has mutually orthogonal angular partial derivatives, each with Euclidean norm at most 1. Hence the hyperspherical map is **1-Lipschitz** in its angular coordinates, and composing it with the ordered projections preserves that property.

An endpoint-inclusive **tensor grid** with \(k\) values along each of the \(D-1\) free angular axes has \(k^{D-1}\) nominal decision rows and guarantees, in exact real arithmetic,

\[
\boxed{d_H(PF,GT_k)\le
\frac{\sqrt{D-1}\,\pi}{4(k-1)}}.
\]

The bound does not depend on the number of redundant objectives \(M\), provided the published definition and ordered thresholds are respected. It is **conservative**, but applies to the full continuous PF, unlike distances observed at finitely many random probes.

Two limitations matter. First, nominal cost grows **exponentially** in \(D\), so tensor grids can become impractical at high intrinsic dimension. Second, hyperspherical parameterization singularities can map different decisions to the same objective vector, reducing effective cardinality.

| DPF3 | Values per angle | Grid decision rows | Exactly unique numerical objectives (float64) | Mathematical bound |
|---|---:|---:|---:|---:|
| \(M=5,D=3\) | 5 | 25 | 23 | 0.277680 |
| \(M=5,D=3\) | 9 | 81 | 75 | 0.138840 |
| \(M=6,D=4\) | 4 | 64 | 45 | 0.453450 |
| \(M=6,D=4\) | 7 | 343 | 266 | 0.226725 |

Those uniqueness counts are **numerical**, not proof of mathematical uniqueness. Spherical singularities identify different angular parameter combinations with the same objective vector, and rounding can introduce or conceal distinctions.

### 5.5. Precision of the inverse-angle transformation

To realize a desired angular grid point as a decision vector, the construction uses \(x_j=(2\theta_j/\pi)^{1/100}\). Storing \(x_j\) in float64 and reconstructing \(\theta_j\) through \(x_j^{100}\) introduces rounding.

With **100-decimal-digit mpmath calculations**, the studied cases showed angular discrepancies around **\(10^{-15}\) to \(10^{-14}\)** and similar objective-space round-trip discrepancies.

**An observed high-precision discrepancy is not a rigorous IEEE rounding-error enclosure.** The Hausdorff certificate holds for the **ideal exact-arithmetic grid**. A claim about a deployed float64 generator additionally requires conservative bounds on the inverse transform, power reconstruction, and objective evaluation. See the [D=2 proof](research/gt-issue119/gt-dpf3-d2-angular-cover-proof-issue119.md) and the [higher-dimensional proof and measurements](research/gt-issue119/gt-dpf3-nd-cover-and-acceptance-issue119.md).


## 6. DPF5: Mathematical Definition of Its Two Components

### 6.1. Dimensions and distance function

DPF5 has two branches of the Pareto front, conventionally labeled **P** and **D**. Branch P has intrinsic geometric dimension \(M-1\), whereas the degenerate branch D has intrinsic dimension \(D-1\). The published formulation requires \(N\ge M\).

Its distance function is

\[
g(x)=(x_M-x_1)^2+\sum_{i=M+1}^{N}(x_i-0.5)^2.
\]

Consequently \(g\ge0\). The condition \(g=0\) requires **\(x_M=x_1\)** and all variables \(x_i\), \(i>M\), to equal \(0.5\). The first \(M-1\) variables determine the position along the front.

### 6.2. Correct equations for P and D

Let \(H_q\) be the standard nonnegative **unit hyperspherical map** with \(q\) coordinates and \(q-1\) angular parameters.

On branch **P**, objective directions use a hypersphere of \(M\) coordinates and all \(M-1\) angles:

\[
u_P=H_M(\theta_1,\ldots,\theta_{M-1}),
\quad\theta_j=(\pi/2)x_j.
\]

On branch **D**, first construct

\[
h=H_D(\theta_1,\ldots,\theta_{D-1}).
\]

The first component \(h_1\), divided by \(\sqrt{M-D+1}\), is repeated \(M-D+1\) times; the other \(D-1\) coordinates are appended:

\[
u_D=(\underbrace{h_1/\sqrt{M-D+1},\ldots,
h_1/\sqrt{M-D+1}}_{M-D+1\ \mathrm{terms}},h_2,\ldots,h_D).
\]

Both maps satisfy \(u\ge0\) and \(\|u\|_2=1\). The published objective function is

\[
F(x)=(1+g(x))u(x)
\]

with the **strict branch definition**

\[
u(x)=
\begin{cases}
u_P(x),&x_1<1/3,\\
u_D(x),&x_1\ge1/3.
\end{cases}
\]

In particular, **\(x_1=1/3\) belongs to D**. This boundary can matter when a structured sampling grid includes that exact value, although a continuous random sampler almost never draws it.

### 6.3. A powerful independent spherical invariant

It follows from the published equations that

\[
\boxed{\sum_{i=1}^{M}f_i(x)^2=(1+g(x))^2}.
\]

The identity holds on both branches and for both \(g=0\) and \(g>0\). It is a useful **necessary** check for repeated or omitted angular coordinates, incorrect indexing, or wrong scale factors. By itself, it does **not** establish Pareto-optimality or sufficient finite sample coverage.

The audited DPF5 evaluator disagreed with the published P branch in several higher-dimensional configurations and with the published D branch in some configurations, notably intrinsic \(D=4\). It also assigned the exact equality case \(x_1=1/3\) to P instead of D.

### 6.4. Global proof of the published DPF5 PS and PF

For a point \(y\) with \(g(y)=0\), we have \(\|F(y)\|_2=1\). If a feasible \(x\) strictly dominated \(y\), nonnegativity of all objective coordinates would imply \(\|F(x)\|_2<1\). Yet every feasible point satisfies \(\|F(x)\|_2=1+g(x)\ge1\), a contradiction.

The same argument applies **across P and D**: a globally optimal point on one branch cannot be dominated by another feasible point from the other branch.

Conversely, for any \(x\) with \(g(x)>0\), preserve its first \(M-1\) position variables and construct a feasible decision \(y\) with \(y_M=y_1\) and \(y_i=0.5\) for \(i>M\). Then \(g(y)=0\), the branch and direction \(u\) remain unchanged, and

\[
F(x)=(1+g(x))F(y).
\]

Thus \(y\) strictly dominates \(x\). Therefore, for the **published model**,

\[
\boxed{PS_{DPF5}=\{x:g(x)=0\}},\qquad PF_{DPF5}=F(PS_{DPF5}).
\]

This proves global membership and completeness on **both** components and at their branch boundary. It does **not** prove that a particular finite sampler covers those components adequately. See the [full mathematical proof](research/gt-issue119/gt-dpf5-global-pareto-proof-issue119.md).

## 7. DPF5: Implementation Discrepancies and Effective Cardinality

### 7.1. Equation discrepancies across dimensions

Independent reconstruction found indexing errors and reuse of trigonometric expressions in the incumbent P branch for larger \(M\). With \(D=2\), interior branches matched the published equations at \(M=3\) and \(M=4\), but P diverged for \(M\ge5\). For intrinsic \(D=3\) or \(D=4\), additional cases diverged, including D at \(D=4\).

A check of the **maximum residual of the sum-of-squares invariant** for P with \(g=0\) found:

| \(M,D\) | Maximum residual, audited evaluator | Maximum residual, published reconstruction |
|---|---:|---:|
| \(3,2\) | \(2.22\times10^{-16}\) | \(2.22\times10^{-16}\) |
| \(5,2\) | 0.9943366 | \(4.44\times10^{-16}\) |
| \(6,3\) | 0.171273 | approximately zero at numerical precision |
| \(6,4\) | 0.314780 | approximately zero at numerical precision |

For branch **D** at \(M=6,D=4\), the maximum observed residual was approximately **0.321684**, while the independent reconstruction respected the identity. These are **equation defects**, not mere choices of PS sampling or filtering.

In trials with **\(g=0.04\)** and \(D=2\), the P branch discrepancy persisted. The maximum residual of the scaled spherical invariant was approximately **0.07764** at \(M=5\), **0.17928** at \(M=6\), and **0.49334** at \(M=10\). Thus the evaluator defect also exists **off** the Pareto set.

### 7.2. Far fewer unique objectives than requested

For \(M=3\), the audited \`DPF5.ps()\` generated a square grid with **\`floor(sqrt(n_points))\`** coordinates on each axis. Consequently:

| \(M=3,D=2\) | Requested decisions | Returned decisions | Unique objectives, audited evaluator | Unique objectives, published reconstruction |
|---|---:|---:|---:|---:|
| Example 1 | 128 | 121 | 51 | 51 |
| Example 2 | 512 | 484 | 190 | 169 |

The counts **190 and 169 are not contradictory**. They describe the **same decision grid**, evaluated under **two different definitions of the P/D boundary**. A grid with 22 values per axis contains \(x_1=1/3\). The incumbent evaluator sends those points to P; the published model sends them to D.

Under the published equations at requested count 512, **154 P decisions** produce 154 unique objectives, while **330 D decisions** produce only **15 unique objectives**. The degenerate branch therefore contains many distinct decisions mapping to the same objective vector.

### 7.3. Sample-relative dominance under the defective evaluator

For DPF5 with \(M=5,D=2\), of 512 generated decisions, **462 objective vectors were dominated within the sample** under the audited evaluation. Branch P had 183 decisions, 151 of which were dominated within P; counting dominance relationships across both branches increased the total to 462.

Re-evaluating those **same decisions** with the independent **published model** eliminated all 462 observed dominance cases, consistent with the theorem that every \(g=0\) point is globally Pareto-optimal.

This illustrates why **evaluator defects** must be distinguished from **sampler defects**. A sample can remain poorly distributed even after every generated point is mathematically optimal. Filtering alone cannot replace correct equations.

## 8. Covering Both Components of DPF5

### 8.1. What was actually measured?

The study evaluated **unchanged decision samples from the incumbent generator** under the **independent published DPF5 equations**. Independent probe decisions were generated separately on P and D. For each branch, the Euclidean objective-space distance to the nearest **same-branch** unique sampled objective was measured. Separating the branches prevents many D points from concealing missing coverage on P.

Finite-probe medians and maxima describe **tested configurations**. They are not certified upper bounds on the continuous Hausdorff error.

| \(M,D\) | Actual PS rows, requested 512 | Distinct published objective vectors | P median | P observed maximum | D median | D observed maximum |
|---|---:|---:|---:|---:|---:|---:|
| \(3,2\) | 484 | 169 | 0.02977 | 0.07854 | 0.01596 | 0.03738 |
| \(5,2\) | 512 | 512 | 0.12904 | 0.32301 | 0.00128 | 0.01118 |
| \(6,3\) | 512 | 512 | 0.16216 | 0.49741 | 0.02042 | 0.09446 |
| \(6,4\) | 512 | 512 | 0.16216 | 0.49741 | 0.04946 | 0.18339 |

For example, at \(M=5,D=2\), the median sampled gap in P was about a hundred times the median in D. This is an **allocation and geometry** disparity, not proof that one branch does not mathematically exist.

### 8.2. Why a universal fixed P/D fraction is unjustified

P is an \((M-1)\)-dimensional manifold while D has dimension \(D-1\). At larger dimensions, P may require many more points than D for similar geometric coverage. Nevertheless, ideal allocations also depend on component extent, curvature, extremes, interfaces, and the total budget.

Allocating **50% of samples to each branch** is not equivalent to balancing geometric error. A rule such as **two-thirds to P** is not universally justified either. The geometry of a mixed-dimensional front does not itself prescribe a unique probability measure over its components.

This does not imply that a good sampler cannot be designed. It means that each **configured DPF5 instance** needs a geometrically supported strategy with explicit effective budget, extremes, and error criterion.

### 8.3. Comparing the incumbent sampler to scrambled Sobol sampling

An experimental alternative used **deterministically scrambled Sobol sequences**, augmented with selected boundary points. It preserved the **exact same count of unique objectives separately on P and D** as the incumbent sample. P used \(M-1\) angular parameters; D used only its \(D-1\) essential angular parameters.

Median nearest-reference distance, 95th percentile, and largest observed distance were compared using **three independently seeded probe sets**, at requested budgets 128 and 512. Results were **mixed**, even at equal effective branch budgets.

For illustration, the table shows the change in **median** nearest-reference distance (incumbent minus Sobol; **positive favors Sobol**):

| \(M,D\) | Requested | Three P median differences | Three D median differences |
|---|---:|---|---|
| \(3,2\) | 128 | +0.00262, −0.00102, +0.00247 | −0.00423, −0.00839, −0.00731 |
| \(5,2\) | 512 | +0.00435, +0.00546, +0.00815 | +0.00036, +0.00034, +0.00038 |
| \(6,3\) | 128 | −0.01026, −0.02384, −0.01700 | +0.00294, +0.00386, +0.00189 |
| \(6,3\) | 512 | +0.00281, −0.00250, +0.00244 | +0.00204, +0.00201, +0.00081 |

**Simply replacing an existing sampler with Sobol does not universally resolve objective-space coverage.** A better median may coexist with poorer extremes or larger maximum gaps. The selected boundary anchors were useful landmarks, not a proof that **all** component boundary features had been covered.

### 8.4. P/D allocation under a fixed total effective budget

A separate experiment held the **total count of branchwise unique objectives fixed** and compared the incumbent distribution with alternatives allocating **half** or **approximately two-thirds** of points to P. It used three sampling seeds and three probing seeds, sampling probes independently per branch.

Values show the **largest observed conditional gap, maximized across sampler seeds**, for requested count 512; **smaller is better**:

| \(M,D\) | Incumbent allocation | 50% to P | Approximately 67% to P |
|---|---:|---:|---:|
| \(3,2\) | **0.07705** | 0.10917 | 0.08702 |
| \(5,2\) | 0.41784 | 0.34187 | **0.33189** |
| \(6,3\) | 0.45091 | 0.42947 | **0.38824** |
| \(6,4\) | 0.45091 | 0.42947 | **0.38824** |

At \(M=3,D=2\), the incumbent's **unique-objective** allocation already put approximately **91% of points in P**, so a fixed two-thirds rule **removed** points from P and worsened the diagnostic. In the studied higher-dimensional cases, increasing P's proportion reduced the largest observed gap. At budget 128, effects likewise depended on geometry and were sometimes substantial.

This establishes **geometry- and budget-sensitive empirical behavior**, not a universally optimal ratio or a certified worst-case distance on the full continuous PF. It is not a statistically exhaustive comparison of all sampling methods.

### 8.5. An analytic conditional covering bound for DPF5 tensor samples

The two DPF5 components also admit **mathematically certified finite covering constructions**, provided each component contains a **complete angular tensor grid**. This follows from the same hyperspherical 1-Lipschitz argument used for DPF3, applied independently to P and D. The replicated-coordinate mapping defining D is a Euclidean isometry: duplicating its first essential coordinate after division by \(\sqrt{M-D+1}\) preserves Euclidean distances.

For P, use \(k_P\ge2\) angular positions per independent coordinate, with first decision coordinate ranging from \(0\) to the closest representable number **strictly below** \(1/3\). Write \(\delta_P=(\pi/2)(1/3-\operatorname{nextafter}(1/3,0))\). The ideal continuous directed-covering bound is

\[
\varepsilon_P(k_P)
\le \frac{\pi}{4(k_P-1)}\sqrt{\frac19+(M-2)}+\delta_P.
\]

For D, use \(k_D\ge2\) angles along each essential coordinate; the first parameter belongs to \([1/3,1]\) and the others to \([0,1]\):

\[
\varepsilon_D(k_D)
\le \frac{\pi}{4(k_D-1)}\sqrt{\frac49+(D-2)}.
\]

If a finite sample retains a complete grid in **both** branches, then every point on the published two-component PF is within

\[
\boxed{d_H(PF,GT)\le\max\{\varepsilon_P(k_P),\varepsilon_D(k_D)\}}
\]

of a sample point in unnormalized Euclidean objective space, under **exact mathematical arithmetic**. In particular, the proof needs **no** probability measure combining manifolds of different intrinsic dimensions. For DPF5, a complete grid requires nominal budgets \(k_P^{M-1}\) and \(k_D^{D-1}\), respectively, so the cost can grow rapidly in high dimensions. Spherical singularities can produce duplicate objective images without invalidating the geometric support of a full grid.

The proposed MOP-specific finite sampler in the DPF3/DPF5 correction work retains nested angular grids whenever its available branch budget permits them, fills objective duplicates with deterministic interior samples, and uses a **conditional** criterion to choose the total P/D allocation. **A returned sample receives this analytic support guarantee only when it contains a complete grid in each component.** Small budgets, high-dimensional cases without complete grids, and fully certified float64 rounding are separately qualified. A finite-probe statistic must never be confused with the real-arithmetic continuous-covering bound.

The [full DPF5 angular covering proof and implementation conditions](research/gt-issue119/gt-dpf5-angular-cover-candidate-issue119.md) explain the assumptions, branch boundary, conditional guarantee and remaining numerical distinctions.

## 9. Consequences for IGD, IGD+, and Hypervolume

The research isolated two effects that must not be confused.

**Evaluation effect:** hold exactly the same **decision vectors** fixed, then compute objectives under the incumbent and independently reconstructed published evaluator. Differences indicate that the mathematical problem being optimized changes, potentially affecting the algorithm's evaluated solutions and search trajectory.

**Reference effect:** hold exactly the same **objective-space population** fixed and recompute quality indicators against two different GT references. Differences are attributable to GT choice, **without rerunning any MOEA**.

The second experiment verified objective-population identity using **SHA-256**. Each trial used 64 candidate reference points and 16 frozen population points, and the repository's actual **IGD, IGD+, and HV** implementations.

For \(M=5,D=2\), seed 101, changing **only the reference** gave the following differences (*published-model reconstruction minus incumbent*):

| Quantity | Observed difference |
|---|---:|
| IGD | +0.02731 |
| IGD+ | +0.00270 |
| GT-relative/absolute HV ratio | +0.02663 |

At \(M=5,D=2\), seed 202, the differences were **+0.02933**, **−0.00076**, and **+0.01801**. At \(M=6,D=3\), seed 101, they were **+0.00967**, **−0.00020**, and **+0.00562**; at \(M=6,D=4\), seed 202, they were **+0.03978**, **+0.01271**, and **−0.00845**. Signs vary; the data do **not** show that correcting GT always increases or always decreases any indicator.

Two additional distinctions are essential:

- The audited **IGD and IGD+** implementation uses **reference-dependent zero-to-one normalization**. Changing GT can change both point locations and the internal scaling. A difference should not be described solely as geometric distance under unchanged normalization bounds.
- The **HV** experiment used common **ideal = 0**, **nadir = 2**, and normalized reference point **1.1**. The **solution population's hypervolume remained unchanged**; the **GT's hypervolume changed**, affecting the ratio. It would be incorrect to claim that the MOEA's solution hypervolume changed when only GT was replaced.

These are controlled numerical sensitivity studies, not universal forecasts of future MOEA performance. To assess the full effect of changing the objective evaluator, one must evaluate comparable decision histories under both definitions or run new studies on the corrected problem.

### 9.1. Finite-PF sampling and the absolute-HV denominator

The exact Pareto front `PF` and a finite calibrated sample
`G\subseteq PF` are different objects. For fixed minimization
normalization and reference point, set inclusion gives
`HV(G) \le HV(PF)`. The inequality may be strict even if **all**
points in `G` are globally Pareto-optimal. Consequently, an
evaluated Pareto-valid population `P` can satisfy
`HV(P)>HV(G)`, producing `HV(P)/HV(G)>1` without exceeding
the true mathematical front.

For example, on the original two-objective DTLZ1 front
`f_1+f_2=0.5`, the sampled endpoints `(0,0.5)` and
`(0.5,0)` have normalized HV `0.21` at reference
`(1.1,1.1)`. The midpoint `(0.25,0.25)` has HV `0.36`;
the old finite-reference ratio is `12/7`. This is an
independently checked **discretization counterexample**, not a
Hypervolume engine defect. Neither clipping values to 1 nor
projecting an already Pareto-optimal point onto the front addresses
the missing reference volume.

A nested sequence of GT samples yields nondecreasing HV under
**unchanged** normalization because of set inclusion. However,
a fresh deterministic `pf(n_points)` grid need not contain
the earlier grid. More importantly, apparent numerical convergence,
even at high density, cannot certify the undiscovered volume of
an arbitrary, disconnected or black-box Pareto front. Density
is an engineering/accuracy choice, not a mathematical certificate.

### 9.2. Current scalar absolute-HV reference specification (ADR 0058)

For `mb.metrics.hv(..., scale="abs")`, MoeaBench still
requires an **explicitly calibrated MOP-owned GT**, returns the usual
scalar `MetricMatrix`, and uses exactly the same normalization
geometry and HV backend conventions as `raw`. The `ref` parameter
chooses normalization bounds; it **does not** supply or replace the
MOP's calibrated GT. The normalized HV reference point is the vector
`(1.1,...,1.1)`, as defined by
[ADR 0020](adr/0020-standardized-reference-point-strategy-for-normalization.md).

Let `V(P)` be the raw HV of the evaluated population in the selected
fixed geometry. The absolute result is

\[
H_{\mathrm{abs}}(P)=\frac{V(P)}{D},\qquad
D=\begin{cases}
HV(PF_{\mathrm{true}}), & \text{if certified and applicable},\\
HV(G), & \text{otherwise}.
\end{cases}
\]

The exact continuous denominator is selected only if **all** of these
conditions hold: the MOP is an **original supported built-in class**
(not a subclass or instance with overridden `pf`, `ps`, or
`evaluation`); GT calibration provenance is explicitly
`analytical`; the selected ideal/nadir match the proved **true
per-axis extrema** of that precise problem configuration; and the
normalized reference geometry is the adopted 1.1 endpoint.
Otherwise, including externally provided or empirically searched GT,
legacy sidecars with unknown provenance, and different normalization
bounds, the denominator is the **finite sampled GT HV**.

The currently certified continuous denominators are:

| Original MOP / configuration | Exact normalized continuous-front HV |
|---|---|
| DTLZ1, any supported `M>=2` | `1.1^M - 1/M!` |
| DTLZ2, DTLZ3 or DTLZ4, any supported `M>=2` | `1.1^M - pi^(M/2) / (2^M Gamma(M/2+1))` |
| DPF1, exactly `M=3, D=2`, with verified projection weights/extrema | `((1.1^3 - 0.1^3)/3) + 0.1*(1.1^2)` |

The formulas describe the **complete original continuous true front**,
not its finite sampling. They cannot be extended by analogy to DTLZ5/6,
other DPF projections, constrained or mixed-dimensional fronts, or
user-defined MOPs. An analytically generated GT that lacks proof of
these conditions still uses `sampled_gt`.

The result diagnostics identify the actual denominator basis:
`hv_abs_reference_basis="continuous_exact"` or `"sampled_gt"`,
`hv_abs_denominator` (positive scalar volume), and
`hv_abs_gt_points` (calibrated finite GT size). The report exposes
the same distinction. When the numerator is **exact** and the
evaluated population is **feasible** in the certified continuous
geometry, `H_abs\le1` follows mathematically. Under the sampled
basis, 1 means equality with the **calibrated sample**, and
`H_abs>1` is permitted and not clipped. With `mode="fast"`,
the numerator is a seeded Monte Carlo estimate, so even an exact
denominator does not certify an approximate numerator. No
HV-specific `value ± uncertainty` return type is introduced.

`scale="raw"` continues to return dominated volume;
`scale="rel"` retains the competitive denominator in
[ADR 0047](adr/0047-conventional-hypervolume-relative-scaling.md),
which may legitimately be exceeded. The three scales preserve
their scalar interface. Their values are comparable across calls
only with matching normalization/basis and their appropriate
reference context. `H_abs=1` denotes a **theoretical optimum**
only in the certified case, never unconditionally.

### 9.3. Reference calibration, indicators and preservation

The finite GT remains required and useful even when `H_abs`
uses a mathematically certified continuous denominator: it is
used by other GT-dependent indicators, remains a calibrated
artifact with provenance, and must not be silently regenerated
or replaced. Updating GT may change IGD/IGD+ and the finite
fallback denominator; it does **not** authorize changing frozen
regression baselines. The exact continuous denominator is an
**indicator reference scalar**, not a substitute PF sample.

The specification of this behavior is governed by
[ADR 0058](adr/0058-scalar-absolute-hv-certified-reference.md)
together with ADRs 0018, 0020, 0043, 0047, 0056 and 0057.
Historical failures, analytic derivations, sampler-density
experiments, and rejected remedies are preserved in the
[issue #89 memorial](research/hv-issue89-research-memorial.md)
and its linked specialized research notes.

### 9.4. Read-only GT quality assessment (ADR 0059)

MoeaBench exposes a read-only `mop.assess_gt()` method and includes the
assessment in `mop.report()` when the instance has an explicitly
calibrated GT in memory. Neither operation generates a hidden
calibration, reads an on-disk sidecar on the user's behalf, writes files,
or replaces scientific references.

```python
mop = mb.mops.DTLZ2(M=3)
candidate = mop.assess_gt(n_points=128, probes=128, seed=42)
print(candidate.validity.status, candidate.coverage.value)
# 'pass' on the independent unit-sphere membership oracle;
# empirical covering statistics without a certified bound.

mop.calibrate(source_baseline="dtlz2-gt.json", n_points=2000)
assessment = mop.assess_gt()   # examines the actual calibrated GT
print(assessment.to_dict())
mop.report()                  # includes the same GT diagnostics
```

The assessment returns a structured `GTAssessment` carrying
`GTProperty` entries for: exact row count, numerical uniqueness,
independent true-front membership when an approved oracle exists,
mathematically known components/extremes, finite-probe covering
distances, nearest-neighbor spatial regularity, feasibility and
reproducibility. Each property has `status`, `value`, `basis`,
`threshold`, and `unit` fields; statuses are `pass`, `fail`,
`unverified` or `not_applicable`. The `summary()` method provides
a compact textual representation and `to_dict()` returns
machine-readable evidence.

**Current implementation limits are explicit.** Independent
objective-space analytical membership oracles are implemented for
original DTLZ1–4, DTLZ7–9 and DPF1–5. These test exact simplex,
positive unit sphere, disconnected DTLZ7 record regions, constrained
DTLZ8 line/patch geometry, DTLZ9 constrained curve and DPF1
nonnegative linear projections, as applicable. Unmodified original
class identity is required. Unsupported MOPs or custom subclasses
are deliberately `unverified` for analytical membership,
not presumed correct. No objective-only GT can independently establish decision
feasibility. Original DTLZ5/6 now accept explicit aligned
`decision_reference` rows for independent decision-bound and
equation verification (Section 3.6.10); all other unproved
decision/constraint claims remain `unverified` without a
corresponding scientific oracle.

Where an independent continuous-PF sampler is scientifically justified,
covering gaps are computed against seeded **independent true-front
probes** in the original Euclidean objective geometry. These are
**finite-sample measurements, not proven continuous Hausdorff
bounds**. For disconnected or separately characterized components,
covering reports independent probe gaps **per component**, explicitly
recording missing regions as unavailable; regularity reports
nearest-neighbor statistics within each known component rather than
combining incompatible geometric measures. On single-component
fronts, it reports ordinary objective-space nearest-neighbor
statistics. These data do not define a canonical mixed-dimensional
probability weighting.
Without independently approved, configuration/budget-specific
acceptance limits, such quality measures remain marked
`unverified` even though their numerical values are available
and comparable under matching geometry and protocol.

The initial unit battery injects incorrect off-front samples,
missing proved extrema/components, and duplicate-heavy samples,
and verifies a corresponding diagnostic failure. For original DTLZ5/6, an independent geometric audit now extracts
`g=||f||_2-1`, verifies the original feasible angular interval
and minimum attainable radial `g` for the point's angular direction,
and rejects points provably radially dominated at the same direction.
At `g=0`, **individual-point** global Pareto membership follows
from the minimum possible objective norm. For `M>=4`, even a GT
consisting exclusively of these proved points remains
`validity=unverified` as a full-original-front assessment, since
it omits the known positive-`g` Pareto regions. For `g>0`, these
are **necessary but not sufficient**
conditions for Pareto efficiency: the quality API reports
`validity=unverified` on otherwise feasible positive-`g` points,
with diagnostic counts and `g=0`/`g>0` occupancy. It does **not**
mislabel this partial characterization as global certification.

A separate optional numerical bound is now available for DTLZ5/6:
`mop.assess_gt(reference=..., dominance_margin=0.01,
dominance_max_intervals=4096)`. Its `dominance` result
reports how many objective vectors **numerically exclude**
all feasible dominating improvements of at least the stated
objective-space margin over the continuous `g` interval.
`mop.report(dominance_margin=0.01)` includes the same
opt-in diagnostic on an existing in-memory calibrated GT.
No run is triggered by default.

This establishes **neither** exact Pareto efficiency (an
improvement smaller than the margin is not ruled out) **nor**
a portable rigorous numerical proof (ordinary floating point
is used with conservative numerical slack, not verified
directed rounding). For those reasons its property status
remains `unverified` regardless of a successful finite-margin
exclusion. The original full-front validity status for
`M>=4` remains `unverified` until stronger independent
scientific evidence is accepted.

An **exact global dominance equivalence** is now available for
the original DTLZ5/6 feasible objective image: a feasible `y`
is globally Pareto-optimal if and only if there is **no**
feasible `z <= y` with `g(z) < g(y)`. This follows directly
from the exact nonnegative spherical norm
`||f||_2 = 1+g`: any strict coordinatewise improvement
must strictly decrease the norm, and vice versa under
`z <= y`. It reduces global dominance to a continuous
**one-scalar g feasibility** problem.

The read-only, opt-in `assess_gt(..., dominance_g_gap=1e-4,
dominance_g_max_intervals=8192)` checks this condition
**only for improvements in g at least the specified gap**.
The result is available as `assessment.global_dominance`
and in `mop.report(dominance_g_gap=1e-4)`. Numerical
witnesses, excluded gap intervals and unresolved attempts
are recorded separately; the default assessment does not
run this potentially costly branch-and-bound. The finite
gap and ordinary floating-point slack prevent a claim
of **exact zero-gap Pareto certification**, so
`global_dominance.status` and the original `M>=4`
overall `validity.status` remain `unverified`.
This is a stronger diagnostic than simple finite-cloud
non-dominance, but not the final scientific certificate.

New source-bound scientific qualification suites must establish
independent sufficient global membership for DTLZ5/6 `g>0`
regions, branch-aware regularity, feasibility and replay checks,
and scientifically grounded acceptance thresholds before claiming
**all** supported built-in GTs are certified. The inspection API does not override
[ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md)
or certify an exact continuous-HV denominator under
[ADR 0058](adr/0058-scalar-absolute-hv-certified-reference.md).


## 10. Historical References, Regression, and Certification

### 10.1. In what sense is a historical baseline “wrong”?

A baseline may play two different roles:

1. **Historical behavioral record:** documents what a particular version of the program actually computed. Even if that program is later recognized as mathematically incorrect, the record can remain valid **as evidence of that version's behavior**.
2. **Scientific gold standard for the intended published problem:** is only justified after the mathematical definition, generation, and numerical fidelity of the reference have been validated.

Therefore, a GT produced by an evaluator that diverges from the paper **must not continue to be represented without qualification as the correct published PF**. This does not imply that every MoeaBench baseline is wrong or that all legacy regression tests should be deleted.

Historical preservation requires **identifying the old reference with its original mathematical definition and source version**. A scientifically corrected successor should be a **new versioned, verified artifact**. Never overwrite historical evidence merely to make the test suite pass.

**An important inaugural-calibration distinction:** provisional numerical fixtures that were generated from a mathematically incorrect evaluator are not necessarily previously accepted *scientific* GT calibrations. With explicit maintainer approval and independent proof-based verification, such a fixture can be replaced by the **first accepted scientific calibration** of the corrected MOP, without pretending the defective provisional values constitute a preceding validated scientific version. Ordinary Git history remains available for technical audit. Once this first calibration is accepted, the usual frozen-reference safeguards apply. This narrowly scoped case is documented in [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md) and [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md).

### 10.2. Correcting objective evaluation and changing GT are separate interventions

Correcting **\`evaluation()\`** changes the evaluated mathematical problem and can alter solver trajectories and final outcomes. Replacing **GT** changes the reference used by indicators and can alter results even if every solver-produced objective value is frozen. These interventions therefore need independent comparisons and provenance.

Relevant architectural decisions have complementary roles:

- [ADR 0006](adr/0006-benchmarks-reimplementation.md): MOP-owned implementation and analytical PS/PF.
- [ADR 0017](adr/0017-comprehensive-calibration-and-testing-framework.md): scientific GT and independent calibration requirements.
- [ADR 0018](adr/0018-consolidated-scientific-report-and-gt-density-refinement.md): historical reference densities. ADR 0057 partially supersedes the idea of a universally fixed density, while retaining current runtime defaults until an operational policy is approved.
- [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md): separate lifecycles for numerical certification, performance baselines, and performance certificates, including immutable versioned records.
- [ADR 0057](adr/0057-proposed-purpose-specific-gt-budgets.md): MOP- and purpose-specific GT generation; explicit calibration; automatic budget selection accepted **in principle**, without a general approved numerical rule.

A historical test may check an **unchanged mathematical invariant**, or intentionally freeze numbers from a previous implementation. Its purpose must be understood before classifying a behavioral change as a correction or regression. **Every change to an existing regression test requires advance warning and explicit maintainer authorization**, and any change conflicting with an accepted ADR requires prior review and authorization.

The investigation identified sensitive references including **\`tests/stability_reference_gt.json\`**, **\`tests/stability_gt_geometry_reference.json\`**, **\`tests/unit/test_dpf3_emd_regression.py\`**, and calibrated sidecars. They serve different purposes and must not be replaced en masse. Persisted references should identify their formula, parameters, origin, and source version.

### 10.3. Calibration, certification, and FAIR are distinct

**GT is a core scientific concept and API capability in MoeaBench, independent of FAIR.** Removing experimental FAIR indicators does not remove the need for correct PS/PF, nor does it make \`calibrate()\` a method of that discontinued research.

Likewise, a **numerical certificate** establishes that a prescribed set of numerical tests and invariants passes in a specified environment; it does not automatically create a new historical reference. A **performance baseline** is an approved, versioned empirical study. Its failure is not permission to overwrite it. ADR 0056 explicitly separates these procedures.

## 11. What Can Be Proven, Measured, and Claimed

DPF3/DPF5 provide a useful hierarchy of scientific evidence:

| Claim | Required evidence | What is insufficient |
|---|---|---|
| Implementation matches the published benchmark | Independent equation comparisons, including boundary and dimensional cases | Plausible plots |
| Reference points lie on the globally optimal PF | Analytic proof or another sound global argument | Sample-relative nondominated filtering |
| Parameterization covers the entire PF | Completeness proof including all components | A necessary geometric identity alone |
| Finite GT has bounded coverage | Continuous geometric bound in the declared metric, with roundoff included when applicable | Largest gap among random probes |
| Requested count was achieved | Actual decision/objective counts and stated uniqueness semantics | The \`n_points\` input |
| Numerical precision is guaranteed | A rigorous implementation-specific error analysis or enclosure | Agreement at a handful of high-precision probes |
| Indicators are comparable | Consistent reference, documented normalization and provenance | Identical indicator names |
| Historical reference remains traceable | Immutable, versioned artifact with an explicit interpretation | Recomputing historical values using new source |

For **published DPF3**, global PF membership/completeness and an **exact-real-arithmetic angular grid coverage bound** have been demonstrated. High-precision numerical experiments measured errors of order \(10^{-14}\) in selected configurations, but they are **not** a general IEEE floating-point certificate. Tensor cost grows as \(k^{D-1}\), so this particular method may require prohibitive resources for large \(D\).

For **published DPF5**, global PF membership and completeness of **both components** have been proved. A complete angular tensor grid on **each** component now also provides a conditional **exact-real-arithmetic continuous Hausdorff covering bound**; this is distinct from empirical diagnostics of non-grid samples and from a deployed float64 error certificate. The mathematical definition alone does not prescribe a universal sample ratio between components of different intrinsic dimensions.

These distinctions delimit what can be claimed scientifically without elevating tentative hypotheses or finite-probe measurements to universal guarantees.

## 12. Scientific Requirements for a Trustworthy GT Generator

A trustworthy MoeaBench reference generator should:

1. **Use the correct mathematical equations** and feasible constraints across all supported configurations.
2. **Generate points from a characterized PS**, or explicitly justify an approximate/empirical source when an analytic solution is unavailable.
3. **Represent all relevant components and extremes** of PF, including interfaces and lower-dimensional regions.
4. **Report cardinality transparently:** requested count, actual decision rows, returned objective rows, duplicate counts, and effective numerical uniqueness.
5. **Assess coverage** in an appropriate declared objective-space metric, distinguishing proved bounds from empirical probe statistics.
6. **Control numerical errors**, including cancellation, underflow, inverse transformations, and parameterization singularities.
7. **Bound computational cost** according to purpose, dimension, and required fidelity.
8. **Be deterministic or reproducibly stochastic**, recording seeds and environment.
9. **Record GT provenance:** MOP identity and parameters, mathematical version, source revision, generator, actual effective budget, normalization, and quality assumptions.
10. **Preserve historical references** and treat evaluator corrections, sampler corrections, and calibration policy changes as independent interventions.

None of these requirements forces different MOPs to use one generator. On the contrary, **MOP geometry determines the scientific method**; shared implementation facilities are justified only when that method remains mathematically equivalent.

## 13. Mathematical Sources and Audit Traceability

**Primary mathematical source:** Zhen and collaborators, *Multiobjective Test Problems with Degenerate Pareto Fronts*, arXiv:1806.02706v2 (2022), Sections IV-C, IV-E, IV-F and Equations (15)–(16), (19)–(21). [arXiv record](https://arxiv.org/abs/1806.02706) and [author-hosted manuscript](https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf).

**Complete mathematical proofs:** [DPF3 global PS/PF](research/gt-issue119/gt-dpf3-global-pareto-proof-issue119.md); [DPF5 global PS/PF](research/gt-issue119/gt-dpf5-global-pareto-proof-issue119.md); [DPF3 angular coverage at D=2](research/gt-issue119/gt-dpf3-d2-angular-cover-proof-issue119.md); [DPF3 angular coverage at general intrinsic D](research/gt-issue119/gt-dpf3-nd-cover-and-acceptance-issue119.md).

**Implementation and precision checks:** [initial sample-cardinality and sample-dominance audit](research/gt-issue119/gt-dpf3-dpf5-audit-issue119.md); [published-formula triage](research/gt-issue119/gt-dpf-published-formula-triage-issue119.md); [general-D formula reconstruction](research/gt-issue119/gt-dpf-general-d-evidence-issue119.md); [numerical kernels and spherical invariants](research/gt-issue119/gt-dpf-numeric-kernel-evidence-issue119.md); [high-precision sweep](research/gt-issue119/gt-dpf-high-precision-sweep-issue119.md); [off-front evaluation and optional dependency isolation](research/gt-issue119/gt-dpf5-offpareto-issue119.md).

**Finite sampling and indicator impact:** [DPF5 analytical angular covering](research/gt-issue119/gt-dpf5-angular-cover-candidate-issue119.md); [DPF5 branch-conditional coverage](research/gt-issue119/gt-dpf5-finite-sampler-coverage-issue119.md); [fixed-budget Sobol comparison](research/gt-issue119/gt-dpf5-branch-sobol-comparison-issue119.md); [P/D allocation study](research/gt-issue119/gt-dpf5-branch-allocation-issue119.md); [IGD, IGD+, and HV impact](research/gt-issue119/gt-dpf5-paired-indicator-impact-issue119.md).

**Change governance and historical research artifacts:** [production impact inventory](research/gt-issue119/gt-dpf-production-impact-inventory-issue119.md), [versioned correction/migration proposal](research/gt-issue119/gt-dpf-correction-migration-proposal-issue119.md), and [research PR pre-merge audit](research/gt-issue119/gt-dpf-premerge-review-issue119.md). These specialized documents contain experiments, historical hypotheses, negative results, and migration decisions. The present reference explains the enduring principles and consolidated findings without requiring readers to follow the investigation chronologically.

**Hypervolume finite-discretization and calibration research:** [consolidated #89 memorial](research/hv-issue89-research-memorial.md) with chronology, reproducible scripts, and negative results; [minimal finite-front counterexample](research/hv-finite-reference-issue89.md); [sample-density sensitivity](research/hv-reference-density-issue89.md); [degenerate DPF1 three-objective proof](research/hv-dpf1-degenerate-issue89.md); [arbitrary-M DTLZ1/2 analytical proof](research/hv-analytic-dtlz-any-m-issue89.md); [DTLZ3/4 spherical equivalence](research/hv-dtlz234-equivalence-issue89.md); and [disconnected/degenerate GT-refinement study](research/hv-gt-refinement-issue89.md). The governing adopted scalar policy is [ADR 0058](adr/0058-scalar-absolute-hv-certified-reference.md), not a research hypothesis.

[Issue #119](https://github.com/monacofj/moeabench/issues/119) retains the discussion history. [PR #139](https://github.com/monacofj/moeabench/pull/139) contains independent reconstructions, reproducible data, new tests, and supporting reports.

---

**Governing principle:** GT must represent the **correct mathematical problem** with **coverage, numerical fidelity, and provenance appropriate for its intended purpose**. Each MOP owns its geometry and reference-generation strategy; supporting infrastructure can be shared only where its scientific validity is established.

### 3.6.12. Theorem 9 — complete zero-gap dominance for M=2 and M=3

For **original** DTLZ5 and DTLZ6 with \(M=2\) or \(M=3\),
every feasible point with \(g>0\) is globally dominated by
a feasible point on the \(g=0\) Pareto curve. Hence, for
these objective counts, the **entire mathematical Pareto
front** is exactly the \(g=0\) curve. This statement does
not extend to the original \(M\ge4\) fronts.

**Proof for \(M=2\).** Every feasible point is
\((1+g)(\cos\theta_1,\sin\theta_1)\). The feasible
\(g=0\) point with identical \(\theta_1\) has both
objectives no larger and a strictly smaller norm
when \(g>0\), so it strictly dominates the original
point.

**Proof for \(M=3\).** Write \(r=1+g>1\),
\(t=\theta_1\), and \(s=\theta_2\).
The feasible objective vector is

\[
y=r(\cos t\cos s,\ \cos t\sin s,\ \sin t).
\]

The angular constraint gives
\(s\in[a(g),\pi/2-a(g)]\), with
\(a(g)=\pi/(4r)\). Thus
\(\sin s,\cos s\ge\sin(\pi/(4r))\).
On \([0,\pi/4]\), sine is concave and vanishes at
zero. Taking the chord between zero and \(\pi/4\),

\[
\sin\!\left(\frac{\pi}{4r}\right)
\ge \frac1r\sin\!\left(\frac\pi4\right)
=\frac{1}{r\sqrt2}.
\tag{DTLZ56-15}
\]

Consequently \(y_1,y_2\ge\cos t/\sqrt2\)
and \(y_3=r\sin t\ge\sin t\).
The \(g=0\) curve point

\[
z=(\cos t/\sqrt2,\ \cos t/\sqrt2,\ \sin t)
\]

is feasible and satisfies \(z\le y\).
Since \(\lVert z\rVert_2=1<r=\lVert y\rVert_2\),
the two vectors cannot be identical: \(z\)
strictly dominates \(y\), including all boundary
angles and arbitrarily small positive \(g\).
Combined with Theorem 2, this establishes the
claimed **necessary and sufficient mathematical
membership characterization** for \(M=2,3\).
\(\square\)

**Independent numerical diagnostic.**
[`_dtlz56_lowdim_curve.py`](../moeabench/mops/_dtlz56_lowdim_curve.py)
constructs the same-\(\theta_1\), \(g=0\) vector from
the objective norm and last coordinate without
calling the production PS/PF, evaluator or scalarizer.
It checks original feasible-image conditions,
componentwise numerical dominance and strictly
smaller norm. This helper is not yet wired into the
public `assess_gt()` result; it is a new directly
tested independent research diagnostic. Its numerical
`witness` status is provisional until ADR 0059's
three source/environment/result gates are satisfied.
Near-boundary cases remain `unverified` instead
of receiving unsupported floating-point certification.

**Remaining global problem:** For \(M\ge4\),
genuine \(g>0\) Pareto-optimal regions exist. Neither
this low-dimensional theorem nor its implementation
supplies the required exhaustive zero-gap nondominance
oracle for those regions. Their scientific certification,
finite-GT covering quality and indicator suitability
remain open under #175/#176.

### 3.6.13. Objective-only reconstruction for M>=4 local-dominance diagnostics

A saved objective-space reference does not normally retain an aligned
decision-space witness. For an original DTLZ5/6 objective vector
\(y\) with independently checked feasible image, the radial
parameter is \(g=\lVert y\rVert_2-1\). Away from angular
coordinate singularities, the published hyperspherical equations
recover \(\theta_1,\ldots,\theta_{M-1}\) by inverse `atan2`
coordinates. For \(g>0\), the original position variables follow

\[
x_1=2\theta_1/\pi,\qquad
x_i=\frac{4(1+g)\theta_i/\pi-1}{2g},
\quad 2\le i\le M-1.
\tag{DTLZ56-16}
\]

For a genuinely indeterminate inner angle (exact zero
numerator and subradius), \(\pi/4\) supplies a feasible
representative; the reconstructed decision then has
\(x_i=1/2\). The distance variables independently realize
\(g\) using the formulas in Section 3.6.1. The resulting
decision must be **re-evaluated by independent equations**
and compared row-for-row to the supplied objective vector;
the feasibility check alone is not sufficient to accept
floating-point inversion.

The research helper
[`_dtlz56_objective_descent.py`](../moeabench/mops/_dtlz56_objective_descent.py)
implements this checked reconstruction and then runs the
existing independent first-order local dominance oracle.
This gives a way to seek **negative membership evidence**
from an objective-only GT, without assuming that the
production `ps()` or `pf()` can supply an aligned witness.
A returned numerical dominance witness is provisional
under ADR 0059; no witness or an ill-conditioned inversion
returns `unverified`.

The helper is also available as an **explicit, opt-in public diagnostic**
through `assess_gt(..., objective_only_local_descent=True)`
and `mop.report(objective_only_local_descent=True)` when a calibrated
GT is already present. It records numerical witnesses in
`local_descent` without writing or changing any GT. This is **not**
a globally sufficient positive-membership test, and **not** a
certificate of any \(M\ge4\) positive-\(g\) front region. Complete zero-gap nondominance,
cross-platform numerical verification, and finite coverage
remain separate #175/#176 acceptance gates.

## Provisional evidence capture for expensive GT experiments

The lightweight command-line helper
[`scripts/audit/gt_evidence_manifest.py`](../scripts/audit/gt_evidence_manifest.py)
can record a reproducible **identity manifest** beside an expensive
locally generated experiment artifact. The helper deliberately does
not implement or claim the three scientific certification gates of
ADR 0059.

The input configuration is a JSON object with at least
`problem`, `parameters`, `seed`, `budget`, `oracle` and
`tolerance`. Include normalization, component definitions, reference
identity, parallelism and any other scientifically relevant settings
under `parameters` for a real experiment. The artifact may be a
binary NumPy result or another immutable output file.

```bash
python scripts/audit/gt_evidence_manifest.py capture \
  --artifact gt-results.npz \
  --config gt-config.json \
  --manifest gt-results.manifest.json

python scripts/audit/gt_evidence_manifest.py verify \
  --artifact gt-results.npz \
  --manifest gt-results.manifest.json \
  --expected-commit FULL_40_CHARACTER_COMMIT_SHA
```

Capture rejects a dirty tracked Git working tree and refuses to overwrite an existing manifest, output artifact or configuration file. It records the
full commit, Python/platform information, installed dependency
versions, NumPy build/BLAS configuration when available,
experiment configuration, artifact digest and canonical manifest
digest. Verification checks recorded manifest integrity and artifact
identity (optionally against an externally supplied expected SHA).
A matching manifest is only evidence of **byte identity and
recorded provenance**, not of mathematical validity, actual
environment certification or equivalent results on a second
platform. The tool leaves `mathematical_oracle`,
`execution_environment` and `result_reproducibility`
explicitly `unverified` and rejects attempts to promote
those fields to `pass` using this helper.

To obtain a scientific GT certificate, the same source-bound
artifact must separately pass the independent mathematical
oracle, the numerical stability controls on its actual execution
environment, and the verified-result/cross-platform comparison
specified by ADR 0059. Certification decisions belong in a
separately reviewed acceptance record, not in this hash manifest.

Large 2,000/10,000-point GT quality matrices and multi-budget
indicator sensitivity runs may benefit from running locally,
but **do not start those until the specific MOP configurations,
independent oracle, thresholds and acceptance artifact are
fixed**. Otherwise heavy computations would produce
reproducible but scientifically inconclusive data.

