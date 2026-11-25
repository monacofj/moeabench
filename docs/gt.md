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

### 3.6.2a. Constructive exact-rational near-endpoint same-ray descent

The necessary minimum-ray-radius condition above admits a **fully
constructive exact-rational certificate**, without a numerical
dominance search or a fixed positive improvement margin.

Let \(u=(u_1,\ldots,u_{M-1})\) be original feasible DTLZ5/6 position
variables, and let \(g>0\) be the independently feasible original
distance value. For each constrained inner angle \(i\ge2\), define

\[
d_i=\frac{g(2u_i-1)}{1+g},\qquad
q=\max_{i\ge2}|d_i|,\qquad
g_{\min}=\frac{q}{1-q},
\tag{DTLZ56-3a}
\]

with \(q=0\) when \(M=2\). If \(g_{\min}<g\), take
\(g'=(g+g_{\min})/2\), and define

\[
u_1'=u_1,\qquad
u_i'=\frac12+\frac{d_i(1+g')}{2g'},\quad i\ge2.
\tag{DTLZ56-3b}
\]

Then \(0\le u_i'\le1\) follows directly from
\(|d_i|\le g'/(1+g')\), and \(0<g'<g\).
The original angular equation gives

\[
\frac{1+2g'u_i'}{1+g'}
=\frac{1+2gu_i}{1+g}
\quad\text{for every }i\ge2.
\]

The first angle also remains unchanged. Hence the new point has the
**same original angular direction**, and its objective vector is exactly

\[
F(g',u')=\frac{1+g'}{1+g}\,F(g,u).
\tag{DTLZ56-3c}
\]

All coordinates are nonnegative, at least one is strictly positive, and
the scale is strictly less than one. Thus the constructed point
**strictly dominates** the original point, for arbitrarily small
positive \(g\) and without a prescribed positive `g` gap.
Both original distance functions realize the smaller
\(g'\in[0,g_{\max}]\), so this is a feasible dominator.
If any constrained inner position is at \(0\) or \(1\), then
\(g_{\min}=g\) may hold and this *same-ray method* abstains:
it does not claim global Pareto efficiency at such a boundary.

The independent helper
[`scripts/audit/dtlz56_exact_same_ray.py`](../scripts/audit/dtlz56_exact_same_ray.py)
implements these algebraic steps using exact Python `Fraction` values,
returning the rational witness positions, smaller radial distance and
objective scale in lossless hexadecimal form. Its replay function
reconstructs the same certificate and verifies every angular equality
and strict radial reduction. No floating-point trigonometry, numerical
optimizer or production GT generator is used. Supplied numerical inputs
are interpreted as exact rational declarations; an external objective
GT row still requires separate *independent feasibility and alignment*
evidence before this conditional theorem can be used as a scientific
claim about that row. The helper therefore leaves
`scientific_certification=unverified` in its output.

This settles a **negative-membership subset** of the open-endpoint
problem; the hard positive-membership problem for \(M\ge4\) remains on
the minimal-ray-radius boundary and requires excluding *different-ray*
dominators as \(g'\uparrow g\). Theorem 1 already identifies this
necessary boundary; this corollary adds an exactly replayable constructive
witness and zero-gap tests, not a general converse.

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

### 3.6.3a. Theorem — a certified positive-g global PF stratum for M>=4

The original DTLZ5 and DTLZ6 problems admit an explicitly
characterized **globally Pareto-optimal positive-\(g\) stratum** for
every \(M\ge4\), not merely numerical evidence of positive-\(g\)
front existence.

Fix any \(0<g_y\le g_{\max}\) and any valid original position
variables \(u=(x_1,\ldots,x_{M-1})\) such that

\[
x_1=0,\qquad x_2=1,\qquad x_3=0,
\tag{DTLZ56-PF1}
\]

with the remaining \(x_i\in[0,1]\) arbitrary. Let
\(y=F(g_y,u)\). Then **\(y\) is globally Pareto-optimal**.

**Proof.** By the original angular equations,
\(\theta_1=0\),
\(\theta_2=b(g_y)=\pi/2-a(g_y)\) and
\(\theta_3=a(g_y)\), where
\(a(g)=\pi/[4(1+g)]\).
The candidate's last objective is \(y_M=0\),
and its \((M-2)\)-th objective is exactly

\[
y_{M-2}
=(1+g_y)\cos\theta_2\sin\theta_3
=(1+g_y)\sin^2 a(g_y)
=:h(g_y).
\tag{DTLZ56-PF2}
\]

Suppose another original feasible vector \(z\le y\)
strictly dominates \(y\). The nonnegative spherical
norm law (DTLZ56-1) implies that the dominator has
\(g_z<g_y\). Because \(z_M\le y_M=0\),
it must have \(z_M=0\), forcing
\(\theta_1(z)=0\). For every \(g\ge0\)
and every feasible pair of inner angles,
\(\cos\theta_2\ge\sin a(g)\) and
\(\sin\theta_3\ge\sin a(g)\). Consequently

\[
z_{M-2}\ge
(1+g_z)\sin^2 a(g_z)=h(g_z).
\tag{DTLZ56-PF3}
\]

It remains to establish that \(h\) decreases
**strictly** on the entire original \(g\ge0\)
domain. Write \(r=1+g\ge1\) and
\(t=\pi/(4r)\in(0,\pi/4]\).
Direct differentiation yields

\[
\frac{dh}{dr}
=\sin t\,(\sin t-2t\cos t).
\tag{DTLZ56-PF4}
\]

The tangent function is convex on
\([0,\pi/4]\) and lies below the chord joining
its endpoint values \(0\) and \(1\).
Thus \(\tan t\le4t/\pi<2t\) for \(t>0\);
hence \(dh/dr<0\). Since \(g_z<g_y\),

\[
z_{M-2}\ge h(g_z)>h(g_y)=y_{M-2},
\]

contradicting \(z\le y\). Therefore no
feasible strict dominator exists. \(\square\)

This is a **sufficient global positive-membership
theorem** for a specific nontrivial stratum;
it does not classify every original positive-\(g\)
boundary point. It also applies at \(g_y=0\)
by Theorem 2, although the positive-\(g\)
statement concerns \(g_y>0\).

The research helper
[`scripts/audit/dtlz56_positive_g_global_pf.py`](../scripts/audit/dtlz56_positive_g_global_pf.py)
checks the exact rational declarations
\(x_1=0,x_2=1,x_3=0\), original
distance range, problem identity and dimension,
and emits a lossless replayable symbolic theorem
record. It does not compute a floating-point
global optimum or invoke the production GT
generator or evaluator. New independent tests
cover original DTLZ5/6 for \(M=4,5,10\),
arbitrarily small positive rational \(g\),
rational endpoints and abstention outside
this stratum.

**Scientific acceptance separation.** The proof
establishes global membership for a *genuinely
feasible and aligned original candidate with
these exact positions*. An objective-only
floating-point GT row must first have
independent source/decision alignment and
feasibility verification before this theorem
can be applied to it. The helper does not
promote the public `assess_gt()` status,
certify a finite GT's component coverage,
or bypass the environment/result gates of
ADR 0059. The full positive-\(g\)
membership problem for arbitrary \(M\ge4\)
points remains open.

### 3.6.3b. Generalized theorem — global positive-g PF strata for all supporting objectives

The positive-\(g\) global-membership argument in Theorem
3.6.3a extends beyond the single \((M-2)\)-th
objective. For every original DTLZ5/6 problem with
\(M\ge4\), each objective index
\(j\in\{1,\ldots,M-2\}\) supplies a separate
**sufficient global Pareto-membership stratum**.

Let \(g_y\in(0,g_{\max}]\) and let
\(y=F(g_y,x)\) be an independently feasible
original decision. Require \(x_1=0\) and:

* For the first objective (\(j=1\)), set
  \(x_2=\cdots=x_{M-1}=1\). Then
  \(p_j=M-2\).
* For \(2\le j\le M-2\), set
  \(x_2=\cdots=x_{M-j}=1\) and
  \(x_{M-j+1}=0\), leaving every subsequent
  position coordinate unrestricted. Then
  \(p_j=M-j\).

All listed exponents satisfy \(p_j\ge2\).
For either pattern, the original hyperspherical
product equations give

\[
y_M=0,\qquad
y_j=h_{p_j}(g_y),\qquad
h_p(g)=(1+g)\sin^p\!\left(\frac{\pi}{4(1+g)}\right).
\tag{DTLZ56-PF5}
\]

**Proof of global nondominance.** Suppose a
feasible \(z\le y\) strictly dominates \(y\).
The nonnegative norm identity requires
\(g_z<g_y\); and \(y_M=0\) forces
\(\theta_1(z)=0\).
At any original feasible \(g\ge0\),
every constrained cosine factor satisfies
\(\cos\theta_i\ge\sin a(g)\), and every
constrained sine factor satisfies
\(\sin\theta_i\ge\sin a(g)\), where
\(a(g)=\pi/[4(1+g)]\). The \(j\)-th
objective therefore obeys

\[
z_j\ge h_{p_j}(g_z).
\tag{DTLZ56-PF6}
\]

Writing \(r=1+g\) and \(t=\pi/(4r)\),
direct differentiation gives

\[
\frac{d}{dr}\bigl(r\sin^p t\bigr)
=\sin^{p-1}t\,(\sin t-p\,t\cos t).
\tag{DTLZ56-PF7}
\]

For \(0<t\le\pi/4\), convexity of tangent
yields \(\tan t\le4t/\pi<2t\le p\,t\).
Thus \(h_p\) is strictly decreasing for
every \(p\ge2\). In particular,

\[
z_j\ge h_{p_j}(g_z)>
h_{p_j}(g_y)=y_j,
\]

contradicting \(z\le y\). Therefore \(y\)
is globally Pareto-optimal. \(\square\)

The previous theorem with
\(x_1=0,x_2=1,x_3=0\) is recovered by
taking \(j=M-2\) and \(p_j=2\).
For \(j=1\), the theorem supplies an
additional all-upper-angle stratum not
covered by that construction. It gives
a family for every other
\(j=2,\ldots,M-3\), including
higher-order powers \(\sin^p a(g)\).
The range of admissible \(g\) is the
full original range; no positive
radial gap, sampling density or
numerical tangent argument is assumed.

The independent research helper
[`scripts/audit/dtlz56_generalized_positive_g_pf.py`](../scripts/audit/dtlz56_generalized_positive_g_pf.py)
checks exact rational position declarations,
reports the supporting objective and sine
power, and records a replayable symbolic
theorem identifier. Tests independently
evaluate the original hyperspherical
equations for DTLZ5/6 with
\(M=4,5,7,10\), all eligible supporting
objectives, very small positive \(g\),
and rational invalid/boundary examples.

This is a **set of sufficient strata**,
not a complete positive-\(g\) front
characterization. A stored objective-only
GT point must first establish independently
that its claimed decision coordinates
and \(g\) really produce that point.
Theorem application does not automatically
pass ADR 0059 method, environment or
result gates, GT component coverage,
or indicator fitness. No default GT
generator or public `assess_gt()` status
is changed.

### 3.6.3c. A nonzero first-angle interval of globally optimal positive-g points

The support construction of Section 3.6.3b can be
extended from the hyperplane \(x_1=0\) to a
**nonzero interval of first-angle positions**, with
a fully global, sufficient non-dominance condition.

Fix an original DTLZ5/6 problem with \(M\ge4\).
Take an independently feasible candidate at radial
\(r_y=1+g_y>1\), where its inner-angle position
variables satisfy one of the support patterns
(DTLZ56-PF5) for an objective
\(j\in\{1,\ldots,M-2\}\) with \(p=p_j\ge2\).
The first angle is now
\(\theta_1=\pi x_1/2\), with \(x_1\ge0\)
**not required to vanish**. Let

\[
c=y_M=r_y\sin\theta_1,\qquad
a(r)=\frac{\pi}{4r}.
\]

The supporting objective of the candidate equals

\[
y_j=\sqrt{r_y^2-c^2}\,\sin^p a(r_y).
\tag{DTLZ56-PF8}
\]

For any hypothetical strictly dominating feasible
\(z\), nonnegative norm ordering gives
\(1\le r_z<r_y\), while the last objective cap
requires \(0\le z_M\le c\). If \(c<1\), then

\[
z_j\ge \sqrt{r_z^2-c^2}\,\sin^p a(r_z)
=: H_p(r_z;c).
\tag{DTLZ56-PF9}
\]

The inequality follows because the candidate's
\(j\)-th objective consists of \(\cos\theta_1\)
and precisely \(p\) constrained inner-angle
sine/cosine factors; each such factor is at
least \(\sin a(r_z)\). Moreover
\(r_z\cos\theta_1(z)=\sqrt{r_z^2-z_M^2}
\ge\sqrt{r_z^2-c^2}\).

**Strict global monotonicity condition.** With
\(t=\pi/(4r)\), logarithmic differentiation
of \(H_p(r;c)\) gives

\[
r\frac{\partial}{\partial r}\log H_p(r;c)
=\frac{r^2}{r^2-c^2}-p\,t\cot t.
\tag{DTLZ56-PF10}
\]

For every \(r\ge1\) and \(0\le c<1\),

\[
\frac{r^2}{r^2-c^2}\le\frac{1}{1-c^2},
\qquad
p\,t\cot t\ge \frac{p\pi}{4}.
\tag{DTLZ56-PF11}
\]

The second inequality follows from the fact
that \(t\cot t\) decreases in \(t\) on
\((0,\pi/4]\), while \(t=\pi/(4r)\le\pi/4\).
Consequently, the **strict sufficient condition**

\[
\boxed{\quad c^2<1-\frac{4}{p\pi}\quad}
\tag{DTLZ56-PF12}
\]

makes \(H_p(r;c)\) strictly decreasing on the
**entire domain \(r\ge1\)**. Thus every
hypothetical dominator obeys

\[
z_j\ge H_p(r_z;c)>H_p(r_y;c)=y_j,
\]

a contradiction. Every candidate meeting the
support pattern and (DTLZ56-PF12) is globally
Pareto-optimal for its original problem, even
with \(x_1>0\), for the **entire open range**
\(0\le g_z<g_y\). The result also contains the
previous \(x_1=0\) strata as a special case.

**Exact-rational implementation without approximate
trigonometry.** The independent helper
[`scripts/audit/dtlz56_positive_g_open_first_angle.py`](../scripts/audit/dtlz56_positive_g_open_first_angle.py)
uses exact rational declarations of \(x_1\)
and \(g_y\), the original support-pattern
checker, and certified rational bounds
\(\pi_{\rm lo}<\pi<\pi_{\rm hi}\) from Machin's
identity. Since \(0\le\sin\theta_1\le\theta_1\),

\[
c^2\le B^2,\qquad
B=\frac{r_y\pi_{\rm hi}x_1}{2}.
\]

The machine-readable proof is accepted only if

\[
\boxed{\quad B^2<
1-\frac{4}{p\pi_{\rm lo}}\quad,}
\tag{DTLZ56-PF13}
\]

which is **strictly stronger** than
(DTLZ56-PF12), and is decided through
exact rational arithmetic. The lossless
hexadecimal certificate records the rational
strict margin and all supplied positions.
An independent replay procedure recomputes
the complete inequality. A failure to satisfy
the conservative bound returns `unverified`:
the candidate may still be Pareto-optimal and
the result must never be interpreted as
a proof of domination.

New tests check nonzero \(x_1\) for all
supporting objectives at \(M=4,5,7,10\),
both original distance functions, very small
positive \(g\), strict global bound behavior,
tampering, invalid inputs and safe abstention.
The independent finite-trigonometric sanity
checks are not premises of the proof.

As for earlier theorems, scientific
application to a stored **objective-only**
GT row requires separately verified
original decision-to-objective alignment,
feasible \(g\), source/environment/result
provenance and adequate finite-reference
coverage under ADR 0059. This is a
sufficient global membership certificate for
a **wider positive-dimensional subset**
of the original front, not a complete
characterization for arbitrary \(M\ge4\).

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

**Bounded, replayable continuous-domain partitions.**
The independent
[`dtlz56_rational_partition.py`](../scripts/audit/dtlz56_rational_partition.py)
builds a binary partition of a **specified closed**
\(g\)-interval with exact rational endpoints.
Each excluded leaf stores a separate strict-rational
certificate from the preceding oracle; unresolved
leaves, including those caused by the explicit node
or depth budget, retain their exact endpoints.
Because adjacent leaves share their endpoints and
their union covers the specified input range,
**a complete partition of excluded leaves proves
the absence of dominated feasible points throughout
that closed interval for the exact supplied objective
caps**. A separate replay method recomputes every
strict inequality from those caps and verifies
continuous endpoint coverage rather than trusting
the stored exclusion labels or margins.

Rational split endpoints are retained exactly
(in particular, not converted to binary floats
when the denominator becomes large). Their
serialized form uses hexadecimal numerator
and denominator. The implementation has finite
node and recursion-depth budgets and never
interprets exhaustion or unresolved leaves as
an absence of possible dominators. It preserves
the original distinction between a mathematical
exclusion of the examined domain and a certified
**GT membership** result.

**Open-endpoint limitation:** even a replay-verified
exclusion of \([0,g_y-\delta]\) does not prove the
absence of dominators for \(g_y-\delta<g<g_y\).
Choosing \(\delta\) smaller or increasing the
partition budget is not a substitute for an exact
endpoint argument. The routine therefore exposes
only `rigorously_excluded_closed_range` or
`unverified`; it never emits a global Pareto
membership `pass`. Candidate feasibility,
sample coverage and all ADR 0059 result/environment
gates remain separate obligations.

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

### 3.6.7a. Exact-rational decision for complete g=0-curve dominance

The complete-curve criterion (DTLZ56-9) can be decided
**without a numerical square root, trigonometric evaluation,
or a nonzero tolerance**, when the input objective caps are
specified as rational numbers.

Let \(c=\min(y_M,1)\), and assume all \(y_j\ge0\).
Since the two sides of (DTLZ56-9) are nonnegative,
the necessary-and-sufficient conditions are exactly

\[
 y_j^2\ge a_j^2(1-c^2),
 \qquad j=1,\ldots,M-1,
 \tag{DTLZ56-9a}
\]

where \(a_1^2=2^{-(M-2)}\) and
\(a_j^2=2^{-(M-j)}\) for \(2\le j\le M-1\).
Every coefficient is dyadic rational. Hence the predicate
requires only exact rational products and comparisons.

If all inequalities hold and \(\|y\|_2^2>1\),
the constructed \(g=0\) vector has unit norm and
is componentwise no larger than \(y\); it must be
strictly different from \(y\) and therefore **strictly
dominates** it. This applies to angular-boundary
candidates and a dominating direction different from
the candidate's original ray. If \(\|y\|_2^2=1\),
a curve fit gives only weak dominance and cannot be
promoted to strict membership rejection. If any
inequality fails, the **entire \(g=0\) curve** is
excluded as a dominator, but a positive-\(g\)
dominator may still exist for original \(M\ge4\).

The independent helper
[`scripts/audit/dtlz56_exact_g0_curve.py`](../scripts/audit/dtlz56_exact_g0_curve.py)
implements this predicate with `Fraction` and returns
a losslessly encoded rational comparison and a
replayable result. Ordinary float inputs are
interpreted as their exact binary-rational values.
The test is mathematically exact **for those values**,
including arbitrarily small rational strict
separations. As with all objective-only tests, an
actual GT row must separately pass the original
feasible-image/decision alignment oracle before
a conditional dominance result can justify a
scientific rejection. The helper does not change
the public assessment's scientific status.

The **still-unsolved** part of the open-endpoint
problem comprises the candidates on minimal
same-ray radial boundaries for which the complete
\(g=0\) curve does not dominate them. A globally
sufficient oracle must also decide possible
different-ray dominators with \(0<g'<g_y\),
including as \(g'\uparrow g_y\).

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

### 3.6.11a. Exact-rational first-order boundary dominance certificates

The differential sufficiency theorem (DTLZ56-13) also admits
**machine-checkable strict rational certificates** for rational
position variables \(u\), distance \(g>0\), and one-sided rational
velocity \(v\). This is useful specifically at the
\(g=g_{\min}(\text{ray})\) angular boundary where
the exact same-ray witness necessarily abstains.

For the original hyperspherical product
\(F_j(g,u)=(1+g)\prod_i h_{ji}(\theta_i)\),
each factor \(h_{ji}\) is either \(\sin\theta_i\),
\(\cos\theta_i\), or \(1\). Along the feasible path
\(u(g')=u+(g'-g)v\), the angle derivatives are

\[
\theta_1'=\frac{\pi v_1}{2},\qquad
\theta_i'=\pi\left(
\frac{2u_i-1}{4(1+g)^2}
+\frac{gv_i}{2(1+g)}
\right),\quad i\ge2.
\tag{DTLZ56-14a}
\]

Product differentiation expresses each \(dF_j/dg\)
using sums and products of \(\sin\theta_i\),
\(\cos\theta_i\), rational quantities and \(\pi\).
Machin rational bounds enclose \(\pi\) exactly;
alternating Taylor sums and monotonicity of
sine and cosine on \([0,\pi/2]\) give rational
lower and upper bounds for each trigonometric
factor. Applying interval arithmetic to the
product derivative yields a rational enclosure
\([\underline d_j,\overline d_j]\) containing
the exact \(dF_j/dg\).

If the prescribed \(v\) obeys the one-sided
box constraints (DTLZ56-12) and all
\(\underline d_j>0\), the strict differential
sufficiency theorem proves that **some feasible
\(g'<g\), arbitrarily close to \(g\), strictly
improves every objective**. This establishes a
zero-gap local non-Pareto-membership proof without
needing to construct a numerically separated
finite-distance witness.

The independent helper
[`scripts/audit/dtlz56_rational_tangent.py`](../scripts/audit/dtlz56_rational_tangent.py)
accepts a rational \(u,g,v\), calculates exact
outward enclosures for the complete original
derivatives, records exact hex-rational margins
and independently replays a certificate.
New tests cover M=4 angular-boundary points
with exact rational directions (including
one or multiple active inner-angle bounds),
finite checks against separately implemented
original spherical equations, and safe
abstention when derivatives are not all
strictly positive.

A **nonpositive or undecidable** derivative
enclosure must return `unverified`. Failure to
find such a direction does not prove
Pareto optimality, because possible dominators
may follow curved paths with higher-order
improvement or arise nonlocally. The supplied
candidate's original feasible-image
correspondence remains a separate requirement,
and public `assess_gt()` and ADR 0059
certification statuses are unchanged.
The global positive-membership problem for
original M>=4 remains open.

### 3.6.11b. Bounded independent discovery of certified tangents

The rigorous first-order derivative certificate of
Section 3.6.11a needs a user-supplied feasible
one-sided rational velocity. A small deterministic
search can find such a direction without trusting
a numerical LP solver or adopting its approximate
derivative output as proof.

The standalone helper
[`dtlz56_rational_tangent_search.py`](../scripts/audit/dtlz56_rational_tangent_search.py)
enumerates a fixed sequence of low-complexity rational
directions (first-angle moves and selected individual
inner-angle moves), skips those violating the original
one-sided box constraints, and sends each feasible
candidate to the **independent exact-rational
Machin/Taylor derivative verifier**. Search cost is
bounded by `max_candidates` (default 16; hard cap 128).
Every accepted witness carries its exact rational
velocity and complete lower/upper derivative certificate.
Replaying it recomputes the enumeration and checks the
underlying proof. This is an opt-in research diagnostic
and does not change GT calibration or public assessment.

This deterministic search can discover zero-gap
different-ray domination of some original
DTLZ5/6 angular-boundary candidates. Nevertheless,
**absence of a found witness is unverified**, even
when the finite search exhausts every candidate.
The enumerated directions are not exhaustive;
strict first-order tangent certificates themselves
are only *sufficient* for negative membership.
The unresolved global-positive-membership problem
includes nonlocal, higher-order and tangential
dominators, as well as full scientifically certified
feasibility and finite reference adequacy. No
ADR 0059 certification gate is promoted by this search.

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

## Paired original DTLZ5/6 GT evidence: independent alignment preflight

The original **M≥4 DTLZ5/6** `ps()` implementation uses
globally bounded **numerical weighted Chebyshev scalarization** and
greedy objective-space sampling to approximate the original
positive-\(g\) front. Its scalarization tolerance is finite.
Consequently, a point produced by `ps()` and `pf()` should
not be assumed to possess a *mathematically exact global PF
membership certificate*. The exact positive-membership
theorems in Section 3.6 instead apply to independently
verified **specific original decision strata**.

The current `mop.calibrate()` `mop_gt_v1` sidecar stores
the **objective-only** `gt_reference`, source label and
dimensions. It does not contain the decision vectors that
produced its rows. The separate `decision_reference` option
in `mop.assess_gt()` can inspect an explicitly supplied
pair, but it cannot recover the original generating decisions
from a pre-existing objective-only sidecar. Generating
`mop.ps(n_points)` again merely supplies a new candidate;
without explicit row alignment it must not be presented
as proof that it produced the archived sidecar.

For source-bound research **without changing either public
calibration or the existing sidecar schema**, use the
independent provisional artifact helper
[`scripts/audit/dtlz56_paired_gt_evidence.py`](../scripts/audit/dtlz56_paired_gt_evidence.py).
Its `sample` command invokes the original `ps()` **once**,
evaluates those decisions with the original production
MOP for capture, and stores both arrays in one
immutable-on-create NPZ bundle:

```bash
python scripts/audit/dtlz56_paired_gt_evidence.py sample \
  --problem DTLZ5 --m 4 --k 4 --points 2000 \
  --bundle dtlz5-M4-paired.npz

python scripts/audit/dtlz56_paired_gt_evidence.py verify \
  --problem DTLZ5 --m 4 --k 4 \
  --bundle dtlz5-M4-paired.npz \
  --expected-sha256 ARTIFACT_SHA256
```

The verifier computes the original DTLZ5/6 position
angles and distance function **independently of the
production MOP evaluation**, checks every decision
against \([0,1]^N\), compares each objective row
against the corresponding independently recomputed
one, and optionally compares the same ordered
objective matrix against an already existing
`mop_gt_v1` sidecar using `--sidecar PATH`.
Any mismatched row, dimension, original problem,
artifact digest or existing-sidecar ordering is
reported as a failed alignment, not silently repaired.
The sampling command refuses to overwrite an
existing candidate bundle or sidecar.

A complete **source- and environment-bound** record
must additionally apply
[`gt_evidence_manifest.py`](../scripts/audit/gt_evidence_manifest.py)
to the produced NPZ, with exact seed, budget,
M, K, algorithm, original evaluator and tolerance
in the configuration. Neither program, alone
or together, proves exact mathematical Pareto
membership of numerically scalarized positive-\(g\)
points, finite GT coverage or the complete
method/environment/result certification gates of
ADR 0059. Those remain `unverified`.

**Cost and location:** generating the current
full 2,000-point DTLZ5/6 samples can involve
thousands of globally bounded scalarizations
plus an objective-space farthest-point selection.
It should be performed locally **only after**
fixing the precise M/K cases, computation budget,
source commit and acceptance thresholds. Do not
start that expense merely to repeat objective-only
calibration. The unit tests for the paired evidence
helper use small explicitly constructed decision
matrices and therefore never invoke the expensive
sampler.

## Conditional per-row mathematical triage of paired GT evidence

The independent paired candidate artifact (above) makes it possible
to ask a narrower and more informative question than a whole-GT
`pass`: **which exact decision vectors underlying the stored
floating-point objective rows fall into a mathematically proved
original DTLZ5/6 stratum?**

The read-only research helper
[`dtlz56_paired_symbolic_audit.py`](../scripts/audit/dtlz56_paired_symbolic_audit.py)
takes an already captured paired NPZ and emits a per-row ledger.
It **never** invokes `ps()`, `pf()` or any production MOP
evaluator and does not modify the bundle:

```bash
python -m scripts.audit.dtlz56_paired_symbolic_audit \
  --bundle dtlz5-M4-paired.npz --problem DTLZ5 --m 4 --k 4 \
  --expected-sha256 ARTIFACT_SHA256
```

Each row can be marked
`symbolic_decision_global_pf` (the unit-radius norm theorem
or the exact supporting-stratum/open-first-angle sufficient
theorems), `symbolic_decision_dominated` (the exact
same-ray construction), or `unverified` (no applicable
mathematical proof, uncertain exact radial value, or failed
objective-row numerical alignment). The ledger reports counts,
row indices, theorem identifiers and exact rational `g`
when demonstrable. It does **not** classify unverified rows
as globally efficient or dominated.

Crucially, a stored floating-point **decision coordinate**
is interpreted as the precise binary rational it encodes.
The original DTLZ5 squared-distance function can then be
recomputed with exact rational arithmetic. DTLZ6 uses tenth
roots of tail variables; the helper requires each rational
tail to be a provable perfect tenth power of a rational
number (using integer root tests). Otherwise, the
radial value is **unverified**, not silently replaced by
a float. For instance, the exact dyadic \(1/1024\)
has tenth root \(1/2\); a stored `0.5`
does not have a rational tenth root.

**The ledger is a conditional decision-image classification,
not a certification of stored floating-point objective
coordinates.** Even when an independent numeric evaluation
agrees with the saved objective row within tolerance,
the stored float is not generally *exactly equal*
to the mathematically defined point involving
transcendental angular functions. Such an observation
is reported separately from the symbolic result.
Accordingly, the whole-reference membership, finite
coverage, and scientific acceptance fields remain
`unverified`. The helper also does not change the
existing public `assess_gt()` semantics or the
calibration sidecar schema.

There is a further sampling implication. The default
DTLZ5/6 M>=4 sampler computes positions using
floating-point scalarization; it need not return
**exact** inner boundary positions required by
the supporting-stratum theorems. A low proportion
of provable rows therefore indicates the reach
of the *present proof predicates* on the
captured decisions, not a demonstrated defect
or low Pareto efficiency of the sampler.
The next scientific gate is to evaluate these
proportions on source-bound representative outputs,
then formulate and validate a suitable global
oracle or explicit GT geometry-aware generation
policy **before** costly full-budget runs.

## Bounded source-bound pilot before full DTLZ5/6 GT generation

To measure the *actual* source code's behavior without prematurely
committing to costly \(n=2000\) references, the independent research
preflight
[`dtlz56_small_scalarization_pilot.py`](../scripts/audit/dtlz56_small_scalarization_pilot.py)
runs **four fixed preference rays** each for the original DTLZ5 and
DTLZ6 at \(M=4,K=1\). The input rays are obtained by evaluating
four fixed, valid decisions with the original MOP and applying
the production sampler's positive weight floor. The audit then
calls the **actual** production `scalarized_candidate()` and
`decision_vector()` for each ray, evaluates the recovered decisions
with the real MOP and the independently implemented original
equations, and applies the conservative per-row symbolic triage
(Section above).

The source-bound GitHub Actions job
[`gt-dtlz56-small-pilot-175.yml`](../.github/workflows/gt-dtlz56-small-pilot-175.yml)
runs that finite pilot for both original problems under an explicitly
declared Python environment, with one numerical BLAS thread and a
15-minute job budget. It records source SHA, versions, problem/M/K,
all four fixed rays, optimizer scalar gaps and explored nodes,
recovered radial values, original decision/objective byte digests,
independent alignment residuals and per-row theorem categories.
Full raw JSON reports are retained as source-bound workflow
artifacts for 90 days. This CI job does **not** run either
`mop.ps(n_points=2000)` or a large reference calibration.

To reproduce the same limited experiment from a pinned checkout:

```bash
python -m scripts.audit.dtlz56_small_scalarization_pilot \
  --problem DTLZ5 --m 4 --k 1 --abs-tol 1e-5 \
  --max-nodes 5000 --output gt175-dtlz5-M4-K1-small-pilot.json
python -m scripts.audit.dtlz56_small_scalarization_pilot \
  --problem DTLZ6 --m 4 --k 1 --abs-tol 1e-5 \
  --max-nodes 5000 --output gt175-dtlz6-M4-K1-small-pilot.json
```

**Interpretation:** These are **eight optimizer-derived candidate
decisions**, four for each problem, not the `ps()`
farthest-point-selected front or a statistically representative
sample of it. Neither the fraction of decisions covered by
our sufficient mathematical proofs nor the returned scalarization
gap implies a certified GT membership rate, empirical coverage
probability, or a Hausdorff bound for the entire continuous front.
DTLZ6 tail tenth roots may not be rational for stored floats, so
a low proof count must not be interpreted as a defect in
the original GT generator. All whole-GT, finite-reference and
ADR 0059 scientific acceptance statuses remain `unverified`.
The finite pilot cannot replace full-budget experiments or
a complete global positive-membership oracle.

## Exact positive-membership certificates with irrational DTLZ6 radial distance

The original DTLZ6 distance is
\(g=\sum_{i=M}^{M+K-1}x_i^{1/10}\).
A saved floating-point tail coordinate is an *exact binary rational
declaration*, but its mathematical tenth root is generally
irrational. The per-row classifier introduced above therefore
correctly abstains when it cannot represent \(g\) as a rational.
That necessary safeguard does **not** imply that its conditional
global-PF theorems are inapplicable.

The independent opt-in extension
[`dtlz56_dtlz6_radial_enclosure.py`](../scripts/audit/dtlz56_dtlz6_radial_enclosure.py)
rigorously encloses the exact root of each rational tail value
\(v\in[0,1]\) in a dyadic rational interval
\([n/2^b,(n+1)/2^b]\). The integer \(n\) is found **only through
exact comparisons** \(n^{10}v_{\rm den}\le
v_{\rm num}2^{10b}\); perfect tenth powers have coincident exact
endpoints. Summing yields exact rational bounds
\(g_-\le g\le g_+\), with the original range
\(0\le g\le K\) enforced. Any nonzero tail coordinate proves
\(g>0\), even if \(g_-=0\) at finite precision.

For a supporting inner-angle pattern of
Section 3.6.3b with \(p\ge2\), the first angle
\(x_1\) need not vanish. The actual last
objective is
\(c=(1+g)\sin(\pi x_1/2)\).
Machin's exact bounds
\(\pi_-\le\pi\le\pi_+\) and
\(\sin t\le t\) give

\[
c^2\le B_+^2,\qquad
B_+=\frac{(1+g_+)\pi_+ x_1}{2}.
\]

Consequently the **strict rational test**

\[
\boxed{B_+^2 < 1-\frac{4}{p\pi_-}}
\tag{DTLZ56-PF14}
\]

proves the actual strict global condition
(DTLZ56-PF12) for the *irrational* real \(g\).
The original positive-\(g\) theorem therefore
proves global Pareto membership of that exact
decision-image point over all \(0\le g'<g\).
This applies even when a float approximation
to \(g\) would not be acceptable as a
scientific proof.

The helper returns an exact hexadecimal rational
root enclosure and strict proof margin; its
replay recomputes the integer-power comparisons
and sufficient inequality. A separate **opt-in**
per-row audit checks source-decision/objective
numerical alignment and reports proved decision
images and unresolved rows. Importantly, the
earlier exact-rational per-row audit and its
existing regression contracts are **unchanged**;
the source-bound eight-ray pilot reports both
classifiers separately.

**Interpretation remains conditional.** An exact
decision-image theorem is not an exact identity
between transcendental original objectives and
their saved floating-point matrix, nor a proof
of GT-wide coverage, general positive-g PF
classification, method/environment identity or
ADR 0059 numerical/result acceptance. These
fields stay `unverified`. An unresolved row
does not establish that it is dominated. The
new pilot is still only four fixed optimizer-derived
rays per original problem, not a representative
full reference.

## Certified distance from original DTLZ6 decisions to proved Pareto-support strata

The source-bound four-ray pilot identified a specific limitation
of the strict supporting-boundary predicates: an optimizer may return
a binary-floating decision with a constrained inner position
*extremely close to*, but not **exactly equal to**, its required
boundary value \(0\) or \(1\). Replacing such a coordinate by its
nominal boundary **without an explicit mathematical distinction**
would turn an approximate generated point into a falsely
certified exact Pareto member. The proper separate question is:
**how far is its exact-real original objective image from
a separately proven Pareto point?**

For original DTLZ6, fix a feasible decision with first-angle
position \(x_1\) and tail \(x_M,\ldots,x_{M+K-1}\);
the exact radial distance \(g\) may be irrational. Consider an
inner-angle position vector \(u'\) obtained by snapping only
the constrained positions needed for a supporting stratum of
Section 3.6.3b (all-upper, or the relevant upper-prefix and
one lower position). Keep \(x_1\) and the **entire tail unchanged**.
The independent irrational-radial theorem of
Section *Exact positive-membership certificates with irrational
DTLZ6 radial distance* must first prove that
\(F(u',g)\) is globally Pareto optimal.

Let \(m\) be the objective dimension and
\(\Delta=\sum_{i=2}^{m-1}|x_i-x'_i|\).
Changing one inner hyperspherical angle at a time and integrating
its original angle derivative gives the **exact-real Euclidean bound**

\[
\bigl\|F(u,g)-F(u',g)\bigr\|_2
\le \frac{\pi g}{2}
  \sum_{i=2}^{m-1}|x_i-x'_i|.
\tag{DTLZ56-PF15}
\]

Indeed, the norm of every angular partial derivative of the
unit hypersphere parameterization is at most \(1\);
multiplication by the radial factor \(1+g\)
and the original relation
\(|\Delta\theta_i|=
\pi g|\Delta x_i|/[2(1+g)]\)
produce (DTLZ56-PF15).

The independent, read-only
[`scripts/audit/dtlz56_dtlz6_support_proximity.py`](../scripts/audit/dtlz56_dtlz6_support_proximity.py)
enumerates the finitely many objective-support patterns,
verifies that the **snapped** decision satisfies the rigorous
global theorem, and reports the smallest **certified**
upper bound among those enumerated strata. It replaces
\(\pi\) and the potentially irrational \(g\) by rigorously
computed **rational upper bounds** from Machin series and
integer tenth-power comparisons. It records the full original
decision declaration, snapped decision, supporting-PF
certificate, objective index, and hexadecimal rational
distance bound for exact replay.

This is a **one-sided distance-to-front upper bound**:
it proves that a particular globally Pareto-optimal point is
within the stated distance of the *exact-real image of the
source decision*. It does not prove that the source decision
is itself Pareto optimal or dominated, that the selected
front point is the closest on the entire Pareto front,
or that a stored IEEE-754 objective row is *identically*
equal to the exact-real objective image. All public and
whole-reference ADR 0059 certification statuses therefore
remain `unverified`.

The source-bound pilot records its complete four source
decision vectors alongside the existing original symbolic
membership ledger and the new proximity result, so that
a tiny boundary reconstruction discrepancy remains
scientifically inspectable rather than being silently
rounded away. The pilot is not the full `ps()` GT,
and neither its few proof outcomes nor the distances
estimate statistical reference coverage.

## Stratified intermediate audit of the original DTLZ5/6 candidate-pool mechanism

Before running the full \(n=2000\) calibration, a
**source-bound, budgeted intermediate diagnostic** now measures the
three distinct preference-ray families that feed the existing
\(M\ge4\) DTLZ5/6 global scalarization mechanism:

1. Sobol rays formed by evaluating original decisions with
   deterministic angular boundary faces;
2. strictly positive Gaussian preference vectors;
3. sparse Gamma preference vectors.

The opt-in research procedure
[`scripts/audit/dtlz56_stratified_medium_pilot.py`](../scripts/audit/dtlz56_stratified_medium_pilot.py)
mirrors the **existing** deterministic ray-source construction at
\(M=4,K=1\), `n_points=64` (source pool size 256), and selects
**eight evenly spaced source indices per ray family**.
It executes the production `scalarized_candidate()` and
`decision_vector()` on the resulting **24 source rays for
each original problem**, with a strict limit of 5,000
branch-and-bound nodes **per scalarization** and
absolute gap budget \(10^{-5}\). It stores the actual
aligned decision/objective matrices and their byte digests;
checks the original equations independently; evaluates
the previous, scientifically conservative exact-decision
Pareto/dominance predicates; and reports per-family
optimizer cost and theorem applicability. The original
source algorithm, branch-and-bound, `ps()` and
the public calibration interface are left unchanged.

The CI workflow
[`gt-dtlz56-stratified-medium-175.yml`](../.github/workflows/gt-dtlz56-stratified-medium-175.yml)
uses a source-bound Python 3.12 runner, one BLAS thread,
a **15-minute job budget**, the same source seed and
no unbounded optimizations. Raw JSON artifacts and
the resulting per-family counts are preserved for 90 days.
Run explicitly from a pinned checkout with:

```bash
python -m scripts.audit.dtlz56_stratified_medium_pilot \
  --problem DTLZ5 --m 4 --k 1 --abs-tol 1e-5 \
  --max-nodes 5000 --output gt175-medium-DTLZ5-M4-K1.json
python -m scripts.audit.dtlz56_stratified_medium_pilot \
  --problem DTLZ6 --m 4 --k 1 --abs-tol 1e-5 \
  --max-nodes 5000 --output gt175-medium-DTLZ6-M4-K1.json
```

**What is and is not measured.** This intermediate run
tests **48 selected, actual optimizer-derived candidate
rays in total**. It samples each preference-source
mechanism but **does not reproduce the production sampler's
full candidate pool** (448 scalarizations for nominal 64
points), its greedy maximin selection, or the geometry
of the resulting finite GT. The eight fixed source
indices per family are not a random sample suitable
for population inference. Per-family proof counts
measure the *reach of known sufficient theorems*, not
the proportion of Pareto-optimal points.
The reported scalarization gaps bound weighted
Chebyshev optimality, not pointwise Pareto membership
or a finite reference's Hausdorff coverage.

The appropriate interpretation gate is:
(i) all original decision/objective rows are
numerically independently aligned;
(ii) every scalarized optimizer stays within its
declared node and gap budgets;
(iii) exact-decision membership, dominance and
unknown counts and any DTLZ6 proven-PF proximity
bounds are reported **separately**;
(iv) the source SHA, seed, ray indices, versions and
matrix digests are preserved. An unsuccessful
optimization or source inconsistency is a
**pilot failure to investigate**, not an accepted GT;
a low exact-theorem applicability count is
**scientifically inconclusive** about GT quality.
No outcome in this pilot automatically certifies
the full GT, changes an ADR 0059 gate, changes
`assess_gt()`, or justifies replacing stored
references or frozen numeric baselines.

The intermediate stage supplies a reproducible
cost/behavior checkpoint before making the distinct
decision to run production \(n=2000\) on a
source/environment-pinned, potentially faster
local machine with the full reference-coverage
and indicator-adequacy acceptance protocol.

### Observed 48-ray stratified diagnostic (2026-11-24, PR #207)

The original bounded workflow
[`run #45480846865`](https://github.com/monacofj/moeabench/actions/runs/45480846865)
passed its new 13 tests and both 24-source-ray production
scalarizations. Its raw output is
[`issue175-stratified-medium-source-ray-audit`](https://github.com/monacofj/moeabench/actions/runs/45480846865/artifacts/14960139061).
The GitHub runner checkout recorded source SHA
`27a7c84cf54f82a56e404d28c8c6233541947561`,
NumPy 2.5.5 and SciPy 1.18.1. The experiment used
the explicitly stated deterministic indices and
M=4, K=1, \(10^{-5}\) scalarization absolute-gap
budget and 5,000 max nodes per solve.

**Observed decisions, not inferences about the final GT:**

| Problem | Total rays | Exact \(g=0\) rows | \(g>0\) rows | Conditional globally efficient exact-decision images | Remaining unverified |
|---|---:|---:|---:|---:|---:|
| Original DTLZ5 | 24 | 22 | 2 | 24 | 0 |
| Original DTLZ6 | 24 | 22 | 2 | 23 | 1 |

In both problems, the only two positive-\(g\) rows
were indices 10 (positive-Gaussian source index 32)
and 18 (sparse-Gamma source index 16).
DTLZ5 had \(g=0.25\) for both; DTLZ6 had
\(g=1.0\) and approximately
\(g=0.9025764465332031\), respectively.
The first DTLZ6 positive-\(g\) row satisfies
the exact-rational supporting-stratum theorem;
its independent irrational-radial certificate
also applies (the latter is **not an additional
distinct row**). The second remains unverified
by the exact membership predicates, but a
separately proved Pareto-support proximity
bound exists; that is **not** a proof of
membership or of domination.

All 24 DTLZ5 source decisions and 23/24 DTLZ6
source decisions had conditional exact
global-membership proofs. In **both**
problems, **22/24** were \(g=0\) anchors of
the known unit-radius theorem. A high total
count therefore says little about the
**difficult positive-\(g\) component**.
Independent original-equation residual maxima
were about \(1.11\times10^{-16}\) for DTLZ5
and exactly zero in the observed DTLZ6
finite-precision comparison. The maximum
scalarization nodes over the selected rays
were 69 (DTLZ5) and 151 (DTLZ6), both below
the declared 5,000 bound; reported finite
scalarization gaps remained approximately
\(10^{-5}\). By family, 8/8 DTLZ5 rays in
each family received exact-decision proofs;
for DTLZ6, the Gaussian and Sobol family
were 8/8 each and the sparse-Gamma family
was 7/8.

**Research decision for the next gate.** The
mechanism-stratified sample showed unexpectedly
many \(g=0\) selections among fixed preference
rays, but has not examined the production
sampler's **full pool** or its **final
farthest-point selected** objective sample.
Before committing to \(n=2000\), compare
the radial distribution \(g\) and objective
coverage on the *actual final selected*
small-\(n\) PS/F reference, preserving
complete paired evidence. A second,
explicitly targeted positive-\(g\) ray study
may diagnose missing sufficient proofs,
but its targeted counts must not be
represented as unbiased GT-wide rates.
No existing baselines, reference data,
ADRs or public GT acceptance criteria
are changed by this observation.

## Source-bound audit of the actual final-selected 16-point original DTLZ5/6 GT

The stratified source-ray diagnostic above inspected 48 candidate
scalarizations **before** the production greedy maximin stage.
Its observed concentration of \(g=0\) cannot establish the
radial distribution, coverage or adequacy of the final GT.
The next bounded experiment therefore calls the **unchanged
public original `mop.ps(n_points=16)` exactly once**
for each of DTLZ5 and DTLZ6 at \(M=4,K=1\).
This exercise includes the real source preference pool,
the default `scalar_tol=1e-7` Chebyshev solves,
\(g=0\) anchors, and the actual objective-space
greedy maximin selection. The original MOP `evaluation()`
is then called on the final selected 16 decisions.

The separate opt-in research program
[`scripts/audit/dtlz56_final_selected_small_gt.py`](../scripts/audit/dtlz56_final_selected_small_gt.py)
uses [the immutable paired-evidence protocol above](#paired-original-dtlz56-gt-evidence-independent-alignment-preflight)
to retain **both PS and ordered F** in an independently
replayable NPZ file, with its SHA-256 in a separate JSON
diagnostic. It independently evaluates the published
original equations; reports exact numeric \(g=0\) counts,
positive-\(g\) counts and radial quantiles;
records objective-component spans and normalized
nearest-neighbor distances **within only the final
selected sample**; and uses the independent,
conditional exact-decision theorem ledgers.
For DTLZ6 it also reports optional irrational-radial
global-PF sufficiency and certified distance to
proven support strata. The source SHA, Python,
NumPy and SciPy versions, source seed, and
expected number of scalarizations are recorded.

The source-bound GitHub Actions
[`gt-dtlz56-final-small-selected-175.yml`](../.github/workflows/gt-dtlz56-final-small-selected-175.yml)
limits the job to **20 minutes** and one BLAS
thread, and archives all four files (two PS/F
bundles and two JSON reports) for 90 days.
Unlike the 48-ray preselection audit, this
experiment invokes the actual final-stage
sampler: with \(n=16\), its deterministic
source pool still has **256 Sobol, 128 Gaussian
and 64 Gamma rays, i.e. 448 scalarizations
per original problem**. No \(n=2000\) run
is launched. The unit tests mock `ps()`
to avoid this substantial cost in the
ordinary regression suite.

Reproduce manually from a pinned checkout:

```bash
python -m scripts.audit.dtlz56_final_selected_small_gt \
  --problem DTLZ5 --m 4 --k 1 --points 16 \
  --bundle gt175-final-DTLZ5-M4-K1-paired.npz \
  --report gt175-final-DTLZ5-M4-K1-report.json

python -m scripts.audit.dtlz56_final_selected_small_gt \
  --problem DTLZ6 --m 4 --k 1 --points 16 \
  --bundle gt175-final-DTLZ6-M4-K1-paired.npz \
  --report gt175-final-DTLZ6-M4-K1-report.json
```

Both outputs use exclusive creation and never replace
a production calibration sidecar, baseline or
historical reference. The verification step
must independently replay the archived PS/F
bundle digest and original equations. The
spherical norm residual and objective-space
distance statistics are **observational**:
they cannot prove exact membership of an
approximate finite-precision objective row,
continuous-front coverage, reference
regularity, unbiased source-selection
probabilities or indicator adequacy.
All global GT, finite-coverage, indicator
and ADR 0059 scientific acceptance statuses
remain `unverified` regardless of a
successful 16-point source-bound experiment.

The actionable next decision is based on
**the radial \(g\) values after maximin
selection**, not on the earlier pool-only
candidate counts. Positive-\(g\) components
that are absent or sparse in this small sample
would justify an explicit larger source-bound
geometry audit, not automatically altering
the production sampler or its regression
tests. Conversely, a healthy radial spread
among just 16 points does not itself prove
that a 2,000-point GT will cover the front.

### Observed final-selected 16-point DTLZ5/6 results and acceptance blocker (2026-11-24)

The source-bound original production final-selection
[GitHub Actions run #45485829134](https://github.com/monacofj/moeabench/actions/runs/45485829134)
passed its 10 independently scoped tests and
executed **448 actual scalarizations per problem**,
including production objective-space greedy
maximin selection, with Python 3.12.16, NumPy
2.5.5, SciPy 1.18.1 and PR checkout source SHA
`93f75b35a00b8678cf3b2f1b0787437773f68eac`.
The immutable raw paired decision/objective
matrices and JSON reports were archived as
[`issue175-original-final-selected-16-point-paired-gt`](https://github.com/monacofj/moeabench/actions/runs/45485829134/artifacts/14962532983).

| Actual final-selected \(M=4,K=1\) | DTLZ5 | DTLZ6 |
|---|---:|---:|
| Total selected | 16 | 16 |
| Numerically exact \(g=0\) | 8 | 6 |
| Positive \(g\) | 8 | 10 |
| Exact-rational-\(g\) conditional global-PF decision proofs | 12 | 8 |
| Exact-rational-\(g\) conditional **strict domination** proofs | **1** | 0 |
| Unverified under that exact-rational-\(g\) classifier | 3 | 8 |
| Additional DTLZ6 irrational-\(g\) support proofs | — | 5 |
| Independent equation maximum residual | \(1.11\cdot10^{-16}\) | \(2.22\cdot10^{-16}\) |
| Sample-normalized nearest-neighbor min / median / max | 0.1752 / 0.2036 / 0.3144 | 0.2719 / 0.2814 / 0.3757 |

The DTLZ6 irrational-\(g\) proofs overlap some
exact-rational proofs. They **must not** be
added to the 8/16 total until deduplicated by
original row index. These selected-sample
nearest-neighbor observations are not a
Hausdorff bound or a certified measure of
coverage of the unknown continuous front.

**Confirmed experimental failure: original DTLZ5
selected row 10 (zero-based).** The archived binary
decision (decimal display) is
\(x_1=0.1671198950208142\),
\(x_2=0.999999658561122\),
\(x_3=0.9999999999999996\) and
\(x_4=0.6805481014263987\), with
\(g\approx0.03259761692867714\).
Both inner positions are **strictly interior** in
their exact IEEE-754 dyadic interpretation.
The independently verified exact same-ray theorem
from Section 3.6.3 produces a valid decision
with strictly smaller \(g\) and the same
objective direction, and hence an **exact
strict componentwise dominator** of this
candidate's original exact-real image.
This is a rigorously detected defect in the
small final-selected *candidate reference*,
independently of the numerical magnitude
of its dominance gap. It is tracked as
[issue #209](https://github.com/monacofj/moeabench/issues/209)
under the open GT certification work #175/#176.

The new research-only report includes a distinct
**experimental reference gate**:
`blocked_proven_dominated_decision`
with exact row indices if any such sufficient
mathematical dominance witness exists; otherwise
it reports `unverified_no_known_dominator`,
never an inferred whole-GT `pass`.
This gate is outside the public calibration
contract and does **not** silently change
`assess_gt()`, an accepted ADR or a
historical numerical baseline.

**Scientific decision:** the source-ray \(g=0\)
concentration from the previous pilot is
strongly altered by final maximin selection;
there are 8 positive-\(g\) DTLZ5 and 10
positive-\(g\) DTLZ6 selected rows.
That is useful observational geometry evidence,
but cannot vindicate the reference given the
positive strict domination witness.
The next work item is to **replay and diagnose
the exact witness** and locate whether the
problem arises during original scalarization,
inverse decision reconstruction and/or
final candidate selection. A production fix
must first be checked against the governing
ADRs and existing regression barriers; no
existing test or baseline should be rewritten
to hide the witness. Full \(n=2000\) generation
remains deferred until this source-bound
quality gate is addressed.

### Reconstruction repair candidate — preserve exact active angular faces (#209)

The actual final-selected sixteen-point DTLZ5
reference exposed an exact-representational
domination witness at selected row 10:
the weighted Chebyshev solver correctly
restricts angular coordinates to its original
admissible interval
\([a(g),b(g)]\), but the original
`decision_vector()` regenerated position
variables from rounded trigonometric
*objective values* using `atan2` and an
inverse scaling with a divisor proportional
to \(g\). A mathematically active angular
bound can consequently emerge as an
IEEE-754 position *slightly inside*
\((0,1)\), losing its exact endpoint
constraint. The original exact same-ray
criterion then finds a strictly
smaller-\(g\) dominator, even when the
objective improvement is smaller than
routine floating-point resolution.

The proposed implementation change
**preserves the solver's discrete active
angular face state** (lower \(0\),
upper \(1\), or interior `None`) in the
research `ScalarizedCandidate` output.
The decision reconstruction emits
*exactly* \(0\) or \(1\) for active faces
and retains the original analytical
inverse for interior angles and
historical candidates without metadata.
This is not a tolerance-driven blanket
snap: only faces known active by the
actual branch-and-bound scalarizer
are reconstructed as exact endpoints.
The original MOP evaluation equations,
objective-space weighted Chebyshev solver,
candidate ordering, final greedy maximin
policy and public calibration contracts
remain unchanged.

**Scientific acceptance requires independent
replay** on the actual final selected
source-bound n=16 paired reference. The
research-only gate should report no
mathematically proven strictly dominated
exact decision images. That necessary
condition alone remains insufficient
to certify arbitrary remaining positive-g
points, continuous-front coverage,
indicator fitness, the identity of
stored floating-point objective rows
with their exact-real images, or full
ADR 0059 acceptance. No regression
test or frozen numerical baseline may
be modified to make this check pass.

### Source-bound validation of the angular active-face repair (#209)

The real production sampler was rerun in
[PR #210](https://github.com/monacofj/moeabench/pull/210)
with the original \(M=4,K=1,n=16\)
parameters and all **448 scalarizations per
problem**, followed by the unchanged greedy
maximin selection. On the first complete
experimental job, 12 new scoped tests
passed, with the following results:

| Original final-selected sampler | Before active-face repair (#208) | After active-face repair (#210) |
|---|---:|---:|
| DTLZ5 exact globally-PF decision-image proofs | 12 | **13** |
| DTLZ5 provably dominated exact decision images | **1** | **0** |
| DTLZ5 unverified exact decision images | 3 | 3 |
| DTLZ6 exact-rational decision-image proofs | 8 | 8 |
| DTLZ6 provably dominated decision images | 0 | 0 |

The fixed-source comparison reveals that the
previously dominated DTLZ5 row is now
reconstructed exactly on the face known active
to the original scalarizer, eliminating its
strict same-ray witness **without changing the
weighted scalar objective optimization or
farthest-point selection**. These are
conditional *exact-decision-image* certificates,
not an exact statement about equality of
transcendental objective coordinates to
stored IEEE-754 values. The remaining
unverified rows must not be silently
promoted to membership or rejection.

The research-only n16 CI gate is now **strict**
for known mathematically proved dominated
rows: a candidate with any exact strict
same-ray dominator makes that source-bound
experiment fail. Passing that negative
guard means only *no such witness detected
among these sixteen decisions*, never
complete GT scientific acceptance. Existing
historical regression tests, numerical
baselines, public `assess_gt()` semantics,
MOP equations, and governing ADR 0059 are
unchanged. Full GT and coverage certification
remains `unverified`.

### Rigorous Taylor refinement of the nonzero-first-angle support theorem (#175)

The original global membership proof in Section 3.6.3c
requires the exact-real last objective
\(c=(1+g)\sin(\pi x_1/2)\) to satisfy

\[
c^2<1-\frac{4}{p\pi},\qquad p\ge2,
\tag{DTLZ56-PF16}
\]

at an exact supporting inner-angle stratum.
The first independent rational checker used the
conservative inequality \(\sin t\le t\).
That overestimate is **too coarse** for
two actual DTLZ5 M=4 K=1 final-selected n16
positive-\(g\) decision images: rows 7 and 11.

No new global membership assumption is necessary.
For \(0\le t\le\pi/2\), the alternating
Taylor series gives a mathematically rigorous
upper bound

\[
\sin t\le T_5(t)
:=t-\frac{t^3}{6}+\frac{t^5}{120}.
\tag{DTLZ56-PF17}
\]

With exact Machin \(\pi_{\rm lo}<\pi<\pi_{\rm hi}\),
put \(t_{\rm hi}=\pi_{\rm hi}x_1/2<2\).
Since sine is increasing on its original
first-angle domain,
\(\sin(\pi x_1/2)\le\sin(t_{\rm hi})
\le T_5(t_{\rm hi})\).
The entire following check is made using exact
fractions (IEEE-754 inputs are read as their
precise dyadic rational declarations):

\[
\boxed{\quad
((1+g)T_5(t_{\rm hi}))^2
<
1-\frac{4}{p\pi_{\rm lo}}.
\quad}
\tag{DTLZ56-PF18}
\]

When this inequality is strictly verified,
the earlier all-\(g'\in[0,g)\) global Pareto
non-dominance theorem applies with no
numerical tolerance or open-endpoint gap.

A separate opt-in
[`dtlz56_tight_sine_global_pf.py`](../scripts/audit/dtlz56_tight_sine_global_pf.py)
implements the exact-rational proof, lossless
replayable inputs and proof margin. It
does **not** alter the previous conservative
checker or its regression contracts.

For the preserved original DTLZ5 M=4 K=1
source decisions from the fixed n16
post-#209 final selected PS/F:
- Row 7: \(x_1\approx0.3108514266364233\),
  \(x_2=1,x_3=0,g=1/4\), with
  actual \(c^2\approx0.34385229\).
- Row 11: \(x_1\approx0.3104519947719909\),
  \(x_2=x_3=1,g=1/4\), with
  actual \(c^2\approx0.34304034\).

Both are strictly below the exact required
threshold \(1-2/\pi\approx0.36338023\),
and the tighter rational inequality
(DTLZ56-PF18) has independently
positive rational margins for both.
The previous \(\sin t\le t\) helper
correctly abstained because its bound
was looser. The new separate theorem
expands exact-decision-image membership
coverage from **13/16 to 15/16** for
that source-bound sample, without changing
the historical decision-image ledger.

Row 10, with \(x_2\) strictly interior
near 1 and \(x_3=1\), does not
satisfy the **exact** support pattern
of this theorem and remains
`unverified`; no arbitrary angular
rounding is applied.

These are conditional theorems about
the original exact-real images of
given feasible **decision vectors**,
not certificates for the identity of
stored finite-precision objective rows,
arbitrary positive-\(g\) points,
finite-reference coverage or ADR 0059
environment/result acceptance. The
source-bound small selected-reference
audit reports original and new theorem
ledgers separately and also their
**deduplicated row-index union**;
unknown rows remain unknown.

