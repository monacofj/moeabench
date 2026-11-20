<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Ground Truth in MoeaBench: Mathematical Foundations, Reference Generation, and Quality Assurance

This document explains how **Ground Truth (GT)** is defined, constructed, used, and validated in MoeaBench. It is written for readers without prior knowledge of the project or its research history. It starts from multiobjective optimization fundamentals, develops the mathematical and numerical findings for DPF3 and DPF5, and explains their implications for experimental assessment, sampling, calibration, and historical references.

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
