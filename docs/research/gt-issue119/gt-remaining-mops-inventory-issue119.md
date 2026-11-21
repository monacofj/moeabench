<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Ground Truth Audit of the Remaining Built-in DTLZ and DPF Problems — Issue #119

This report documents a **read-only scientific inventory** of ten remaining
MOP families in MoeaBench, following the DPF3/DPF5 corrections integrated in
[PR #140](https://github.com/monacofj/moeabench/pull/140).
It explains which numerical findings are evidence of mathematical
incorrectness, which reflect **finite-sampling quality**, and which are
ordinary floating-point artifacts. No production generator, calibration
default, historical regression test, stored GT or baseline is modified by
this audit.

The permanent concepts and API responsibilities are explained in
[docs/gt.md](../../gt.md). This report is an **issue-specific experimental
record**, not a substitute for a permanent GT specification.

## 1. Scope and sources

The MOPs examined are **DTLZ1–4, DTLZ7–9, DPF1, DPF2 and DPF4**.
DTLZ5/6 completeness for four or more objectives belongs to [issue
#65](https://github.com/monacofj/moeabench/issues/65) and is explicitly
excluded. Corrected DPF3/DPF5 are already covered by the proofs and
regression suites of the previous phase.

The mathematical specifications are those of:

- Deb, Thiele, Laumanns and Zitzler, *Scalable Multi-Objective Optimization
  Test Problems* (DTLZ technical report, 2001), especially Sections 8.1–8.9,
  [original manuscript](https://tik-old.ee.ethz.ch/file/c7c5e610a0c7e26d566a3601e5cce2f4/DTLZ2001a.pdf).
- Zhen et al., *Multiobjective Test Problems with Degenerate Pareto Fronts*,
  arXiv:1806.02706v2, especially Sections IV-A, IV-B, IV-D, and **IV-F**
  specifying increasingly sorted chaotic thresholds for DPF3 **and DPF4**,
  [arXiv manuscript](https://arxiv.org/pdf/1806.02706).

The implementation was inspected directly in
\`moeabench/mops/DTLZ*.py\`, \`DPF*.py\`,
\`base_dpf.py\` and \`_dtlz7_pareto_regions.py\` on
\`main\` after commit \`f4ff281e\`.

## 2. Scientific interpretation of the checks

A generator can return exactly the requested number of distinct objective
rows, be deterministic, satisfy constraints and contain **no internally
dominated point**, yet still fail to represent the mathematically defined
continuous PF. These are **necessary but not sufficient conditions**.

The independent audit records requested/returned/unique PS/PF counts,
replay, feasibility, coordinate-domain validity, exact-vs-evaluated
agreement, internal dominance, nearest-neighbor spacing and selected
geometric invariants. For DTLZ1, independent **Dirichlet(1)** simplex
reference points are generated from exponential variates; for DTLZ2/3/4,
independent positive unit hyperspherical directions are generated from
normalized absolute Gaussian variates. All of these are independent of the
MOP's **ps()** method. Their distance statistics are **finite-probe
diagnostics**, not upper bounds for continuous Hausdorff covering.

The DTLZ7/DPF2 checks count disconnected Pareto-record parameter
components, but merely touching a component does not certify adequate
within-component coverage. For DTLZ8/9 the reported
\`feasible\` indicator uses the current strict **\`G<=0\`** comparison, while
the audit separately distinguishes violations exceeding \`1e-12\`.
No tolerance is silently applied to production constraints.

## 3. Baseline survey: 46 configurations

The audit originally tested **46 cases** with
\`M=3,5\`, \`n_points=128,512\` and, where meaningful,
\`DPF.D=2,3\`. All 46 returned the requested number of numerically
distinct objectives, had zero internally dominated **distinct** PF rows,
replayed deterministically, and had no out-of-box decisions. **This
statement does not establish that all 46 are scientifically adequate.**

| MOP | Mathematical PF / existing generator | Observed properties | Scientific interpretation |
|---|---|---|---|
| **DTLZ1** | Nonnegative simplex \(\sum f_i=0.5\); 42-seeded random position variables | Correct simplex equation, requested count, no sample-relative dominance; exact simplex vertices **not included** | Pareto membership is plausible, but the finite GT omits critical extrema and is not objective-simplex uniform |
| **DTLZ2** | Positive unit hypersphere; normalized absolute Gaussian directions, mapped back to decisions | Correct unit norm, exact axis anchors, deterministic requested count | Existing geometry-specific reference is supported by independent checks |
| **DTLZ3** | Same positive spherical PF with multimodal off-front \(g\) | Same objective geometry, anchors and count as DTLZ2 | Sampling equivalent to DTLZ2 is **mathematically appropriate**; do not invent a different sampler merely to avoid code reuse |
| **DTLZ4** | Same PF with \(x^{100}\) angular bias; inverse transform of spherical directions | Correct sphere, axes and count; numerical inverse transform is distinct from DTLZ2 | Shared spherical **target** is valid, with DTLZ4-specific inverse decisions |
| **DTLZ7** | \(2^{M-1}\) disconnected analytical Pareto-record regions | All 4/4 (M=3) and 16/16 (M=5) components observed at 128/512 | Region restriction is correct; high-M component occupancy and **within-component** coverage need separate qualification |
| **DTLZ8** | Constrained line/plane geometry; static CSV for M=3/5/10, fallback guided arms otherwise | Static files pass small budgets, but impose a **1,000-row ceiling**; fallback M=4 returns fewer/duplicated points | A dedicated complete-front sampler must respect requested/effective count and cover the geometrical components |
| **DTLZ9** | Constrained curve \(f_1=\cdots=f_{M-1}\), \(f_j^2+f_M^2=1\) | Correct count and nondominance; strict feasibility flags are affected by roundoff near the equality constraints | Distinguish genuine infeasibility from float64 arithmetic; spacing in objective arc length differs from spacing in \(f_M\) |
| **DPF1** | DTLZ1-like essential simplex followed by linear redundant projections | For D=2, structured sampling includes the two essential extremes; for D=3, random sampling omits vertices | MOP-owned sampling should preserve extremes in the **essential** simplex; a mathematically equivalent simplex kernel may be shared |
| **DPF2** | DTLZ7-like disconnected essential front and nonnegative increasing nonlinear projections | 2/2 (D=2) and 4/4 (D=3) record components observed, no sample-relative dominated points | The exact record restriction is supported; higher-D component coverage is not automatic |
| **DPF4** | Inverted hyperspherical essentials and squared implicit min/max projections | Requested count and internal nondominance pass, **but evaluator disagrees with published increasingly sorted \(\eta\)** | **Mathematical equation defect** despite apparently healthy finite cardinality and dominance diagnostics |

### 3.1. DTLZ1 and DPF1: why extrema matter

With a mathematical simplex reference, the coordinate-axis vertices are
genuine Pareto-optimal extreme solutions. A decision-uniform random sample
almost surely omits them exactly. That omission can bias estimated ideal/nadir
bounds, reduce measured coverage and affect GT-normalized indicators.

At \`n_points=128\`, the *largest distance in unnormalized objective space
to an exact essential simplex vertex* was:

| Case | Largest extreme gap, n=128 | Largest extreme gap, n=512 |
|---|---:|---:|
| DTLZ1 M=3 | 0.11989 | 0.05743 |
| DTLZ1 M=5 | 0.29022 | 0.23130 |
| DPF1 M=5, D=2 (essential simplex) | approximately 0 | approximately 0 |
| DPF1 M=5, D=3 (essential simplex) | 0.11989 | 0.05743 |

These gaps are **measured** for the reproducible incumbent samples. They
are not proof of a universal worst-case sampling error. Retaining every exact
simplex vertex is nevertheless a sound and inexpensive necessary condition
for the geometry-specific generators of DTLZ1 and DPF1.

For DPF1, the redundant objective map is a **nonnegative linear mapping**
of the essential simplex, so the scientific simplex parameterization
is identical to DTLZ1's essential mathematical kernel. Sharing a pure
simplex-to-position conversion helper in this case is legitimate
mathematical reuse; each MOP must still verify its own evaluation and
objective-space coverage.

### 3.2. DTLZ8: static and fallback paths both require improvement

The static analytical lookup files for \`M=3,5,10\` each contain exactly
**1,000 decision rows**. The implementation returns all file rows when
\`n_points >= 1000\`, regardless of the requested count.

| M in {3,5,10} | Requested | Actual | Unique evaluated objectives |
|---|---:|---:|---:|
| Each static configuration | 2,000 | 1,000 | 1,000 |
| Each static configuration | 10,000 | 1,000 | 1,000 |

This matters because **2,000** is the historical DTLZ default GT
calibration budget; an unqualified \`pf(2000)\` currently underproduces by
half in these configurations.

For M=4, which has no bundled static file, the alternative constructor
uses integer square-root side lengths and repeats objective vectors at
several arm interfaces:

| M=4 requested | Decisions returned | Distinct objectives | Constraint violations above 1e-12 |
|---|---:|---:|---:|
| 64 | 64 | 56 | 0 |
| 128 | 122 | 112 | 0 |
| 512 | 491 | 469 | 0 |

The fallback failure is **not** an infeasibility argument: violations at
the level of \(10^{-16}\) are numerical boundary effects; finite
coverage and budget cardinality are separate issues. The original DTLZ
paper describes the correct PF as a **line and a constrained
hyperplane/surface**. A reliable DTLZ8 generator must explicitly sample
both parts, avoid repeated objective images at interfaces, and expose
effective cardinality without imposing an unsupported arbitrary
mixed-dimensional probability law.

### 3.3. DTLZ9: floating-point constraint flags

DTLZ9 creates a Pareto curve satisfying

\[
f_1=\dots=f_{M-1}=\sqrt{1-f_M^2}
\]

in the published formulation. The native generator samples \(f_M\)
linearly and maps each coordinate back into decision blocks via the
inverse \(x=f^{10}\).

At M=3, 42/128 and 185/512 of the generated rows were flagged
**infeasible by the strict float64 comparison**, even though the
largest positive constraint residuals were respectively
\(\approx4.44\times10^{-16}\) and
\(\approx5.55\times10^{-16}\). **No point exceeded
the independently reported 1e-12 diagnostic tolerance.**
The same per-budget count was observed at M=4,5,10; this arithmetic
artifact must not be mistaken for mathematically infeasible PS points.

Constraint handling in \`evaluation()\` is a **public numerical
contract**. A possible tolerance or stable inverse transformation
requires separate impact and ADR analysis, rather than silently
changing the meaning of \`feasible\` while auditing GT.

The curve's spacing is not generally uniform in *objective arc
length*. If \(f_M=\sin\theta\) and
\(f_1=\cdots=f_{M-1}=\cos\theta\), then

\[
\mathrm ds = \sqrt{(M-1)\sin^2\theta+\cos^2\theta}\,\mathrm d\theta .
\]

Uniform \(f_M\) spacing is thus not the same as uniform arc-length
sampling, particularly for higher M. Reproducibility and correctness
of membership do not by themselves certify spatial regularity.

### 3.4. DTLZ7 and DPF2: the component-count lower bound

The published record-region constructions contain **two disconnected
parameter intervals per independent position coordinate**, hence
\(2^{M-1}\) possible Cartesian components in DTLZ7 and
\(2^{D-1}\) in DPF2. Touching all components requires **at least that
many distinct reference points**, irrespective of how good an
individual point may be.

The incumbent reproducible pseudo-random sampler independently samples
record intervals and does not stratify **by component identity**.
The high-dimensional audit measured:

| MOP | Objectives / intrinsic D | Requested points | Components observed | Total components | Missing components |
|---|---|---:|---:|---:|---:|
| DTLZ7 | M=10 | 128 | 108 | 512 | 404 |
| DTLZ7 | M=10 | 512 | 319 | 512 | 193 |
| DPF2 | M=10, D=9 | 128 | 94 | 256 | 162 |
| DPF2 | M=10, D=9 | 512 | 216 | 256 | 40 |

At budget 128, covering all 512 DTLZ7 components is **mathematically
impossible**. At budget 512, component presence is theoretically possible
with exactly one point per component, yet the present sampler observed
only 319: random selection **with component replacement** cannot
guarantee full component occupancy. Similarly, DPF2 with 256 possible
components failed to occupy all of them at a budget of 512.

For a meaningful reference, the native MOP generator should explicitly
control component occupancy whenever the budget permits it. A budget
smaller than the component count must be **reported as incapable of
complete component presence**, not represented as full-front coverage.
Even after all components are touched, one point in each cannot
establish good **within-component spatial coverage**.

A geometry-based calibration default must ultimately account for the
combinatorial growth of components and computational cost. The
algorithm and numeric budget policy of \`mop.calibrate()\` remain a
separate ADR 0057 decision; this audit does not adopt a default.

### 3.5. DPF4: independently demonstrated divergence from the paper

The original DPF paper requires \(\eta_1<\dots<\eta_{M-D}\) for
**both DPF3 and DPF4** (Section IV-F), and DPF4 applies increasing
\(\phi(t)=t^2\) to its implicit min/max projections.
For an essential last objective \(t=\gamma_D\), ordered
projection coordinates \(p_i(t)\) satisfy

\[
\sum_{i=0}^{M-D}p_i(t)=t+\sum_{j=1}^{M-D}\eta_j.
\]

The current DPF4 source uses the **unsorted** logistic sequence inherited
from \`BaseDPF\`; with multiple thresholds, the identity can fail and
the evaluated objective vectors differ from the original published
equations. The source equation was reconstructed independently in
\`gt_dpf4_published_equations_issue119.py\`, without calling the
\`DPF4.eval_base_functions()\` kernel.

| DPF4 M,D | Raw thresholds ordered? | Max component difference to published equation | Projection telescoping residual |
|---|---|---:|---:|
| 3,2 | Trivially (one threshold) | 0 | 0 |
| 5,2 | No | 0.5096667133 | 0.3844009575 |
| 6,3 | No | 0.5096667133 | 0.3844009575 |
| 7,4 | No | 0.5096667133 | 0.3844009575 |

These are maxima **over the selected test decisions**, not global
certificates of worst-case error. The math is nevertheless decisive:
the source projection does not implement the sorted published
thresholds. Sorting the **shared** BaseDPF chaos stream would
unintentionally alter the intended DPF1/2 weighting semantics.
Any correction must remain **DPF4-specific**.

A separate numerical probe at \`x1=1e-10\`, D=2, g=0 revealed another
defect: direct float64 \`1-cos(theta)\` evaluation gives **zero**, whereas
a stable equivalent \`2*sin(theta/2)^2\` followed by \(\phi(t)=t^2\)
gives approximately **1.5220170474e-40**, which is readily
representable in float64. This is avoidable cancellation, not underflow.

Under the **published ordered equation**, the global PS can be
characterized analytically: \(\phi(t)=t^2\) is strictly increasing on
nonnegative inputs, and the ordered projection's telescoping identity
preserves essential objective ordering. The DPF4 essentials have the
same inverted nonnegative unit-hypersphere structure as DPF3, with
linear rather than \(x^{100}\) angles and nonnegative multimodal
Rastrigin \(g\). Exactly the \(g=0\) decisions are globally
Pareto-optimal; their objective images form the published PF.
A corrected finite reference can therefore sample those angular
parameters directly, rather than relying on arbitrary decision-space
random samples.

## 4. Engineering implications without silent policy changes

The audit distinguishes:

1. **Mathematical noncompliance:** DPF4's unsorted thresholds and
   cancellation alter published objectives, independently of the GT
   generator. Correct source equations and an independent positive
   oracle test are needed before accepting a new DPF4 reference.
2. **Finite reference incompleteness:** DTLZ8's 1,000-row static ceiling
   and its fallback square-grid truncation; DTLZ1/DPF1 missing simplex
   vertices; high-dimensional DTLZ7/DPF2 disconnected components require
   explicit occupancy control and sometimes more points than the budget.
3. **Numerical interface contract:** DTLZ9's tiny strict-feasibility
   violations require careful treatment of \`G\` and \`feasible\`
   semantics, separate from objective-space coverage.
4. **Validated mathematical reuse:** the DTLZ2/3/4 spherical target
   and the DTLZ1/DPF1 essential simplex are genuinely equivalent
   mathematical geometries and may share verified kernels. This does
   **not** authorize a universal sampler for arbitrary MOPs.

Changes to \`moeabench/mops/\`, existing tests, stability fixtures,
default \`mop.calibrate()\` budgets, or ADR-governed contracts are
**outside this read-only audit PR**. In particular, ADR 0057's dated
implementation authorization explicitly covered DPF3/DPF5;
new MOP-specific corrections require a fresh explicit approval when
they affect an accepted architectural decision. Any future changes to
existing regression tests or numerical references must be explained
and approved **before** editing them.

## 5. Reproducibility and exact evidence locations

Run the independent inventory from repository root:

\`\`\`bash
python -m scripts.audit.gt_remaining_mops_issue119 \
  --objectives 3 5 --budgets 128 512 --intrinsic 2 3 --probes 2048

python -m scripts.audit.gt_remaining_mops_issue119 \
  --names DTLZ8 DTLZ9 --objectives 3 4 5 10 \
  --budgets 64 128 512 --probes 512

python -m scripts.audit.gt_dpf4_published_equations_issue119
\`\`\`

The isolated PR workflow runs **39 new independent tests** and exports
structured JSON for the MOP audit, constrained dimensions, DPF4
published equation comparison, and high-dimensional disconnected-component occupancy. Its
diagnostic metrics are measurements, never automatic permission to
rewrite a frozen scientific reference.

See [issue #119](https://github.com/monacofj/moeabench/issues/119) for
the overarching acceptance matrix and
[umbrella #122](https://github.com/monacofj/moeabench/issues/122) for
coordination with issue #65, absolute-HV research #89, and dependency
quality tracks.
