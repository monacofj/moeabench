<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF5: Constructive Angular Covering for Both Published Pareto Components

**Scope:** an independent **real-arithmetic** geometric bound for a specific finite reference construction under the **published DPF5 formulas**. This document accompanies the production correction in [Draft PR #140](https://github.com/monacofj/moeabench/pull/140) and the mathematical results of [issue #119](https://github.com/monacofj/moeabench/issues/119). The result does **not** establish a general float64 rounding certificate or a universal probability measure for combining components.

## 1. Geometry of the two branches

The published objective function has the radial form \(F(x)=(1+g(x))u(x)\), with \(g(x)\ge0\) and \(\|u(x)\|_2=1\). The established published-model global Pareto set is \(g(x)=0\); therefore its front consists of unit-sphere directions.

On branch **P**, the \(M-1\) independent angular parameters are

\[
\theta_1 \in [0,\pi/6), \qquad
\theta_2,\ldots,\theta_{M-1}\in[0,\pi/2].
\]

On the degenerate branch **D**, only \(D-1\) angles influence the objectives:

\[
\phi_1\in[\pi/6,\pi/2],\qquad
\phi_2,\ldots,\phi_{D-1}\in[0,\pi/2].
\]

For P the direction is the standard unit hypersphere \(H_M(\theta)\); for D it is a linear isometric embedding of the \(D\)-dimensional hypersphere \(H_D(\phi)\), obtained by repeating its first coordinate \(M-D+1\) times after division by \(\sqrt{M-D+1}\). Both are **1-Lipschitz** in their own Euclidean angular coordinates: the hyperspherical angular derivatives are orthogonal and each has norm at most 1; D's replicated linear map preserves Euclidean distances.

Importantly, **the component dimensions are different** (\(M-1\) and \(D-1\)). The following bounds are **conditional by component**; no cross-dimensional uniform sampling probability is imposed.

## 2. Complete tensor grids and real-arithmetic upper bounds

For each component, choose \(k\ge2\) endpoint-inclusive, equally spaced values for **each independent angular parameter**, and include their full tensor product in the reference.

For P, the exact boundary \(x_1=1/3\) belongs to D, so it cannot be included as a P sample. Use instead the nearest representable float64 value below \(1/3\), denoted \(b=\operatorname{nextafter}(1/3,0)\), as the P angular grid's last first-coordinate parameter, and define the small omitted-edge difference

\[
\delta_P=\frac{\pi}{2}(1/3-b).
\]

The P rectangle has angular side lengths \((\pi/6,\pi/2,\ldots,\pi/2)\). Every point is at angular Euclidean distance at most half its grid-cell diagonal **plus the omitted-edge correction**. Consequently, the **exact-real-arithmetic** directed covering error in objective space is bounded by

\[
\boxed{\varepsilon_P(k)
\le \frac{\pi}{4(k-1)}
\sqrt{\frac19+(M-2)}+\delta_P .}
\]

For D, the angular side lengths are \((\pi/3,\pi/2,\ldots,\pi/2)\). The complete D grid yields

\[
\boxed{\varepsilon_D(k)
\le \frac{\pi}{4(k-1)}
\sqrt{\frac49+(D-2)} .}
\]

When the finite GT contains complete P and D tensor grids with \(k_P\) and \(k_D\) subdivisions, the **whole published PF** has a continuous directed-covering and Hausdorff upper bound

\[
\boxed{d_H(PF,GT)\le\max\{\varepsilon_P(k_P),\varepsilon_D(k_D)\}.}
\]

The reverse Hausdorff direction vanishes because the sample decisions satisfy \(g=0\). The supremum over the P component is meaningful even though the published P patch is half-open.

**Proof:** On each angular axis, choose the nearest grid coordinate. The per-coordinate angular error is at most half the grid step, except for the omitted P upper endpoint, for which it is additionally bounded by \(\delta_P\). By the Euclidean triangle inequality and the 1-Lipschitz property, the objective-space nearest-sample error is no larger than the angular discrepancy. Taking the maximum of the two branchwise upper bounds covers the union of both branches. The proof does not require any assumption about relative probability mass, area, or sample density between P and D.

## 3. Geometry-specific candidate implementation and exact budget

The candidate \`DPF5.ps(n_points)\` in PR #140 independently builds **nested full grids** at \(k=2,3,5,9,\ldots\), retaining complete levels where their nominal \(k^{M-1}\) (P) or \(k^{D-1}\) (D) decision count fits its candidate budget. Each finer grid includes the coarser positions; duplicate objective images are replenished with deterministic Sobol interior points, and selected special branch boundary/extreme anchors are preserved.

At fixed total requested \(n\), feasible integer allocations between P and D are evaluated. Among those considered, the primary comparison is the **largest of the two analytic bounds for completely retained grid levels**; an independent deterministic finite-probe nearest-neighbor maximum breaks ties. Candidate allocation thresholds include exact tensor-grid completion counts to avoid missing discrete improvements of the analytic bound.

This policy is **MOP-specific** and keeps \`n_points\` as an explicit total target; it does not introduce an automatic \`calibrate()\` budget, a mixed-dimensional probability law, hidden persistence, or changes to other MOPs.

**Qualification:** The bound applies to a returned **prefix that contains a complete grid for both components**. For exceptionally small budgets or very high intrinsic dimension (when even a 2-point-per-angle tensor grid cannot fit), that condition can fail, and only the empirical diagnostics remain available. A full grid's nominal point count can exceed its mathematically or numerically unique objective-image count at spherical singularities; deduplication preserves the geometric support but alters the effective unique count. The candidate refills rather than silently reducing the returned count.

## 4. Reproduced implementation evidence (research CI, 2026-11-20)

The isolated [correction workflow #44851871732](https://github.com/monacofj/moeabench/actions/runs/44851871732) passed **43 independent new tests** and exported `gt119-dpf5-corrected-cover.json`. The objective values from the corrected production evaluator agreed with a separately reconstructed published DPF5 formula to the displayed precision (maximum component error 0 in the evaluated examples).

The table reports **returned numerically distinct objectives**, the largest **fully retained** nested grid level in each branch, and the consequent **ideal exact-real-arithmetic** whole-front covering bound.

| M,D | Requested = returned = unique | P / D objectives | Complete grid kP / kD | Ideal real covering upper bound |
|---|---:|---:|---:|---:|
| 3,2 | 128 | 111 / 17 | 9 / 17 | 0.103485 |
| 3,2 | 512 | 447 / 65 | 17 / 65 | 0.051743 |
| 5,2 | 128 | 125 / 3 | 3 / 3 | 0.692656 |
| 5,2 | 512 | 326 / 186 | 3 / 129 | 0.692656 |
| 6,3 | 128 | 86 / 42 | 2 / 5 | 1.592464 |
| 6,3 | 512 | 416 / 96 | 3 / 9 | 0.796232 |
| 6,4 | 128 | 85 / 43 | 2 / 3 | 1.592464 |
| 6,4 | 512 | 437 / 75 | 3 / 3 | 0.796232 |

The larger bounds for higher intrinsic dimensions are **conservative**. A bound that does not improve from 128 to 512 in one configuration indicates that the entire higher tensor-grid level still does not fit; finite-probe regularity can improve between such thresholds without improving this particular mathematical certificate. These are **not** bounds certified for floating-point evaluation.

Cost observations on the same CI runner: DPF3 M5/D3 generated 2,000 unique objectives in approximately 0.014 s; DPF5 M5/D2 produced 2,000 in 0.063 s and 10,000 in 0.118 s; DPF5 M6/D4 produced 2,000 in 0.025 s. They are descriptive single-execution observations, not portable performance certifications.

## 5. What is and is not certified

- **Proved:** exact membership and completeness of published DPF5 PS/PF; 1-Lipschitz angular directions; branchwise and union continuous coverage upper bounds for **complete ideal tensor grids** in **unnormalized Euclidean objective space**.
- **Measured:** finite sample cardinality and the deterministic holdout conditional distances used to compare candidate allocations.
- **Not yet proved:** a uniform IEEE float64 rounding-error envelope covering decision construction, \`sin\`/\`cos\`, near-branch values, and \`evaluation()\`; a global finite-covering guarantee for budgets lacking both full grids; equality of mathematical and float64 objective uniqueness; asymptotic optimality or universal density of the selected allocation; affordable operation across arbitrarily high dimensions.
- **Not changed:** historical numerical stability baselines, any preexisting regression test, the persisted GT schema, or \`calibrate()\`'s historical automatic budget. Those require separate versioned migration and explicit maintainer authorization.

A finite-probe maximum alone must never be presented as a mathematical covering certificate. The analytic bounds above allow the finite GT's **ideal geometric support** to be certified where complete grids are retained, but fully certified **implementation output** also requires a bounded numerical error analysis.

For the broader conceptual explanation see [Ground Truth in MoeaBench](../../gt.md). For previous negative evidence and experimental allocation tradeoffs see [DPF5 branch allocation audit](gt-dpf5-branch-allocation-issue119.md) and [published DPF5 Pareto proof](gt-dpf5-global-pareto-proof-issue119.md).
