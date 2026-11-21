<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3 arbitrary intrinsic-D angular covering: analytic bound and float round-trip — issue #119

**Status:** mathematical theorem for exact published DPF3 with **sorted η** and an experimental float64 realization; not a new public sampler or certified floating-point interval bound. [Research CI #44760469860](https://github.com/monacofj/moeabench/actions/runs/44760469860) PASSED **94 independent new tests**. Script `scripts/audit/gt_dpf3_nd_angular_cover_issue119.py`, seven new unit tests `tests/unit/test_gt_dpf3_nd_angular_cover_issue119.py`. Archived JSON includes each parameterization.

## Mathematical theorem

Write `q=D−1` and `theta_i=(pi/2)*x_i^100` for the free D−1 Pareto-set parameters. The published essential vector is `gamma(theta)=1−H_D(theta)` with `H_D` the nonnegative unit-hyperspherical coordinate map, and the final M−D+1 objectives are ordered saturating `min/max` projections of `gamma_D`.

The differential of `H_D` in positive-orthant hyperspherical angles has mutually orthogonal columns whose Euclidean norm is at most 1. Hence its operator norm is at most 1 for every interior angle vector. By continuity the same global Lipschitz bound extends to the closed angular cube:

```text
||H_D(theta)-H_D(phi)||_2 <= ||theta-phi||_2.
```

Complementing H_D does not change distances. With the published *increasingly sorted* eta, the redundant-projection vector follows a polygonal path whose successive segments change one objective coordinate at a time, with total Euclidean path length exactly the corresponding scalar change. Thus its Euclidean Lipschitz constant is at most 1, and the complete reconstructed F is 1-Lipschitz in theta:

```text
||F(theta)-F(phi)||_2 <= ||theta-phi||_2.
```

An endpoint-inclusive tensor grid of `k>=2` equally spaced angles on each of q dimensions has `k^q` **nominal decision points**. Every theta in `[0,pi/2]^q` lies within Euclidean distance `sqrt(q)*pi/[4(k−1)]` of one grid angle. The separate global PF membership theorem says every exact grid image is Pareto-optimal, so the **exact Hausdorff distance** between the published continuous PF and its ideal tensor-image sample obeys:

```text
d_H(PF, PF_grid) <= sqrt(D−1)*pi/[4(k−1)]
```

This proof is independent of redundant objective count M and η values if the thresholds satisfy the published strict sorting requirement. It generalizes the earlier `D=2` result. The bound refers to **unnormalized Euclidean objective distance**, and is intentionally conservative.

## Measured finite precision, samples and effective counts

| M,D | k per angle | Nominal grid rows | Exact-unique float64 objectives | Analytic real bound | Finite-probe max | High-precision angular round-trip estimate | Objective float64 round-trip max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3,2 | 33 | 33 | 33 | 0.0245437 | 0.0245243 | 8.16e-15 | 8.31e-15 |
| 5,2 | 129 | 129 | 129 | 0.0061359 | 0.0061322 | 1.00e-14 | 1.02e-14 |
| 5,3 | 5 | 25 | 23 | 0.2776802 | 0.2491901 | 8.81e-15 | 6.63e-15 |
| 5,3 | 9 | 81 | 75 | 0.1388401 | 0.1295600 | 8.81e-15 | 6.63e-15 |
| 6,4 | 4 | 64 | 45 | 0.4534498 | 0.3671144 | 1.48e-15 | 1.38e-15 |
| 6,4 | 7 | 343 | 266 | 0.2267249 | 0.1867501 | 1.08e-14 | 6.50e-15 |

High precision measures the *actual float64 PS decision coordinates* against ideal exact angular grid fractions using `mpmath` at 100 decimal digits, rather than claiming that float64 `x^100` exactly reconstructs the intended angle. The objective discrepancy compares an independently evaluated published formula at rounded PS decisions with the ideal float angular sample. These are **observed/estimated errors**, not interval-certified IEEE roundoff maxima. The independent probe maxima (1024 probes/case) are not true worst-case geometric limits; only the analytic bound above is proved.

## Important separate cardinality issue

For D>2 the spherical angular map has coordinate singularities and noninjective boundary fibers (notably at angle endpoints), so a tensor grid with k^(D−1) decision rows **need not produce k^(D−1) distinct objective-space vectors**. Some mathematically duplicate vectors can differ by floating-point rounding at a boundary; conversely, exact `np.unique` is not a proof of distinct *mathematical* points. The table shows empirically exact-unique float64 vectors, not certified mathematical cardinality. A generator promising exact *unique objective* budget must explicitly detect degeneracy, allocate/replenish samples and then re-evaluate its actual covering bound. Silently dropping duplicates without replacement preserves the sampled support but changes the effective budget.

This is separate from the DPF3 current `pf()` filtering collapse caused by source equation discrepancies and precision-sensitive domination.

## Gate for finite GT acceptance (no numerical thresholds adopted yet)

1. **Published-mathematics match:** evaluator and PS/PF map agree with independent formula across supported M,D and valid g, including boundary cases; current DPF3 and DPF5 do not meet this gate.
2. **Exact global membership:** analytic PS/PF proofs for the published models now exist for DPF3 and DPF5. Finite candidates must be generated from these proved sets.
3. **Declared cardinality:** report requested decision rows, returned rows, mathematical-equivalence considerations, exact-unique numeric objectives, and whether any deduplication changed coverage or effective budget.
4. **Coverage:** require a provable continuous covering/Hausdorff error bound in the declared objective metric when feasible, or clearly label finite-probe statistics (multiple independent seeds) as empirical, not certification. Report critical branch/stratum extremes and disconnected components.
5. **Numerical fidelity:** distinguish exact-real theorem from PS inverse-transform and objective evaluation roundoff; attach a conservative verified rounding envelope if asserting a deployed float64 covering *guarantee*. The present 100-digit comparisons are measurements and cannot replace rigorous rounding analysis.
6. **Complexity:** document the tensor-grid exponential point count k^(D−1). Automatic use for large D is impractical; future adaptive/geometry-aware algorithms require their own proof, precision and budget assessment. Record peak memory and elapsed runtime under an approved bounded environment.
7. **Versioned provenance:** store M,D,N,K, mathematical formula version, source revision, seed/sample method, numerical environment, exact count, coverage bound and precision assumptions. Do not overwrite historical GT/certification artifacts.
8. **ADR0057 calibration policy:** mathematical bounds enable *candidate* error-target → budget relationships but do not choose an error tolerance, cost cap, universal branch measure or public `calibrate()` default. All remain subject to separate maintainer approval.

**No changes** to the production MOP evaluator, `ps()/pf()/calibrate()`, existing regression tests, accepted ADRs, frozen GT, sidecars, baseline or FAIR research. PR #139 stays research-only Draft/unmerged.
