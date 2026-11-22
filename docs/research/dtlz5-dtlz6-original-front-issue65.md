<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Original DTLZ5/DTLZ6 complete Pareto front — research protocol (#65)

**Status:** research design; not a certified complete Pareto-front generator.  
**Scope:** original equations for M >= 4; no production/API/ADR/regression changes.

## Why the existing reference is incomplete

The original implementations in `moeabench/mops/DTLZ5.py` and `DTLZ6.py` return a deterministic Pareto-set sample with `g=0`. This subset has the intended degenerate curve geometry but is not the complete Pareto front for M >= 4. The issue is known in the scientific literature, notably H. Ishibuchi et al., *Pareto Fronts of Many-Objective Degenerate Test Problems*, IEEE Transactions on Evolutionary Computation 20(5), 2016. The original equations and subsequent published analyses identify Pareto-optimal regions away from `g=0`. The historical warning in both `ps()` implementations did not make these samples complete GT; the canonical methods now fail closed for M>=4.

## Exact parameterization of the original decision-space image

Let `K = N-M+1`, `g >= 0`, `u_1 = x_1`, and `u_i = x_i` for `i=2,...,M-1`, each in `[0,1]`. The angular parameters are

- `theta_1 = pi*u_1/2`;
- `theta_i = pi*(1+2*g*u_i)/(4*(1+g))` for `i=2,...,M-1`.

The objective vector is the usual nonnegative spherical DTLZ map scaled by `(1+g)`, with these angles. In particular, for a fixed `g`, the allowed angular interval is `[pi/(4*(1+g)), pi*(1+2*g)/(4*(1+g))]` for every `theta_i` with `i>=2`; the full interval is attainable.

For **DTLZ5**, `g = sum_{j=M}^N (x_j-1/2)^2`; the **exact feasible interval** is `[0,K/4]`. A constructive realization of any chosen `g` in the interval is `x_j = 1/2 + sqrt(g/K)` for all distance variables.

For **DTLZ6**, `g = sum_{j=M}^N x_j^0.1`; the **exact feasible interval** is `[0,K]`. A constructive realization of any chosen `g` in the interval is `x_j = (g/K)^10` for all distance variables.

Thus a future source generator can parameterize the full *feasible objective image* using `(g,u_1,...,u_{M-1})`, without an optimizer and without relying on an incorrectly assumed `g=0` manifold. **Feasible image is not the same as the Pareto front**: an independent criterion is required to identify all globally nondominated regions and prove coverage.

## Preliminary independent diagnostic (not a Pareto-optimality proof)

A deterministic NumPy-only check, using the same published spherical/angle equations but not calling the MoeaBench sampler, used `M=4`, `K=10`, and a fixed seed `41`. It drew `6000` samples with `u_1` uniform in `[0,1]`, `g` uniform in `[0.001,0.4]` and the remaining two position coordinates uniform in `[0,1]`. It checked each candidate for dominance against `2001` evenly spaced points of the exact `g=0` parametrized curve. Of the 6000 samples, 190 were **not dominated by any of those finite curve samples**. This diagnostic is reproducible with the below code. It is only evidence that the frozen curve is an inadequate *finite* comparison set; it **does not** establish nondominance versus the entire continuous `g=0` curve, much less the full feasible image.

```python
import numpy as np

def f(m, u1, us, g):
    th = np.r_[np.pi*u1/2,
               np.pi*(1+2*g*np.asarray(us))/(4*(1+g))]
    z = []
    for i in range(m):
        v = 1+g
        if i < m-1:
            v *= np.prod(np.cos(th[:m-i-1]))
        if i > 0:
            v *= np.sin(th[m-i-1])
        z.append(v)
    return np.asarray(z)

m = 4
curve = np.array([f(m, t, [0.5, 0.5], 0)
                  for t in np.linspace(0, 1, 2001)])
rng = np.random.default_rng(41)
u1 = rng.random(6000)
g = rng.uniform(0.001, 0.4, 6000)
us = rng.random((6000, 2))
candidates = np.array([f(m, t, u, h)
                       for t, u, h in zip(u1, us, g)])
dominated_by_curve = np.array([
    np.any(np.all(curve <= p + 1e-12, axis=1) &
           np.any(curve < p - 1e-12, axis=1))
    for p in candidates
])
print(np.count_nonzero(~dominated_by_curve))  # 190
```

## Exact continuous g=0-curve dominance test (new)

The older 2,001-point discrete diagnostic can now be strengthened to an **exact continuous-curve statement**, independently of grid density. For the intended `g=0` Pareto curve, write

```
C(t) = (a_1 cos(t), ..., a_{M-1} cos(t), sin(t)),  0 <= t <= pi/2,
a_1 = 2^(-(M-2)/2),
a_j = 2^(-(M-j)/2) for j=2,...,M-1.
```

For any nonnegative objective vector `y`, a point on this **entire continuous curve** weakly dominates `y` if and only if

```
y_j >= a_j * sqrt(1 - min(y_M,1)^2)   for every j=1,...,M-1.
```

**Proof.** `sin(t) <= y_M` restricts `t <= arcsin(min(y_M,1))`. Since all preceding curve coordinates are positive multiples of the decreasing function `cos(t)`, the largest feasible angle minimizes them simultaneously. Therefore the displayed inequalities are necessary and sufficient. For `y_M>=1`, the endpoint `t=pi/2` dominates the nonnegative vector.

Applying this criterion to the same fixed-seed 6,000-sample M=4 DTLZ5 diagnostic gives **190 points not dominated by any point on the continuous g=0 curve**; the earlier finite-curve result was not an artifact of the 2,001-point discretization. One explicit witness uses `u=(0.173406,0.976618,0.066887)`, `g=0.328135`, with objectives approximately `(0.5893373,0.4183464,1.0554343,0.3573078)`. It violates the required inequality in objective 2 by approximately `0.04865`. The same `(g,u)` objectives are feasible in original DTLZ6, using the appropriate original decision-space realization. The independent continuous-oracle test is in `tests/unit/test_issue65_continuous_curve_oracle.py`.

**Crucial qualification:** non-domination by the g=0 curve is **not** proof of global Pareto optimality against all feasible `g>0` points. A necessary/sufficient global characterization remains outstanding.

## Proven reduction: only minimal-g radial-boundary points can be Pareto optimal

For any feasible candidate, write `theta=(theta_1,...,theta_{M-1})`. For `i>=2`, the original formula permits angles in

```
pi/[4(1+g)] <= theta_i <= pi*(1+2*g)/[4(1+g)]
```

(the interval centered on `pi/4`). Define

```
q(theta) = max_{i=2,...,M-1} abs(4*theta_i/pi - 1)
g_min(theta) = q(theta)/(1-q(theta)).
```

Within the original DTLZ5/DTLZ6 feasible ranges, **`g_min(theta)` is exactly the least feasible g for that angular direction**. Its value is at most the original candidate's g, and every angle remains realizable at that lower value. A point with `g > g_min(theta)` is *strictly dominated* by the feasible point at the same angular direction and `g_min(theta)`, because its objectives are a positive multiple of `1+g`.

**Consequences:** every Pareto-optimal point lies on the minimal radial boundary `g=g_min(theta)`. In particular, if `g>0`, **at least one** `x_i` with `2<=i<=M-1` is exactly `0` or `1`. This eliminates all interior-angle nonminimal-g candidates from the true Pareto-front search. The full-front membership question is reduced to checking global dominance *among the minimal-radius boundary points*, not the original full decision-variable hypercube.

For the earlier illustrative witness `(u_1,u_2,u_3)=(0.173406,0.976618,0.066887)`, `g=0.328135`, we obtain `g_min≈0.3080628989`, `(u'_2,u'_3)≈(1,0.04563928)`. The smaller-g feasible point has **all objectives reduced by approximately 1.5113%**. Thus this witness, although not dominated by *any* point of the continuous `g=0` curve, **is not Pareto-optimal**. This demonstrates why comparing only with the degenerate curve cannot certify a complete front. A focused independent test implements this construction in `tests/unit/test_issue65_minimal_radius_oracle.py`.

This result is a **necessary condition**, not a sufficient condition. Points on the minimal-radius boundary may still dominate one another; a complete certified sample must establish that remaining dominance relation and coverage.

## Safety contract now active

As an immediate correctness measure, original `DTLZ5.ps()/pf()` and `DTLZ6.ps()/pf()` for `M>=4` **raise `NotImplementedError`** instead of returning a known-incomplete GT. `evaluation()` remains unchanged and callable. The explicitly obsolete regression asserting legacy g=0 output and the historical warn-only safeguard were replaced with fail-closed tests under the maintainer's authorization. ADR 0006 documents this scientific exception. This is not completion of the full-front implementation.

## Required steps before replacing canonical PS/PF

1. **Characterize global membership.** Derive necessary/sufficient Pareto-optimality conditions in the entire feasible `(g,u)` domain, using original/follow-up papers. Finite nondominance against a grid is insufficient. Validate boundaries, mixed intrinsic dimensions, high-dimensional cases and both DTLZ functions separately.
2. **Construct deterministic objective-space samples.** Cover every proven Pareto component, relevant boundary and extreme; define measure/weighting if component dimensions differ; examine resolution-dependent indicators.
3. **Verify independently.** Use separate mathematical evaluators and continuous or certified numerical dominance/coverage bounds, multiresolution tests, and independent code paths. Then investigate IGD/IGD+ and HV impacts.
4. **Review public behavior.** The current `tests/unit/test_dtlz_hypersphere_sampling.py::test_dtlz5_preserves_legacy_degenerate_pareto_set_sampling` expressly freezes the incomplete sample. Revising canonical `ps()/pf()` requires **prior maintainer authorization for any test change**, applicable ADR review (notably ADR 0006 and GT sampling decisions), and a controlled transition of numerical references/certifications. No existing baseline may be silently widened or overwritten.
5. **JOSS readiness.** Until completion, state explicitly that M>=4 DTLZ5/6 original `ps()/pf()` represent only the `g=0` subset; do not claim complete GT or validate IGD/IGD+/absolute-HV against that subset as the original mathematical truth.

References: [Ishibuchi et al. (2016) author manuscript](https://ci-labo-omu.github.io/assets/paper/pdf_file/multiobjective/IEEE_TEVC_Accepted_Paper_IMN.pdf); [pymoo DTLZ equations](https://www.pymoo.org/problems/many/dtlz.html); [issue #65](https://github.com/monacofj/moeabench/issues/65).
