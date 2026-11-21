<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# A rigorous finite-GT covering guarantee for published DPF3 with D=2 — issue #119

**Research candidate, not an adopted public sampler.** Independent implementation: `scripts/audit/gt_dpf3_d2_angular_cover_issue119.py`; tests: `tests/unit/test_gt_dpf3_d2_angular_cover_issue119.py`. This mathematically certifies the *ideal, real-arithmetic published DPF3 curve* under strictly increasing chaotic thresholds, **only for D=2**. The separate DPF3 global PS proof covers general D, but this finite covering theorem does not.

## Geometry and analytic bound

Let theta=(pi/2)*x1^100 in [0,pi/2]. On the exact published g=0 PS and with ordered eta thresholds, the essential D=2 objectives are

```text
gamma1(theta) = 1 - cos(theta)
t(theta) = gamma2(theta) = 1 - sin(theta)
```

The output F(theta) concatenates gamma1 and the ordered `min/max` projection vector P(t) previously proved to satisfy sum(P_i(t))=t+sum(eta). Between adjacent eta thresholds only one coordinate P_i changes linearly with t; all other P_i are constant. Thus **P is 1-Lipschitz as a vector function of the scalar t** (its path advances along orthogonal coordinate segments, so Euclidean chord length never exceeds total scalar travel), and F satisfies

```text
||F(theta)-F(phi)||_2
  <= sqrt( [cos(theta)-cos(phi)]² + [sin(theta)-sin(phi)]² )
   = 2*sin(|theta-phi|/2) <= |theta-phi|.
```

This inequality includes all projection-kink and saturated intervals.

Choose `n>=2` endpoint-inclusive **uniform angles** `theta_i=i*(pi/2)/(n-1)`, map each to a PS decision `x1=(2*theta_i/pi)^(1/100)`, and all distance coordinates x2..xN=0.5. The published global membership theorem certifies every corresponding objective F(theta_i) is on PF. Since every theta is within half an angular step of an endpoint-inclusive sample, the **exact continuous PF's directed covering radius** obeys

```text
sup_{F(theta) in PF} min_i ||F(theta)-F(theta_i)||_2
    <= pi/[4(n-1)].
```

The reverse directed PF-sample distance is zero because each sample is Pareto-optimal, so the same bound holds for the **Hausdorff distance** between the continuous PF and the **ideal exact** n-point sample, using unnormalized Euclidean objective distances. Unlike earlier finite random-probe maxima, this is a rigorous analytic bound for the published DPF3 with D=2.

For n=128, the bound is about 0.006185 objective units. For n=512, about 0.001537. The bound is independent of the number of redundant objectives M and of the ordered eta values, although the curve can contain nondifferentiable projection transitions.

## Floating-point conditions and experimental checks

The independent research implementation uses `2*sin(theta/2)^2` to compute `1-cos(theta)` stably, and the sorted paper eta. It produces exactly `n` distinct floating-point objective vectors in the studied n128/512 cases, as the first objective is strictly increasing along uniform-angle samples. It also records actual current `mop.pf(n)` returned count and maximum mismatch of the unmodified current evaluator on the candidate PS decisions.

**Critical distinction:** the proven bound is an exact-real-arithmetic statement; a complete floating-point guaranteed bound would additionally require a validated rounding-error envelope for construction and evaluation, including conversion of angular positions to x1 and x1^100 reconstruction. These are not claimed by this study. The proof also does not provide a universal n for all DPF3 D>2, nor a configured normalized-space or IGD/IGD+ quality guarantee; those are different requirements.

**No production change or calibration policy approval.** The future promotion decision must separately verify the meaning of requested vs effective n, D>2 covering, use of real-arithmetic vs float64 error margins, time/memory cost and intended `mop.calibrate()` quality tolerance per ADR0057. Old references and numeric regression fixtures remain frozen and versioned under ADR0056. Do not overwrite them to make a new evaluator pass.

## Acceptance-check design

Independent candidate evaluation should record: (1) exact Pareto membership under the published model, (2) actual cardinality, exact objective uniqueness, and deterministic PS provenance, (3) **mathematically proved directed PF-to-GT covering radius** or a clearly labeled finite probe estimate if a proof is unavailable, (4) essential objective extremes and boundary/stratum representation, (5) numerical error budget and environment, and (6) bounded wall-clock/memory cost. Automatic `calibrate()` budget selection remains pending until the maintainer approves quantitative target accuracy and maximum cost; these values should not be inferred from a particular sample.

No existing regression test, accepted ADR, MOP evaluator, PS/PF, GT archive, baseline, sidecar or FAIR implementation changed by this proof.
