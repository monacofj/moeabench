# Conventional metric numerical references v1 — issue #97

**Scope:** fixed-input numerical stability, unrelated to MOEA search trajectories.
The frozen values live in `tests/conventional_metric_reference_v1.json` and are
checked by `tests/test_stability_metrics.py` in numerical stability **basic**.

## Independent derivation

Every objective is minimized. The external reference defines the reference
front for GD, IGD, GD+, IGD+, EMD and the fixed ideal/nadir box for HV.
Distances use the `[0,1]` unit ranges in each objective. In the normalized
HV geometry the dominated reference corner is `(1.1, …, 1.1)`.

### Two objectives

Reference `R = {(0,1), (1,0)}`; evaluated front
`P = {(0.2,0.8), (0.8,0.2)}`.

- GD and IGD: each closest-pair Euclidean distance is
  `sqrt(0.2² + 0.2²) = sqrt(0.08)`.
- GD+ and IGD+: each nearest **positive-part** distance is `0.2`.
- EMD: each marginal comprises two equally weighted values shifted by
  `0.2`, so the mean 1D Wasserstein distance is `0.2`.
- Raw HV: two rectangles of area `0.9×0.3` with common area
  `0.3×0.3`: `2×0.27 − 0.09 = 0.45`.
- HV of the reference front: `2×(1.1×0.1) − 0.1² = 0.21`.
  The relative value is `0.45/0.21 = 15/7` (greater than 1; not clipped).
- For **absolute** HV, a synthetic MOP is calibrated with
  `GT = {(0,1),(sqrt(1/2),sqrt(1/2)),(1,0)}`. These three points lie
  on the positive unit circle; the exact GT HV is
  `0.21 + (1 − sqrt(1/2))²`. Thus
  `H_abs = 0.45 / [0.21 + (1 − sqrt(1/2))²]`.
  This independently protects the distinction between geometry `ref`
  and MOP-owned GT.

### Three objectives

Reference `R = {(1,0,0),(0,1,0),(0,0,1)}`; evaluated front
`P = {(1/4,1/4,1/4)}`. The synthetic evaluated point tests the metric
in isolation and need not be produced by an actual MOEA.

- GD/IGD: distance from P to each reference axis is
  `sqrt((3/4)² + 2×(1/4)²) = sqrt(11)/4`.
- GD+/IGD+: after keeping only worsened objective coordinates for
  minimization, each nearest pair has two nonzero terms of `1/4`,
  giving `sqrt(2)/4`.
- EMD: each marginal of R contains `{0,0,1}`. The Wasserstein cost
  from constant `1/4` is `(1/4+1/4+3/4)/3 = 5/12`.
- Raw HV of P: `(1.1−0.25)³ = 0.85³`.
- HV of R: by three-set inclusion/exclusion,
  `3×(0.1×1.1²) − 3×(0.1²×1.1) + 0.1³ = 0.331`.
  Both relative HV (reference R) and absolute HV (synthetic MOP GT=R)
  are `0.85³/0.331`.

Identical `P=R` inputs form an additional frozen zero-distance case.
This exercises all five distance indicators with the mathematically exact
answer zero.

## Reproducibility and acceptance

The expected values are computed by elementary geometry in this document,
**not** by regenerating them from MoeaBench or pymoo. Their values are committed
as explicit constants to detect subsequent accidental changes. The fixed
float64 comparisons use `atol=1e-10`, `rtol=0` for low-dimensional
elementary formulas; this threshold is an explicit **new** check, and does
not replace or relax a historical regression test.

Approximate HV uses `mode='fast'`, 80,000 samples and seed 23. It checks
same-environment repeatability and agreement with the exact geometric value
within an absolute `0.05` **estimation** allowance. It deliberately does
not freeze one platform's Monte Carlo bits as universal mathematical truth.

The two-dimensional/three-dimensional fixed-input tests do not launch an MOEA
or consume a run seed. The historical NSGA3×DPF1 drift remains in issue #92.
Source/backend libraries for the tested implementation are pymoo (GD/IGD(+)
and exact HV), SciPy (marginal Wasserstein) and moocore (approximate HV).
Versions for release certification are governed by ADR 0050, not inferred
from these algebraic references.

**Review status:** mathematically derived, added as candidate baseline for
review. An independent maintainer sign-off is still required to claim an
independently *reviewed* certified numerical reference. No existing frozen
reference, calibration sidecar, performance baseline or historical regression
assertion is overwritten.
