<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3 precision and DPF5 published spherical-kernel verification — issue #119

**Research only, no production modification.** Actual unmodified DPF3/DPF5 MOP evaluations versus independently constructed mathematical counterfactuals. No global Pareto-completeness certification.

**Evidence:** [CI #44623683461](https://github.com/monacofj/moeabench/actions/runs/44623683461), [raw three-JSON artifact #14578127730](https://github.com/monacofj/moeabench/actions/runs/44623683461/artifacts/14578127730), source `scripts/audit/gt_dpf_numerics_kernel_issue119.py`, new independent unit tests. Published equations: Zhen et al., *Multiobjective Test Problems with Degenerate Pareto Fronts*, §IV-C/E, (15), (16), (19), (20), https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf.

## DPF3: objective precision

In the original equation `gamma_1=1-cos(theta_1)` for D=2, `theta_1=(pi/2)x_1^100`. Direct floating-point subtraction loses small positive values to cancellation. The mathematically identical `2*sin(theta_1/2)^2` rescues many of them; it can itself underflow when `x_1^200` is below float64's representable range. The other components also lose resolution, notably `1-sin(theta)` rounding to one.

| M | Requested | Direct-zero | Stable-zero | Newly positive | ND actual | ND if only first component is replaced |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 128 | 106 | 4 | 102 | 23 | 45 |
| 3 | 512 | 424 | 13 | 411 | 88 | 171 |
| 5 | 128 | 106 | 4 | 102 | 23 | 45 |
| 5 | 512 | 424 | 13 | 411 | 88 | 171 |

**Interpretation:** finite-precision loss is measured; only the first objective was replaced in memory. The equality of observed M3 and M5 counts does not certify correct chaotic-threshold ordering (the separate source audit establishes it is unsorted at M-D>=2). Replacing one component is not sufficient to recover all mathematically distinct Pareto points. Before choosing a numerical remedy, measure high-precision analytic values and assess acceptable representability, evaluation semantics and the desired decision-/objective-space sampling law.

## DPF5: independently rebuilt spherical beta/gamma objectives

For analytically valid g=0 PS, published beta objectives parameterize the positive unit hypersphere of M objectives using **M−1 distinct angles**. The current implementation uses a shortened/incorrect product for beta1/beta2 and repeats beta components for M=5,D=2; the current D branch agrees with the independent gamma formula away from equality. The exact published first-branch condition is `x1 < 1/3`, while runtime currently assigns the equality point to P.

| M | Requested | P max abs sphere-norm² residual (actual) | P max norm² residual (published reconstruction) | P max component difference | Internal dominated actual / reconstructed, across union |
|---|---:|---:|---:|---:|---:|
| 3 | 128 | 2.22e-16 | 2.22e-16 | 0 | 0 / 0 |
| 3 | 512 | 2.22e-16 | 2.22e-16 | 0 | 0 / 0 |
| 5 | 128 | 0.8327941 | 4.44e-16 | 0.8457755 | 38 / 0 |
| 5 | 512 | 0.9943366 | 4.44e-16 | 0.9852624 | 462 / 0 |

D branch M5 actual/reconstructed max objective difference is zero away from the equality boundary, and observed norm² residual is at floating precision. At M3, both normal branches agree with the independently reconstructed equations.

**Boundary warning:** for M3, a grid can contain exact `x1 == 1/3`. The pure branch comparison above excludes such equality points; production currently assigns them to P, and the published piecewise rule assigns them to D. This is separately tested in `gt_dpf_equation_probes_issue119.py`. Do not conflate the norm mismatch for M5 P with the boundary-contract discrepancy.

**Scientific interpretation:** the M5 P sphere-norm failure is a powerful independent necessary-condition falsification of the original published PS/PF mapping. The corrected in-memory objectives satisfy this necessary identity, and all 462 sampled internally dominated points disappear. This is not a proof of global Pareto completeness or an approval to modify production; require complete term-by-term independent validation and an explicit impact/migration plan for scientific GT, IGD(+), HV, regression and archived references before production corrections.

## Safeguards and proposed order

1. Validate beta/gamma equations and boundary convention against primary published source/code; test multiple M,D geometries with g=0 and g>0, including continuity and which angular variables must be used.
2. Derive high-precision DPF3 analytical objective values and quantify unavoidable float64 coalescence; separate formula evaluation from objective-space sampling allocation.
3. Review accepted ADR 0006/0018/0057 and enumerate public API/numerical effects. **Any incompatible ADR change or existing regression-test modification requires advance warning and explicit maintainer authorization.**
4. Only after that review propose a correction to production; keep existing numerical references versioned, with explicit provenance and release notes.

No existing regression, ADR, calibration file, public MOP evaluation, `ps()`, `pf()`, `calibrate()` or FAIR module changed by this study.
