<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3 high-precision and DPF5 cross-dimensional follow-up — issue #119

**Status:** read-only research result. The globally correct Pareto front and release/migration policy have **not** been certified. No production changes approved.

**Provenance:** [PR #139](https://github.com/monacofj/moeabench/pull/139), CI [run #44626951719](https://github.com/monacofj/moeabench/actions/runs/44626951719), script `scripts/audit/gt_dpf_high_precision_issue119.py`, new independent tests `tests/unit/test_gt_dpf_high_precision_issue119.py`. The original MOP implementations were executed without modification. Independent published D=2 beta/gamma objective reconstruction is in `scripts/audit/gt_dpf_numerics_kernel_issue119.py`.

## High-precision DPF3 analysis (M=3,D=2)

The high-precision reference evaluates `2 sin²((π/2)x_1^100/2)` with `mpmath` at **160 decimal digits** on the exact float64 decision coordinates. This identity avoids even high-precision cancellation in `1-cos(theta)`. Each grid sample except x=0 has a strictly positive mathematical first objective.

| n | Positive high-precision f1 | Positive lost by direct float64 1-cos | Positive lost by stable float64 2sin² | Reference f1 rounded to float64 zero (includes x=0) |
|---:|---:|---:|---:|---:|
| 128 | 127 | 105 | 3 | 4 |
| 512 | 511 | 423 | 12 | 13 |

The stable double computation retains many more mathematically positive coordinates, but cannot represent extremely tiny f1 values. The exact x=0 mathematical zero is included in both direct and stable zero totals. High precision does not by itself solve geometric undersampling or the separate DPF3 eta-order discrepancy. Unit tests compare high-precision values across 120/200 digits on a smaller sample; no global PF certification results from those finite tests.

## DPF5 D=2 dimension sweep

For analytically generated g=0 PS, independently rebuild the published M-dimensional beta hyperspherical coordinates and D branch gamma; compare against the actual evaluator for strict x1<1/3 (P) and x1>1/3 (D). The x1=1/3 convention remains a separately verified discrepancy.

| M | Requested n | Max P abs(sum(f²)-1), production | Max P abs(sum(f²)-1), independent | Max D abs component difference |
|---:|---:|---:|---:|---:|
| 3 | 512 | 2.22e-16 | 2.22e-16 | 1.11e-16 |
| 4 | 512 | 2.22e-16 | 2.22e-16 | 5.55e-17 |
| 5 | 512 | 0.9943366 | 4.44e-16 | 0 |
| 6 | 512 | 1.3802877 | 4.44e-16 | 5.55e-17 |
| 10 | 512 | 0.9882569 | 4.44e-16 | 5.55e-17 |

M=3,4 match the published D=2 reconstruction in both strict branches; M>=5 P does not. The discrepancy is specifically **dimension-dependent**, while the D branch continues to match. At M=6 the squared-norm error even exceeds one, confirming that treating production P output as spherical ground truth is unjustified for these configurations.

**Limitations:** the reference is reconstructed for **D=2** only. Generalizing the result to other intrinsic dimensions requires independent proof, and g>0 evaluations need separate checks before modifying any evaluator. Satisfying the sphere identity is necessary for the claimed analytic PF mapping, but it does not alone prove global Pareto completeness or certify cross-branch dominance. Numerical data are from deterministic finite PS samples (n=128/512), not a continuous exhaustive proof.

## Next review gate

1. Independently verify published beta/gamma indexing for M>=5 and arbitrary D; check g>0 scaling and x1=1/3.
2. Determine DPF3 float64 representability and exact PF/PS sampling requirements separately from numerical evaluator accuracy.
3. Inventory all affected GT, reference metrics and regression expectations; include a versioned migration strategy.
4. Before any accepted-ADR incompatibility or change to **existing regression tests**, warn the maintainer and obtain explicit authorization. Production evaluators, ps/pf, calibrate, frozen GT, baselines and accepted ADRs remain untouched.
