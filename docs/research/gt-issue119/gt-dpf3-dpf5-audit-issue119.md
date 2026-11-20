<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3/DPF5 finite-sample PF diagnostic — issue #119, PR #139

**Status:** reproducible read-only evidence. **No global PF mathematical certification** and no authorization to change the production samplers or GT policy.

## Reproducible execution

Actual repository MOP implementations, Python 3.12, branch `research/119-dpf3-dpf5-branch-diagnostics`, Actions [run #44617511658](https://github.com/monacofj/moeabench/actions/runs/44617511658), [raw JSON artifact #14576056220](https://github.com/monacofj/moeabench/actions/runs/44617511658/artifacts/14576056220). All six new focused tests, existing regression suite and separate numeric stability workflow passed.

```bash
python -m pytest tests/unit/test_gt_dpf_branch_diagnostics_issue119.py -q
python -m scripts.audit.gt_dpf_branch_diagnostics_issue119 --objectives 3 5 --budgets 128 512
```

## Actual MOP output summary

| Problem | M | Requested | PS/PF returned | Distinct PF | Internally dominated PF |
|---|---:|---:|---:|---:|---:|
| DPF3 | 3 | 128 | 128 / 23 | 23 | 0 (filtered) |
| DPF3 | 3 | 512 | 512 / 88 | 88 | 0 (filtered) |
| DPF3 | 5 | 128 | 128 / 23 | 23 | 0 (filtered) |
| DPF3 | 5 | 512 | 512 / 88 | 88 | 0 (filtered) |
| DPF5 | 3 | 128 | 121 / 121 | 51 | 0 |
| DPF5 | 3 | 512 | 484 / 484 | 190 | 0 |
| DPF5 | 5 | 128 | 128 / 128 | 128 | 38 |
| DPF5 | 5 | 512 | 512 / 512 | 512 | 462 |

The DPF3 `pf()` override filters the base generated objectives using nondominated sorting. It therefore returns fewer than requested and cannot be assumed aligned index-by-index with the `ps()` output. The research script verifies returned-objective membership using nearest-set matching, not row order.

DPF5 at M=3 chooses `floor(sqrt(n))` grid points per axis, explaining 121 and 484 actual outputs. For requested 512: branch P has 176 decision vectors/176 distinct objective vectors; branch D has 308 decision vectors but only **14 distinct objective vectors** (294 duplicates), corresponding to the conditional objective projection. For requested 128: P 44/44 distinct, D 77/7 distinct, yielding 51 overall unique.

DPF5 M=5 n=512: P has 183 decision/objective vectors, of which 151 are dominated *within the P subset*; D has 329 unique vectors with zero dominated *within D*. The **462** dominated points in the union show that cross-branch dominance must be separately investigated; summing within-branch dominance counts is insufficient. At n=128: P 47 with 16 within-P dominated, D 81 with none within D, and **38** dominated across the union.

## Scientific interpretation and next independent study

**Internal nondominance is necessary but not sufficient for global Pareto optimality.** Filtering the sampled set alone is not an approved remedy. The current empirical results show scientific quality defects or unresolved source-law questions, not a mathematically certified new GT.

1. Reconstruct the **original published DPF3 and DPF5 objective equations** and compare them term-by-term with current implementation, particularly the DPF5 branch-specific objective-kernel assignments for M>D+1 and the DPF3 min/max chaotic projections.
2. Derive analytical Pareto dominance and feasible PS conditions independently, including across the DPF5 P/D branch boundary. Distinguish an implementation formula defect from a sampler defect. For DPF3 determine whether omitted sampled candidates are actually off the mathematical PF rather than merely internally dominated.
3. Build a *separate* candidate reference from that independently validated geometry. Count per-component unique points, missing extremes and distances at a fixed **effective unique-objective-point budget**, reporting requested and returned counts.
4. Only then compare deterministic, cost-bounded alternatives and sensitivity of IGD/IGD+ or GT-dependent HV on **frozen MOEA objective outputs**.
5. Obtain maintainer approval before any production fix that changes existing ADR-governed contracts, historical numerical calibration references, regression tests, or release budget defaults. ADR 0006/0018/0057 remain in force.

No production `ps()`, `pf()`, MOP evaluation, `calibrate()`, calibration sidecar, frozen baseline, FAIR research, accepted ADR or existing regression test is modified in this PR.
