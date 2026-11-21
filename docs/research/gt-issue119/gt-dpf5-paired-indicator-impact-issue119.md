<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Controlled DPF5 evaluator and reference impact — issue #119

**Status: nonmutating research only, PR #139.** The experiment does not claim that either finite sampled reference is globally Pareto-complete or constitutes an approved production replacement.

## Reproducibility and isolation

[Research CI run #44636811438](https://github.com/monacofj/moeabench/actions/runs/44636811438) **passed 53 new focused tests**, including the eight new paired-isolation assertions. Script: `scripts/audit/gt_dpf5_fixed_impact_issue119.py`, tests `tests/unit/test_gt_dpf5_fixed_impact_issue119.py`, archived JSON workflow artifact.

Deterministic MOP decision vectors are generated once for each (M,D,seed); the **same** decisions are evaluated using (i) unchanged runtime DPF5 and (ii) independent reconstructed published beta/gamma equations. The reference sample contains 64 decision points on the assumed g=0 analytical PS; a distinct set of 16 decision points at g=0.01 produces a fixed objective population using **the current evaluator**. For the reference-only indicator experiment, this *exact objective matrix is never changed* (verified with SHA256 and equality). The candidate reference arrays differ, but the front is held fixed.

Actual repository metric classes `GEN_igd`, `GEN_igdplus`, `GEN_hypervolume` are used. For HV, **both references and the fixed population** share ideal=0, nadir=2 and normalized reference point=1.1; `H_sol` is exactly identical in both calculations, while `H_GT` and `H_abs=H_sol/H_GT` can change. IGD and IGD+ use their current pymoo `zero_to_one=True` behavior with **reference-dependent internal normalization**, so differences incorporate both point positions and each reference's normalization, not purely a common-normalized geometric distance.

## Exploratory results (counterfactual minus current, nGT=64, fixed population=16)

| (M,D) | seed | max GT objective change | max objective change at fixed population decisions | ΔIGD | ΔIGD+ | Δ absolute HV ratio |
|---|---:|---:|---:|---:|---:|---:|
| (3,2) | 101 | ≈0 | ≈0 | ≈0 | ≈0 | 0 |
| (3,2) | 202 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| (5,2) | 101 | 0.85064 | 0.79861 | +0.02731 | +0.00270 | +0.02663 |
| (5,2) | 202 | 0.82500 | 0.24318 | +0.02933 | −0.00076 | +0.01801 |
| (6,3) | 101 | 0.74585 | 0.58685 | +0.00967 | −0.00020 | +0.00562 |
| (6,3) | 202 | 0.61013 | 0.70401 | +0.01959 | +0.00153 | +0.01708 |
| (6,4) | 101 | 0.59736 | 0.53020 | +0.01539 | +0.00017 | +0.00281 |
| (6,4) | 202 | 0.59938 | 0.54786 | +0.03978 | +0.01271 | −0.00845 |

The nonzero deltas at M>=5 demonstrate that **changing GT alone materially changes measured indicators even when objective-space algorithm outcomes are held fixed**. Separate fixed-decision objective differences establish an evaluator-change effect independently. These observations do not estimate revised actual MOEA performance: that would require running an algorithm with the corrected evaluator or replaying exactly fixed decision histories.

## Migration and scientific safeguards

- **Two distinct interventions must be evaluated separately:** correcting the mathematical evaluator changes the MOP and potentially all MOEA objective traces; correcting the GT changes GT-dependent measurements even on frozen objective populations. Do not lump these effects together.
- Historic DPF5 3-objective geometry may remain numerically unchanged for ordinary branch-interior samples, but strict threshold equality, duplicate GT and requested cardinality remain unresolved. No blanket exemption for `M=3` is authorized.
- Do not mutate `tests/stability_reference_gt.json`, `tests/stability_gt_geometry_reference.json`, historical DPF3 regression files, certification reference archives or any previously accepted ADR. Preserve the provenance of old references. Changes to pre-existing regression tests require advance warning and explicit maintainer approval.
- The earlier fixed DPF sampling budgets 10,000 (ADR 0018) remain runtime defaults until the distinct ADR 0057 budget policy is numerically reviewed.
- Next: provide a written migration proposal, backed by comprehensive global PF membership analysis and an inventory of all reference-sensitive interfaces, before any production PR. This Draft PR remains solely independent research.
