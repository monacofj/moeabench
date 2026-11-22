> **Historical investigation, archived 2026-11-22.** This report retains the original provisional diagnosis. The subsequent controlled experiment documented in [issue #92](https://github.com/monacofj/moeabench/issues/92) reproduced the historical numerical reference using pinned NumPy CPU dispatch and OpenBLAS Haswell settings; the current testing contract is documented in [docs/test.md](../test.md). Do not interpret the original “unresolved” header as the current project status.

# NSGA3 × DPF1: controlled cross-run numerical divergence audit

Date: 2026-11-18. Status: **unresolved; regression gate retained**.

## Evidence sources

- [Passing numerical-basic CI run 44307380840](https://github.com/monacofj/moeabench/actions/runs/44307380840), numerical-basic 18/18.
- [Failing numerical-basic CI run 44307597870](https://github.com/monacofj/moeabench/actions/runs/44307597870), 17/18; only `NSGA3 × DPF1` failed.
- [Controlled historical comparison run 44306750837](https://github.com/monacofj/moeabench/actions/runs/44306750837): `main`, `old-fair`, and PR #91 all yielded two final nondominated points under the same pinned environment; historical frozen expectation has three.
- [Single-thread and frozen-direction probe run 44311637132](https://github.com/monacofj/moeabench/actions/runs/44311637132).

All reference runs used Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, pymoo 0.6.2; the NSGA3 evolutionary seed was 12345, and the reference-direction component seed was 3748133013. Full CI artifacts, including generation vectors, are available in the run records.

## Exact paired-run analysis (success vs failure)

The success and failure produced different energy reference-direction coordinates with **maximum absolute delta 8.881784197001252e-16**, affecting 41 out of 72 coordinates. Both direction arrays become **identical** when rounded to 13 decimal places. The recorded successful and failing directions had sorted-row SHA256 fingerprints:

- Success: `d814a5d2313c9230e9ba1cfd1844ac6f90ced8e991ff99008e70191f3ace1d66`
- Failure: `bc8dcaabadfb23a5c8de5d1f8308a96f72f05bd714beb4554719408d8892bb4f`

Generation 0 recorded **bit-identical decision variables and objective values**. The first differences occurred in generation 1 and were minute: maximum absolute objective difference **2.842170943040401e-14**, maximum variable difference **1.1102230246251565e-16**. Through generation 6, maximum absolute objective differences remained below `1e-12`, with no component differences above `1e-7`. By generation 7, the objective maximum absolute difference was around **73.6**, with **17 of 24 population members** differing by more than `1e-7`; the non-dominated front counts were 4 versus 3. In generation 11 the front counts were 3 versus 2. The **major evolutionary bifurcation** thus occurs at generation 7, with small numerical differences observable earlier.

The rounding diagnostic used `np.round(..., decimals=13)` at the MoeaBench NSGA3 wrapper's reference-direction generation call. In each inspected CI environment, it produced **exactly the same recorded trajectory as the corresponding unrounded run**, so rounding is not currently a supported fix.

## Frozen-direction / single-thread diagnostic

A diagnostic-only probe embedded the **lexicographically sorted set of direction vectors** recorded by the successful run; these are not a certified baseline and are sorted rather than guaranteed to reproduce the original generator's row order. No production algorithm or regression fixture was changed.

In CI run 44311637132, the default numerical runner and a separate BLAS-single-thread runner both produced the failing two-point result. On both runners, the generated direction fingerprint was the failing `bc8d...` value, and substituting the sorted frozen set from the successful run resulted in the **same** two-point final front. All generation population hashes matched between those two runners. The default job used an Azure runner kernel and the single-thread job an AWS runner kernel; NumPy-reported SIMD capabilities differed (AVX512-SPR available only on the latter). This particular control did not eliminate the divergence.

This does **not** demonstrate that direction generation or BLAS behavior can never cause drift. It shows that the tested direction replacement, thread setting and host combination did not correct it. The generation-by-generation dynamics, additional numerical operations, population selection/tie handling, and exact inputs to DPF1 evaluation remain candidate sources.

## Next controlled experiment

Evaluate fixed generation-1 decision vectors captured in the successful run directly through `DPF1.evaluation`, with no optimization process. Compare outputs against the recorded reference, and observe vector NumPy cosine, scalar `math.cos`, and `np.dot` versus explicit projection arithmetic. Use the read-only script `scripts/audit/probe_dpf1_fixed_population.py` in default and single-thread CI jobs. If evaluations match across environments, investigate selection, variation and numerical sorting in pymoo. If they differ, isolate mathematical primitive/library dispatch. This is **diagnostic evidence only**.

**Guardrail:** no weakening, skipping, or deletion of regression tests; no reference or certified-manifest refresh; no production NSGA3 rounding or seeding changes until the numerical cause is established and reviewed.

## Isolation of first diverging operation (CI run 44313665835)

The pinned-environment CI run [44313665835](https://github.com/monacofj/moeabench/actions/runs/44313665835) passed the numerical-basic suite (3 final points) in its default reference job, while its separate single-thread diagnostic job produced 2 points. Both jobs ran the same unmodified NSGA3/DPF1 configuration and traced the real pymoo operators without changing their outputs.

The **very first SBX crossover** received identical input arrays (SHA256 prefix `117951bb05a7`) and identical reported generator state before (`6bf88bb2cc9b`) and after (`ef914b729269`) the call. Yet the produced arrays differed: default job output prefix `7ebb5ff46d6f`; single-thread job output prefix `e25e371e6450`. The subsequent polynomial mutation received these different outputs, and its reported random-generator states also agreed. This localizes the first detected numerical discrepancy to the SBX crossover's arithmetic or another hidden runtime input, **before DPF1 evaluation**.

A separate fixed-input experiment evaluated 24 decision vectors archived from the passing generation 1. Both jobs reproduced all 72 recorded DPF1 objective values **bit-for-bit**, including the projected redundant objective. NumPy cosine and scalar `math.cos` agreed for those inputs; `np.dot` and explicit multiplication-plus-addition differed locally by at most `1.4210854715202004e-14` but the actual DPF1 output matched the historical observation exactly. The fixed-input result supports that the observed cross-run divergence begins in variable generation, not when the same variables are passed into DPF1 in the tested environments.

**Remaining diagnosis:** compare the exact first SBX output values across the success/failure jobs (not only their hashes), test whether host-specific SIMD or transcendental functions explain the difference, and determine how the sub-ulp variation propagates into selection at generation 7. No rounding or production change is justified solely by these observations.

## Exact isolated SBX replay and scalar-power control (2026-11-18)

Source review of `pymoo==0.6.2` identified multiple `np.power` calls in the Simulated Binary Crossover calculation. A new **diagnostic-only** replay captured the first 2×12×6 parent array and complete NumPy PCG64 state at the start of the crossover. Replaying the actual `SBX._do` with precisely those inputs and state reproduced the recorded vectorized output bit-for-bit.

In the failing environment, the NumPy-vectorized result had SHA256 `e25e371e64507004408a6234839e828a61409c982d84b54035eadb88d81e5ad5`. Temporarily evaluating the same crossover's power operations with scalar `math.pow` instead produced SHA256 `7ebb5ff46d6fe547fc76e32690b10bc1b09a4e288b81fa876dcd82ed12f9e507`: this **exactly equals the first SBX output previously recorded in the passing CI run**. The maximum absolute element difference was `5.551115123125783e-17`. This is direct evidence of an environment-sensitive floating-point computation inside pymoo's SBX implementation, independent of stochastic seed selection.

A separate whole-trajectory diagnostic, [run 44315326465](https://github.com/monacofj/moeabench/actions/runs/44315326465), performed three otherwise identical twelve-generation NSGA3/DPF1 runs in both the default and single-thread numerical jobs:

| Diagnostic mode | Final point count | Sorted final-front SHA256 | Same in both tested jobs? |
|---|---:|---|---|
| Unmodified pymoo operators | 2 | `2601f31848814a6d7f818bddceddcc6dde8601da7373d5ce9cd477e51f393f43` | Yes |
| Scalar power **only within SBX** | 3 | `e5ff34be8acb8036db9c994295dc5345a27173d60aa20b3f8ec417c96a8b9f51` | Yes |
| Scalar power within SBX and polynomial mutation | 3 | `a302cf44654205e6445595816ca56b97b8e7b9880f9eb8f27757c679d31481f5` | Yes |

The certified historical successful front is `77dc14fcde9e939ad162ec4513254a5809a5dd6b76fd32e7a58e8c5b97205086` (3 points). **Neither diagnostic scalar mode reproduces its bit-level values.** The experiment therefore supports `np.power` inside SBX as a demonstrable source of the initial drift, while it does **not** establish a production-compatible replacement. Matching only the nondominated point count is insufficient to certify reproducibility.

All scalar-power substitutions used temporary Python mocking inside the diagnostic script; the production pymoo adapter, mutation policy, existing regression tests, seeds and golden references remain unmodified.

**Release decision gate:** do not adopt scalar `math.pow`, relax the front comparison, or refresh reference values solely to obtain a green check. A production change would alter solver numerical behavior and needs explicit scientific review of impact across MOEAs/MOPs, reproducibility guarantees, external reference compatibility and applicable ADRs. PR #91 remains draft.
