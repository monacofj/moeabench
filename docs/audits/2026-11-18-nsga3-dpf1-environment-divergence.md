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
