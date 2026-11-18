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
