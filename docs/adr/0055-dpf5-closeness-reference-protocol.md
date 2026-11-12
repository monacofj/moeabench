# ADR 0055 — Explicit DPF5 Closeness reference laws and calibration protocol

**Status:** Proposed for scientific approval; **fixed working protocol for independent validation**, **not a released baseline**  
**Date:** 2026-11-12  
**Issue:** [#81](https://github.com/monacofj/moeabench/issues/81)  
**Evidence:** [ADR 0053](0053-dpf5-mixed-dimensional-reference-measure.md), [ADR 0054](0054-dominance-constrained-one-sided-gt-blur.md), [ADR 0037](0037-plausible-q1-and-search-drive-headway-v0.13.1.md)  
**Research branch:** `research/81-dpf5-calibration-protocol`

## Question and scope

Q-Closeness is an **algorithm-independent score of geometric convergence** relative to
an explicitly calibrated, unfavorable *reference geometry*; it is **not** a classifier
of failed optimization trajectories and does not require a particular MOEA's behavior.
The benchmark MOP and K determine the calibration. Separate error dimensions are
handled by the other FAIR indicators. We keep existing frozen v0.14.0 GT/baselines
and the canonical `calibration/audit_report.html` unchanged.

For DPF5 with **M=3, D=2**, the Pareto set has two free parameters `x1,x2`.
The objective map at g=0 has a 2D patch P where `x1<=1/3` and a 1D curve D
where `x1>1/3`, independent of `x2`. The archived GT is exactly the set
of distinct normalized analytical objective vectors on the original
100×100 decision grid: **3,300 P + 67 D = 3,367** objective vectors.
Its geometrical support is valid in this bounded reconstruction; its
empirical row frequencies are not the original decision-space distribution.

## Working scientific choice A: independent *ideal* reference

For a fresh ideal sample, independently draw `x1,x2 ~ Uniform(0,1)`,
set `x3=x1` and all later decisions to 0.5, and evaluate the actual
DPF5 analytical objective function at g=0. **Do not deduplicate** the
resulting objectives. Then normalize using the *original frozen* ideal/nadir
and reject any sample that coincides with the frozen GT.

This is the decision-parameter pushforward law

```text
mu_ideal = (1/3) mu_P + (2/3) mu_D
```

where both conditional laws are induced by uniform parameters in
the corresponding decision regions. The 2/3 weight is a declared
property of the **existing decision-grid parameterization**, not a
claim of uniform objective-area sampling. A uniform 2D objective-area
law would assign the 1D D branch zero weight, which we have **not**
chosen. An artificial 50:50 mixture is likewise **not adopted**.

For each population size `K`, define

```text
s_K = median nearest-neighbor spacing on the K-point
      farthest-point-sampling subset of the registered GT (seed 0)

epsilon_K = median_{x ~ independent mu_ideal}
            [ d(normalized F(x), GT) / s_K ].
```

The ideal-reference random seed, sample count, GT/normalization
hashes and `s_K` must accompany each experimental estimate.
A reported `epsilon_K=0` is **not** an acceptable substitute for
missing or uncertified independent ideal reference.

## Working scientific choice B: unfavorable *geometric* reference

Use the corrected one-sided **outward** (dominated-side) half-normal
GT blur from [merged PR #84](https://github.com/monacofj/moeabench/pull/84):

```text
For EACH distinct archived GT row g_i, give exactly one anchor vote.
Estimate its local PCA outward normal (or documented cone-projected
outward surrogate on degenerate boundaries).
Draw t_i = |N(0, sigma_K^2)| independently with a private seed.
Build b_i = g_i + t_i * d_i, with d_i componentwise nonnegative.
Require g_i weakly dominates b_i; at least one objective is worse.
```

This selects an **equal-weight discrete GT-anchor law** for the negative
reference. Its D branch thus has source-anchor mass **67/3367≈1.99%**.
That differs intentionally from the ideal pushforward's 2/3 mass:
they model **different distributions and different questions**. We do
**not** silently reweight D anchors by 100 to match the original
decision grid, because a controlled same-blur analysis changed
Q-Closeness by up to ~0.03032 on two optimizer trajectories.
The negative-reference law therefore remains the *original*
distinct-GT-anchor convention for this experiment.

The blur is **not** asserted to be a feasible decision-space point or
a plausible individual trajectory from a particular suboptimal MOEA.
It is a reproducible *geometric convergence-error reference*, purposely
independent of the evaluated optimizer.

## Intensity and Q-score mapping unchanged

Keep the original controller, for **each K**:

```text
Choose sigma_K so that
median[ d(b_i, GT) / s_K ] ≈ 2.
Initial sigma_K = 2*s_K; at most 6 multiplicative adjustments;
target relative tolerance 5% ON THE CALIBRATION DRAW.
```

The independent blur draw used to construct the failure ECDF may have
a different realized median. **Do not mistake 5% controller
convergence for a 5% guarantee for every independent realization.**
Retain the *same* k-neighbor/PCA rule, outward direction code,
nearest-GT tree and no-change GT. The Closeness failure profile is
the empirical distribution of

```text
max(0, d(b_i,GT)/s_K - epsilon_K)
```

with equal weight per distinct GT row, stored as 200 sorted empirical
quantiles under the current baseline schema. Do not mix the
positive-offset correction in the physical metric with an uncorrected
failure profile. `Q_CLOSENESS` continues using Wasserstein-1 against
the ideal zero profile and the **same corrected failure ECDF**.
No change to Q's mapping or to any other FAIR metric is proposed.

## Required validation and honest uncertainty

Validate this **specified** model independently of optimizer choice.
First, for K=40, 100, 200, test multiple independent seeds of the
continuous ideal PF (e.g., 4,096/16,384 points), and separately several
sigma-calibration seeds and independent negative blur draws per sigma.
Record `epsilon_K`, fitted `sigma_K`, calibration-draw and
independent-draw median distances, failure-ECDF quantiles (e.g. 10/50/90%)
and across-seed variability. **Do not fabricate a pass/fail
statistical threshold**: quantify variation first, then choose a
release tolerance based on evidence.

Second, repeat paired real optimizer-output audits for **the identical
P0/Pf/GT/`s_K`** under old and candidate Closeness records, checking
that all other FAIR physical and Q families remain unchanged. These
trajectories are a *sensitivity test*, **not** the source of the
reference distribution. A two-optimizer-seed pilot is **not** a
statistically certified release.

Third, only if per-K source laws, seed sensitivity, reference provenance
and Q effects are reviewed, stage a **new versioned Closeness-only**
baseline artifact with exact schema, metadata and source hashes.
Do **not overwrite** frozen `baselines_v0.14.0.json`, GT NPZs or the
canonical `audit_report`. Baseline producer v0.14.0 need not match the
current library version. Handling of the other 13 MOPs, including
invalid historical DPF2/DTLZ7 GTs and unsupported DPF3/DTLZ8,
remains a separate release gate in issue #81.

## Decision boundary

- **Fixed for the next experiment:** ideal = iid uniform decision
  parameter pushforward (D=2/3); failure = equal vote per distinct
  frozen GT point plus outward-only half-normal; median calibration
  target 2*s_K; no MOEA identity enters either reference.
- **Still subject to scientific approval:** publication of that model
  as the normative DPF5 Q-Closeness baseline, the necessary precision/
  uncertainty evidence, choice of production seeds/sample count and
  the versioned reference data. The experiments below cannot
  silently approve these matters.
- **Not attempted:** replacing the geometric failure null by a
  limited-budget NSGA-II, changing D-branch anchor weights,
  changing sigma target, changing Q's Wasserstein mapping,
  repairing other MOPs' GTs, or rebuilding the archival report.

## Evidence log

- **2026-11-12 — PROTOCOL DEFINED:** Historical source law, frozen GT
  construction, branch-mass sensitivity and algorithm-independent
  outward-only geometry verified from ADRs 0053/0054 and main code.
  Scientific options reduced to the explicit two-reference model
  above for controlled tests, without modifying canonical artifacts.
  A read-only multi-K/multi-seed validation script and unit tests
  will be added on this branch; failures and results must be
  appended here before further recommendations.


### 2026-11-12 — Multi-K, multi-seed experimental validation PASSED (but not certified)

- **PASS — [GitHub Actions #43240515113](https://github.com/monacofj/moeabench/actions/runs/43240515113):** All **9/9 focused unit tests** and the full read-only three-K scientific study succeeded. Original DPF5 M3/D2 frozen GT and NPZ were checked equal; its **3,367 objectives** reproduce exactly the 100×100 analytical Pareto-set grid after objective deduplication. Verified that every generated negative sample was componentwise dominated by its own GT anchor and that sigma calibration used the **unchanged six-iteration 2*s_K / 5%-fit** criterion. The full experiment output is preserved as Actions artifact `issue81-dpf5-dual-reference-multiK-seed-uncertainty`, ID **13954731198**. No source FAIR changes and no official baselines, GT packages or audit report changed.
- **Design:** For each K in **40,100,200**, sampled two ideal-reference sizes `N=4,096` and `N=16,384`, **four independent ideal seeds** 123–126. For the failure model, ran **four independent sigma-fitting seeds** 1100–1103 and then **four negative-reference seeds** 1600–1603 reused at each fitted sigma (**16 paired sigma×noise combinations per K; the four underlying noise sequences are reused, not 16 independent realizations**). Recorded the source GT and sample SHA256 values, K-specific frozen `s_K`, `epsilon_K`, fitted `sigma_K`, controller-fit median, independent-ECDF realized median, and 200-order-statistic corrected ECDF p10/p50/p90 by seed. No optimization algorithm was used in deriving the references.
- **Independent ideal, N=16,384, across four seeds:**

  | K | `epsilon_K` minimum | median | maximum | seed range |
  |---:|---:|---:|---:|---:|
  | 40 | 0.0305055524 | 0.0307632043 | 0.0308554907 | 0.0003499383 |
  | 100 | 0.0520999461 | 0.0525399856 | 0.0526976001 | 0.0005976540 |
  | 200 | 0.0747230108 | 0.0753541261 | 0.0755801807 | 0.0008571700 |

- **Fitted sigma across four seeds:**

  | K | minimum | median | maximum | range |
  |---:|---:|---:|---:|---:|
  | 40 | 0.447369081 | 0.459516554 | 0.474310932 | 0.026941851 |
  | 100 | 0.261943475 | 0.269056062 | 0.277718464 | 0.015774989 |
  | 200 | 0.182637728 | 0.187596916 | 0.193636697 | 0.010998969 |

- **Held-out negative ECDF fit×noise combinations:** At each K, the 16 paired blur outcomes had **raw median normalized distance 1.940172–2.146925**, with median **2.035256**. **14/16** were within the controller's ±5% band, **2/16 outside**, although all four sigma-fitting seeds had met the original controller's convergence criterion. These out-of-band final draws are **observed sampling variability**, not a silent failure of the sigma-fitting loop. The corresponding corrected ECDF p50 spans were K40 **1.908602–2.118409**, K100 **1.886825–2.096633**, and K200 **1.864011–2.073819**. The median-of-four ideal epsilon was used for these ECDF-only sensitivity values; **it has not been selected as a released estimator**.
- **Important non-independence caveat:** Because the same frozen GT, ideal-reference seeds, sigma-fitting seeds and blur seeds are reused for K=40,100,200, the **normalized negative median range is mathematically the same** after sigma scales with `s_K` (up to floating-point variation). There are **16 fitted-sigma×blur combinations per K, not 16 statistically independent final blurs**: four blur seeds are deliberately reused across four fitted sigmas (a paired *common-random-numbers* design), so those runs share underlying Gaussian variates. The same four sequences recur for the three K values. Therefore **neither 16 nor 48 is an independent-sample count for formal uncertainty inference**. The three-K comparison tests the *scale-normalization protocol*, not three independent stochastic data sets. The observed 14/16 frequency must **not** be interpreted as a population success probability or a statistical confidence interval.
- **Interpretation:** The explicitly specified two-reference model passes reproducibility and geometry tests, but these finite-sample min/max spans are not confidence intervals and **no quantitative release-precision tolerance has yet been approved**. A next scientific release decision should choose (a) a priori reference sample count and seeds, (b) a reproducible way to aggregate/calibrate sigma and final ECDF under independent draws, (c) the acceptable stability of `Q_CLOSENESS` on a broader paired real-MOEA experiment panel, and (d) a separately versioned Closeness-only calibration file. Only then might the DPF5 model transition from **Proposed** to **Accepted for publication** under issue #81. The other 13 MOPs remain outside this DPF5 experiment.



### 2026-11-12 — Independent held-out blur noise recheck PASSED; controller tolerance is not final-realization tolerance

- **DESIGN REFINEMENT:** The preceding pilot used a *common-random-numbers* comparison (the same four blur-seed streams at each of four independently fitted sigmas), which is useful for paired sigma sensitivity but statistically couples the sixteen outcomes. Retained that mode for exact reproduction, and added `--independent-noise` as an **explicit second experimental design**, using `blur_seed = base_seed + 1000*fit_index + 10000*K`. The code and a new unit test assert **48 distinct blur seeds across the full 3K×4fit×4blur grid**. The four sigma-fitting seeds remain reused across K, so this is **not** 48 independent full calibration procedures.
- **PASS — [GitHub Actions #43242022548](https://github.com/monacofj/moeabench/actions/runs/43242022548):** **10/10 focused tests** and the full read-only 3K independent-noise replication passed; the frozen DPF5 3,367-point GT, source law, sigma target, normalization and original baselines were preserved. Evidence JSON artifact `issue81-dpf5-distinct-negative-noise-multiK`, ID **13955386368**, includes all per-K fit sigmas, blur seeds, SHA256, independent final median values and corrected ECDF quantile summaries. No official baseline, GT or canonical `audit_report` changed.
- **Four sigma-fit seeds × four independently chosen negative-noise seeds for each K** yielded:

  | K | Held-out raw median distance range (in `s_K`) | Held-out median-of-medians | Within controller's ±5% target band | Corrected ECDF p50 range |
  |---:|---:|---:|---:|---:|
  | 40 | 1.8768795–2.1233402 | 1.9957449 | 13/16 | 1.8433235–2.0974643 |
  | 100 | 1.8642009–2.0990860 | 1.9784487 | 15/16 | 1.8138013–2.0434063 |
  | 200 | 1.8997104–2.1308425 | 2.0203043 | 13/16 | 1.8221943–2.0544054 |

  All sigma fits retained their original 5% **controller draw** acceptance. `ideal_res` and fitted sigma ranges were unchanged from the earlier paired study, since only held-out blur noise streams were changed. Under the independent-noise design, normalized negative median ranges **no longer coincide across K**; they did so previously only because the same Gaussian draws and proportional sigma/`s_K` scaling were reused.
- **Interpretation (not a new normative threshold):** This confirms the distinction between a converged `sigma_K` controller on one draw and finite-sample spread of the *published* negative ECDF on another. Across 48 distinct final noise sequences, 41 fit×noise outcomes happened to lie within the 5% controller band; this frequency **is descriptive, not a binomial success-probability estimate**, because sigma-fitting seeds and the registered GT are reused across K and there are only four full fitting seeds. The empirical min/max spans above are **not confidence intervals** and cannot alone justify a release tolerance.
- **Remaining release question:** Whether to use a predeclared fixed seed, a pooled failure-reference distribution over independent seeds, or another reproducible aggregation/uncertainty protocol for the 200 stored ECDF order statistics is still **scientifically undecided**. Whichever rule is chosen must be accompanied by paired real-optimizer Q sensitivity and a separately versioned Closeness-only record; neither the original frozen baseline nor the canonical audit report is overwritten here.


### 2026-11-12 — Next validation gate specified (design only; not executed)

The two completed studies quantify variability of the **reference construction**, not the resulting variability of Q-Closeness over real optimization outputs. Before selecting any release estimator or precision threshold, the next experiment must compare **the same final populations and original GT** against several independently realized candidate negative ECDFs. This avoids certifying a reference merely because its realized median happens to be close to 2.

1. Use a fixed, documented panel of **authentic** optimizer runs (including multiple seeds and, where useful, distinct MOEAs/budgets), preserving each run's actual P0 and Pf. For each run, cache its objective vectors and check they are byte-identical across counterfactual reference evaluations. Do not generate new runs as a substitute for missing historical P0.
2. For K=40,100,200, construct independent *complete* calibration replicates: independently drawn ideal PF sample, independent sigma-fitting noise, and an independent held-out final blur, without reusing a random stream between replicates. Record all seeds, GT/normalization/source hashes and each resulting 200-value corrected ECDF. The previous 48 **distinct held-out streams** must not be presented as 48 complete independent calibrations.
3. Evaluate the **identical real populations** against each candidate calibration replicate, preserving the current Wasserstein-1 Q mapping. Report absolute and signed variation in Q-Closeness, its rank/order stability over the matched optimizer panel, and the comparison against the historical frozen baseline. Verify exactly that all other FAIR physical/Q families, GT and s_K remain invariant.
4. Compare, as research alternatives only, a single predeclared full calibration draw and a pooled/aggregated collection of independent full draws. Any aggregated ECDF must remain monotone, be reproducible and document the precise construction of the 200 stored order statistics. Do **not** substitute averaging sigma for averaging ECDFs without demonstrating equivalence; they need not be equivalent.
5. Quantify uncertainty at the level of **complete independent calibration replicates**, not reused Gaussian streams or paired K cells. Derive any acceptable Q stability threshold from observed decision/ranking sensitivity and a separately justified use-case criterion. Do not invent a 5% acceptance tolerance for Q from the sigma-controller's 5% fit rule.

**Gate status:** protocol specified; no new computation performed in this entry, no estimator/threshold/source law accepted for publication, no baseline or canonical audit report changed. Issue #81 remains open and Draft PR #86 remains research-only.
