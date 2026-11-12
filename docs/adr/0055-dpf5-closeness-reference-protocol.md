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
- **Design:** For each K in **40,100,200**, sampled two ideal-reference sizes `N=4,096` and `N=16,384`, **four independent ideal seeds** 123–126. For the failure model, ran **four independent sigma-fitting seeds** 1100–1103 and then **four negative-reference seeds** 1600–1603 at each fitted sigma (**16 independent negative draws per K**). Recorded the source GT and sample SHA256 values, K-specific frozen `s_K`, `epsilon_K`, fitted `sigma_K`, controller-fit median, independent-ECDF realized median, and 200-order-statistic corrected ECDF p10/p50/p90 by seed. No optimization algorithm was used in deriving the references.
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

- **Independent negative ECDF realizations:** At each K, the 16 held-out blur draws had **raw median normalized distance 1.940172–2.146925**, with median **2.035256**. **14/16** were within the controller's ±5% band, **2/16 outside**, although all four sigma-fitting seeds had met the original controller's convergence criterion. These out-of-band final draws are **observed sampling variability**, not a silent failure of the sigma-fitting loop. The corresponding corrected ECDF p50 spans were K40 **1.908602–2.118409**, K100 **1.886825–2.096633**, and K200 **1.864011–2.073819**. The median-of-four ideal epsilon was used for these ECDF-only sensitivity values; **it has not been selected as a released estimator**.
- **Important non-independence caveat:** Because the same frozen GT, ideal-reference seeds, sigma-fitting seeds and blur seeds are reused for K=40,100,200, the **normalized negative median range is mathematically the same** after sigma scales with `s_K` (up to floating-point variation). Thus there are **16 distinct negative reference random realizations**, **not 48 independent trials**. The three-K comparison tests the *scale-normalization protocol*, not three independent stochastic data sets. The observed 14/16 frequency must **not** be interpreted as a population success probability or a statistical confidence interval.
- **Interpretation:** The explicitly specified two-reference model passes reproducibility and geometry tests, but these finite-sample min/max spans are not confidence intervals and **no quantitative release-precision tolerance has yet been approved**. A next scientific release decision should choose (a) a priori reference sample count and seeds, (b) a reproducible way to aggregate/calibrate sigma and final ECDF under independent draws, (c) the acceptable stability of `Q_CLOSENESS` on a broader paired real-MOEA experiment panel, and (d) a separately versioned Closeness-only calibration file. Only then might the DPF5 model transition from **Proposed** to **Accepted for publication** under issue #81. The other 13 MOPs remain outside this DPF5 experiment.

