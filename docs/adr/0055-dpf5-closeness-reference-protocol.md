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


### 2026-11-12 — Full-calibration / real-Q sensitivity implementation staged (validation pending)

- Added `calibration/scripts/study_dpf5_real_q_full_calibrations_issue81.py` on this research branch. For each registered K, every calibration replicate independently draws a finite-PF ideal sample, fits sigma using a separate noise stream and generates a third held-out outward GT blur. All seed namespaces are disjoint across replicate and K in the configured experiment. The source keeps the frozen DPF5 GT, original `s_K`, the original six-step 5% **fitting** rule, 200 sorted order statistics, and the current Wasserstein Q mapping.
- The experiment obtains real P0/Pf from fresh seeded NSGA-II runs, keeps each final population fixed across all counterfactual calibrations, checks the existing physical Closeness against recomputed GT distances, records Q variation/ranking and explicit SHA256 provenance, and refuses to write to registered artifacts or the canonical report. The initial panel is NSGA-II only; it is **not** a cross-algorithm certification.
- Added focused tests for disjoint RNG namespaces, W1 correction, deterministic ranking and domain validation. A temporary branch-specific GitHub Actions pilot workflow requests K100, three complete calibrations, two real optimizer seeds, 40 generations and a JSON evidence artifact. **At the time of this log there is no verified completed workflow run, test result or real-Q numerical output**; this is *implementation staged*, not an empirical PASS. Do not adopt a calibration or scientific threshold on its basis.
- Remaining execution gates: inspect CI result and any discrepancies, expand matched real-MOEA conditions (including more than one algorithm), compare single-draw versus pooled ECDF reference with independently justified precision/decision-stability criteria, and remove the temporary CI workflow after evidence is captured. **No official baseline/GT/audit report has been changed.**


### 2026-11-12 — Extended real-Q research code, CI evidence still unavailable

- Added a second, explicitly **exploratory** candidate built by equal-weight mixing each complete replicate's **200 stored ECDF order statistics**, then extracting 200 monotone order statistics from that mixture; the physical ideal correction for real populations uses the arithmetic mean of the corresponding independently fitted epsilons. This is a *comparison estimator*, **not** an exact pool of the original 3,367 blurred rows, not an approved normative reference, and not assumed equivalent to fitting once on pooled raw draws.
- The real-Q study now computes candidate Q on unchanged authentic final populations, checks historical `Q_CLOSENESS` against a direct W1 recomputation from the registered frozen ECDF, and separately records both individual-calibration Q variation and pooled-minus-individual-mean Q. Added focused deterministic/disjoint-seed and monotone-pooling tests.
- The temporary test workflow was changed to include a `pull_request` trigger as well as `push`, since the available workflow-run query only exposes PR-associated runs. **The query still returned no verified execution for the newest commit.** No unit-test PASS, paired-Q result, ranking stability or full-calibration statistical conclusion can yet be claimed. Any unknown CI status must remain unknown; do not infer success.
- Pending: obtain actual Actions run and inspect logs/artifacts, correct failures, then expand beyond the small NSGA-II pilot and evaluate ranking ambiguity/ties before any scientific release. No protected baseline/GT/canonical report changes were made.


### 2026-11-12 — Direct-script import fix; execution access blocked

- **Static defect fixed:** direct invocation `python calibration/scripts/study_dpf5_real_q_full_calibrations_issue81.py ...` did not explicitly prepend the repository root to `sys.path` before importing the `calibration` namespace. The existing original issue81 script correctly does this. New script now mirrors that pattern (commit `8fe30d5`). The temporary research CI checks `compileall` and direct `--help` invocation before running focused pytest and the small authentic optimizer pilot (commit `1a2d6ed`).
- **Execution limitation:** a direct `git ls-remote https://github.com/monacofj/moeabench.git HEAD` attempt in the available compute container failed with `Could not resolve host: github.com`, so the repository could not be cloned for local execution. The connected GitHub interface supports file edits and reading some workflow results, but the commit-to-workflow lookup has not returned a completed run and there is no available verified workflow dispatch result. **No fresh numeric or test PASS evidence exists** for this work in this session. Do not claim that the source-level fix by itself validates the experiment.
- **Next gate:** execute the temporary branch workflow via GitHub Actions or a local checkout and inspect the emitted JSON and test logs. Preserve all failures in this append-only evidence log before selecting any baseline. Frozen data and canonical reports remain untouched.

### 2026-11-12 — Local audit invalidates pilot independence; RNG reservation corrected

- Read the complete ADR, issue #81 including its discussion, PR #86 description/discussion, both experimental producers, focused tests, and the consumed sigma/blur/Q/context implementations before editing. Working tree started clean at `dfb2495` on the requested research branch. `/tmp/issue81-full-real-q.json` was available. The original six unit tests were rerun locally: **6 passed in 3.01 s**. This verifies their assertions, not the pilot's independence claim.
- **IMPLEMENTATION ERROR:** `seed_triplet` reserved only `(base, base+1, base+2)` per replicate, whereas the unchanged `calibrate_sigma` consumes `fit_seed+it`, `it=0..5`. Its second fitting attempt therefore reused the purported held-out blur seed. Later attempts could also intersect the next replicate's streams. In the available three-replicate pilot, every final held-out median **exactly equals** its controller median: `2.026149419032174`, `2.0656822659149663`, `1.936933923810005`. The JSON's max Q range `0.008430541927552326` and stable ranking were real observations, but **not evidence from three independent complete calibrations**. Earlier entries describing stream independence are superseded by this audit. The original JSON is retained as invalid-design evidence, not silently discarded.
- **RESEARCH-ONLY CORRECTION:** reserve ten seed slots per replicate: `base=100000+1000000*K+10*r`, ideal=`base`, fitting=`base+1..base+6`, held-out=`base+7`. Reserve/check all eight possible consumed seeds across every replicate and K. No controller, normals, sampling law, FAIR runtime or Q mapping changed. Also require strictly separated fresh ideal samples from finite GT, as the earlier validator already did. Seed separation is a pseudorandom stream design; it is not a proof of statistical independence of arbitrary PRNG sequences.
- Regression tests now cover the complete six-attempt reservation and instrument actual RandomState consumption during calibration. **TEST-DEVELOPMENT ERROR:** the first instrumentation replaced the RandomState class with a function, breaking `default_rng`'s internal `isinstance` check (1 failed, 35 passed). Replacing it with a tracing subclass fixed the instrumentation; subsequent run **36 passed in 5.40 s**. Expanded focused FAIR/Q regression run: **53 passed in 5.75 s**. These failures and passes concern local execution, not unverified CI.
- **CORRECTED PILOT:** K100, N16384, three full draws, NSGA-II generations40, seeds1/7. Authentic P0/Pf hashes are exactly identical to the original pilot for both runs. Historical Q reproduced unchanged. Seed1: historical `0.8477746903660618`, candidates `[0.8644571442067234, 0.8609353070266142, 0.8614431652146657]`, pool `0.8623986484322954`, range `0.003521837180109122`. Seed7: historical `0.9176454226436139`, candidates `[0.9364936121140962, 0.9349160528419531, 0.9350546256172211]`, pool `0.9355442645063815`, range `0.0015775592721430929`. All three rankings retain seed7 above seed1. Corrected fit/held-out medians respectively `(2.023359207649923,2.0464153858551604)`, `(1.910501990249257,1.9384042693575512)`, `(2.038753968671414,1.9961668805476893)`. The new seed schedule changes ideal and fit streams as well as blur, so the smaller range is **not an isolated causal estimate of the leakage bias**.
- **METHOD AUDIT:** stable index sorting conceals exact ties, so the analysis now records signed score gaps for every run pair, exact ties and strict reversals separately, including historical and pooled comparisons. Populations, normalized raw distances, individual and pooled ECDFs are persisted, not only their hashes. Complete-replicate variability uses fixed optimizer outputs; optimizer-seed variability is computed separately within algorithm/K/generation-budget for each fixed calibration. No optimizer output enters reference estimation.
- **AGGREGATION LIMITATION RETAINED:** the exploratory pool is a re-compression of the equal mixture of 200-point corrected ECDFs with arithmetic-mean epsilon. It is not a pool of original raw blurs, may introduce compression effects and is not a scientific release choice. A before/after snapshot of the existing audit only detects mutation; it is not an independent clinical recomputation under installed candidate records. Candidates are computed locally from distances and cannot change other FAIR families by construction. No production baseline is installed in this experiment.
- Local execution emitted Matplotlib/font-cache permissions warnings before setting `MPLCONFIGDIR=/tmp/issue81-mpl`; the pilot nevertheless completed. No numerical failure was inferred from those cache warnings. The expanded study uses the writable temporary cache. Release decisions and all other-MOP gates remain open.
- **ADDITIONAL HISTORICAL DEPENDENCE FOUND:** traced the earlier validator's fitting seeds at K100: 1100 consumes `[1100,1101]`; 1101 consumes `[1101,1102]`; 1102 consumes `[1102,1103,1104]`; 1103 consumes `[1103,1104]`. The latter two even produce exactly the same sigma `0.2619434753723717` and fit median `1.9691476755122945`. Thus the earlier four fits were **not independent complete fitting procedures**, in addition to the already documented common-noise/K dependencies. Distinct held-out final seeds in the second historical study do not repair this fitting dependence. Earlier numeric observations remain valid descriptive outputs, but claims of four independent sigma fits are withdrawn. The historical validator is retained to reproduce its declared seed lists; the corrected complete-calibration producer is the source of the new independent-block study.

### 2026-11-12 — Expanded complete-calibration / two-MOEA study executed locally (not certified)

**Evidence location:** [`calibration/experimental/issue81/2026-11-12`](../../calibration/experimental/issue81/2026-11-12/README.md). The [CSV](../../calibration/experimental/issue81/2026-11-12/real-q-by-run.csv) records **every individual historical, eight-calibration and pooled Q value**. Compressed full JSONs retain authentic normalized P0/Pf arrays, per-K resolution, all ECDFs and calibration parameters/seeds/hashes; the manifest retains original and compressed SHA256. The invalid-design pilot is preserved separately. No local `/tmp` path is the only surviving evidence.

**Design and execution:** K40/100/200 × NSGA-II(DEAP)/SPEA2(pymoo) × generation budgets40/150 × seeds1/7/19/43 = **48 authentic runs**, 16 per K. Each K has **eight complete reference draws**, N16384 ideal points, disjoint ten-slot RNG blocks including all six possible fitting attempts. Same 16 final populations per K are evaluated against all eight references and their exploratory pool. References are constructed before the optimizer panel and consume no optimizer results. The exact producer at execution is commit `ccb7445`; commands and environment versions accompany the evidence. Temporary pilot CI workflow removed after local evidence capture; **no new Actions result is claimed**.

**Metadata reporting defect identified after execution and corrected for future runs:** original expanded JSON's legacy top-level `optimizer=NSGA2` and `generations=300` reflect old single-condition defaults. Actual `algorithms`, `generation_budgets` and **every per-run optimizer/generations field** correctly record the design above. Original bytes are retained; current producer reports `MULTIPLE`/null for the corresponding legacy fields. Historical outputs are not rewritten to hide this error.

| K | Maximum Q range on one fixed run, across 8 calibrations | Signed candidate-minus-historical Q, all cells | Strict rank reversals across calibrations | Pairs with reversal vs historical |
|---:|---:|---:|---:|---:|
| 40 | 0.01882943083 | −0.00065445451 to +0.01817497632 | 0/120 | 0/120 |
| 100 | 0.01717405679 | −0.00077689100 to +0.01992354509 | 0/120 | 0/120 |
| 200 | 0.00691201151 | +0.02032171113 to +0.03129951010 | 1/120 | 3/120 |

Each denominator is the **120 pairs within one K's 16-run panel**, not independent Bernoulli trials. No exact candidate ties, Q=0 saturation or Q=1 saturation occurred. All within-algorithm/budget seed orderings were stable in this panel; the observed reversals involve different algorithm/budget conditions. Absolute Q ranges alone therefore do not establish decision stability, especially for nearly tied outputs.

**Specific K200 ordering evidence:**

- NSGA-II G40 seed19 minus SPEA2 G150 seed19: historical gap `+0.0000200888510281`; eight compressed-candidate gaps range `−0.000242991827106` to `+0.000505154672957` (two negative, six positive); pooled gap `+0.000520268018386`. This is a strict reversal across complete calibration draws, not a sorting tie.
- NSGA-II G150 seed7 minus SPEA2 G40 seed1: historical `+0.000707742060294`; all candidate gaps negative, range `−0.00181814049203` to `−0.00169206846635`; pool `−0.00173646678122`.
- NSGA-II G150 seed43 minus SPEA2 G40 seed1: historical `+0.000227167024686`; all candidate gaps negative, range `−0.00249360551205` to `−0.00233782420627`; pool `−0.00238353819007`.

**Calibration versus optimizer variability:** the candidate ranges above vary reference columns while holding each authentic run fixed. The following ranges vary the four optimizer seeds **inside each algorithm/K/budget at a fixed calibration**, then report the minimum and maximum of that seed-range over eight reference columns. Complete sample-SD values for both kinds of variation are in the analysis artifact. These summaries are descriptive; they are not variance components from a random-effects model or confidence intervals.

| K | NSGA-II G40 optimizer-seed range | NSGA-II G150 | SPEA2 G40 | SPEA2 G150 |
|---:|---:|---:|---:|---:|
| 40 | 0.179484–0.192589 | 0.045572–0.048914 | 0.063119–0.067729 | 0.057483–0.061668 |
| 100 | 0.077211–0.084000 | 0.171988–0.181908 | 0.076192–0.084581 | 0.056520–0.062351 |
| 200 | 0.107187–0.111052 | 0.076538–0.078977 | 0.069002–0.071940 | 0.014817–0.015736 |

Optimizer seed dispersion is larger than calibration dispersion in these grouped comparisons, but that does **not** make the close-pair inversion irrelevant. The same seeds are reused across budgets and K; NSGA-II budget prefixes may coincide. NSGA-II records G+1 histories including generation zero; SPEA2 records G histories. The study compares generation budgets, **not equal function-evaluation cost**, and makes no algorithm-superiority claim.

**Reference variability and one-factor sensitivity:**

| K | epsilon range | Fit median range / s_K | Independent final median range / s_K | Largest Q range varying only epsilon, fixed pooled ECDF | Largest Q range varying only ECDF, fixed mean epsilon |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.03048732701–0.03103627483 | 1.952987613–2.089088229 | 1.938585609–2.108959500 | 0.000230704 | 0.018659692 |
| 100 | 0.05169666842–0.05302621507 | 1.909121321–2.079969661 | 1.922580117–2.116681781 | 0.000530668 | 0.017451090 |
| 200 | 0.07500617512–0.07707909216 | 1.933011948–2.024288568 | 1.925421491–2.050238446 | 0.000800292 | 0.006939547 |

The one-factor comparisons are paired counterfactuals, not independent extra calibration samples or an additive variance decomposition. Under these conditions negative-ECDF variability has the larger effect. Every fitting draw met the original controller criterion; some final draws exceeded its band, which remains a fitting tolerance only.

**Aggregation and compression investigation:** pooled-minus-individual-mean Q ranged K40 `+0.000120641..+0.001288721`, K100 `+0.000288571..+0.001099017`, K200 `−0.000558226..+0.000759656`. The pooled result is not an arithmetic average of Q and can give a pair gap outside the individual gap interval. To isolate numerical representation, rebuilt the actual 3,367 corrected negative distances for each calibration, evaluated unchanged optimizer populations against those full empirical laws, and compared with the mandated 200-point representation. No reference was installed.

| K | Individual stored200 Q minus full3367 Q, all run×calibration cells | Pooled200 Q minus uncompressed mixture of stored quantiles | Pooled200 Q minus mixture of all raw corrected blur rows |
|---:|---:|---:|---:|
| 40 | +0.000123471 to +0.002549048 | +0.000107652 to +0.001152626 | +0.000279464 to +0.002989554 |
| 100 | −0.001434610 to +0.001858712 | +0.000249803 to +0.000947441 | −0.000209873 to +0.002238908 |
| 200 | −0.001041674 to +0.001013433 | −0.000530318 to +0.000751515 | −0.001061744 to +0.001119973 |

**The K200 across-calibration inversion disappears using full3367 reference laws:** NSGA-II G40 seed19 minus SPEA2 G150 seed19 is then always negative, `−0.000567990395744..−0.000168535584739`, and the mixture of all raw corrected draws gives gap `−0.000333047020858`. Thus the observed compressed-schema inversion is sensitive to the reference representation, not evidence of an optimizer-input mutation. No within-K pair reverses across calibrations under the full3367 diagnostic variant. This does **not** approve changing the 200-point schema or adopting raw pooling; those remain scientific/release decisions. Both compressed and raw variants retain the three K200 reversals relative to historical Q.

**Verification and preserved scope:** rebuilt **24/24 complete calibrations exactly**, reaudited **48/48 saved authentic P0/Pf pairs exactly** (historical Q and all five other FAIR physical/Q families), verified all stored candidate/pool Q values, population/ECDF hashes and **61 protected file hashes**. Independent SciPy Wasserstein computation agrees within maximum absolute Q difference **3.3306690738754696e−16**; its `1e−12` numerical check is **not a scientific acceptance tolerance**. Final focused tests: **53 passed in 5.47 s**. Source corrections and evidence serialization were checked with `git diff --check`. No FAIR runtime, MOP, official baseline, GT or canonical audit report changed.

**Evidence hashes (uncompressed bytes):** expanded original JSON `412e77e289b71e45e877e7aa5dd64a52a18a3a85da4534886cf4c48d6f38bdd1`; final analysis hashes and gzip hashes are in the committed manifest. Source hashes and environment are recorded in the JSON; after the schema-reporting fix, full-file reproduction hashes may change while numerical cells remain reproducible.

**Limitations and scientific decisions still requiring approval:** this is DPF5 M3/D2 at three K values, two optimizer implementations, two generation budgets, four seeds and eight complete calibrations/K on one fixed GT/normalization, not a certified population of runs. The 128 per-K run×calibration Q cells share rows/columns and are not 128 independent experiments. Finite min/max and sample SD are not uncertainty bounds. Approve separately (1) normative ideal/failure laws, (2) production sample counts/seeds and precision/ranking requirements tied to use, (3) whether the 200-point representation is sufficient for near ties and how aggregation should be defined, and (4) any separately versioned Closeness-only artifact after review. No tolerance, estimator, schema revision, normative law or scientific baseline is adopted here; issue #81 stays open and PR #86 stays Draft without merge. Other-MOP release gates remain outstanding.
