<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Analytical GT validation protocol — issue #119

**Status:** experimental, read-only diagnostic pilot. No GT replacement is
approved by this document. Original MOP evaluation equations are immutable
unless separately reviewed. Issues: [#119](https://github.com/monacofj/moeabench/issues/119)
and [#65](https://github.com/monacofj/moeabench/issues/65).

## Approved GT interface and budget direction (ADR 0057)

ADR 0057 now records the accepted architectural direction (not yet a
runtime implementation):

- `mop.ps(n_points)` and `mop.pf(n_points)` remain generators of
  mathematical Pareto-set/front samples, with the existing public roles.
  Do not add calibration side effects or a `mode="calibrate"` switch.
- `mop.calibrate()` retains the MOP-owned reference, external/search
  source, load/reload, persisted sidecar and `calibrated_gt` contracts.
  If no budget is explicitly requested **and new generation is needed**,
  a future geometry-/quality-/cost-aware policy will select it.
  Calibrate should not regenerate when an existing source is loaded.
- A visualization budget is independent of scientific GT resolution.
  Persisting the *actual GT used* in `exp.save()` is separate integration
  work subject to existing selective persistence contracts.
- No general implicit on-disk cache for every `pf()` call; no fixed
  precomputed catalog for arbitrary M/K/D.
- The existing 2,000 DTLZ / 10,000 DPF runtime calibration defaults
  remain **unchanged** until the numerical replacement and migration
  policy are independently verified and authorized.
- Do not change archived GTs, frozen numerical references or regression
  tests without prior maintainer warning/approval.

**Research next:** derive a defensible default budget rule from
intrinsic geometry, coverage-error convergence and bounded runtime;
test MOP-specific deterministic samplers and distinguish exact
geometric checks from finite dense-reference diagnostics. Report
actual returned cardinality, duplicates, extremes and relative
errors as `M`/intrinsic dimension and budgets vary. Changes to
`calibrate()` occur only after approval of the concrete algorithm.

## Distinct validity questions

1. **Pareto membership and completeness.** Establish the true feasible
   nondominated locus from mathematics or independent references. A set
   may have no internal dominance and nevertheless omit entire Pareto
   components (DTLZ5/6 M>=4 is a known example). Norm or objective-sum
   identities demonstrate *surface membership*, not Pareto completeness.
2. **Coverage in objective space.** Measure distances from a separate,
   high-resolution geometrical reference to the candidate finite GT.
   Report mean, 95th percentile, maximum; inspect missing components,
   boundary extremes and disconnected regions independently.
3. **Spatial regularity.** Report nearest-neighbor distance distribution
   and coefficient of variation (CV), along with duplicate fraction.
   Low CV does **not** imply good global coverage (a dense cluster can
   have very regular spacing); always interpret with covering radius.
4. **Sampling law and representativeness.** Define the objective-space
   geometrical measure, not just a uniform distribution over decision
   parameters. A probability-uniform *random* sample is not necessarily
   regular; regularity requires a separate finite-sample check. For
   mixed intrinsic dimensions, report each component separately and
   seek explicit scientific authorization for a cross-component weight.
5. **Reproducibility and convergence.** Verify same seed/budget yields the
   same ordered points; compare independent references/seeds and fixed
   budgets; assess indicator sensitivity with *unchanged* MOEA outputs.

## Pilot (DTLZ1–4)

Run from project root after installing the project and SciPy:

```bash
python scripts/audit/gt_geometry_pilot.py --budget 256 --budget 1024 --reference-size 4096 --objectives 3 5
python -m pytest tests/unit/test_gt_geometry_pilot_issue119.py -q
```

The pilot deliberately draws independent objective-space references:

- **DTLZ1:** normalize independent exponential variables, then scale by
  0.5. This is uniform in simplex `sum(f_i)=0.5` with respect to its
  intrinsic area measure, in any `M>=2`.
- **DTLZ2/3/4:** normalize absolute standard Gaussian vectors. This is
  uniform in surface area of the positive unit hypersphere.

Reference samples use a *different* RNG (`default_rng`) and seed from
MOP `ps()` (historically `RandomState(42)`). Their finite size means
measured distances are estimators, **not** exact continuous-surface
Hausdorff bounds. Different reference seeds should be compared. For
DTLZ1, objective-space surface-uniform sampling is **not** equivalent
to the current uniform decision-parameter sampling. This is a
hypothesis to measure, not permission to rewrite the canonical sampler.

The output is deterministic JSON with explicit sample budgets,
independent reference size and seed, objective-space bidirectional
nearest distances, finite-sample neighbor spacing, geometric residuals,
uniqueness and extrema coverage. No diagnostic threshold is
misrepresented as a mathematical theorem or CI acceptance gate.

## Controlled farthest-point candidate experiment

A second **read-only** diagnostic compares the current actual MOP
`pf(n_points)` with an independent objective-space greedy
farthest-point selection (FPS) candidate. It uses the same finite-budget
quality diagnostics and a third, independently seeded diagnostic
reference; no candidate is chosen from that diagnostic reference.

```bash
python scripts/audit/gt_fps_comparison.py --objectives 3 5 --budget 256 1024
python -m pytest tests/unit/test_gt_fps_comparison_issue119.py -q
```

FPS initializes at all analytical objective-axis extrema, then
selects the candidate with the largest current nearest-sample
distance. Its `--candidate-pool`, `--reference-size` and seed
are explicit and its output is reproducible. The candidate is
constructed in objective space *only*: this study does not propose a
replacement for MOP `ps()`, which must return decision vectors
mapping to the chosen front. Constructing valid decision-space
preimages remains a separate correctness obligation.

Importantly, the classical 2-approximation guarantee for finite
metric k-center applies relative to **the supplied finite candidate
pool**, with the algorithm's seed-center constraints accounted for;
it is **not** a certified bound on covering an entire continuous
Pareto front. Runtime measurements are exploratory and system
dependent. The independent finite reference's worst nearest distance
is only a lower bound on the continuous covering radius; no pass/fail
claim may be derived from it alone. The experiment must be repeated
with candidate pool/reference sizes and dimensions varied before
selecting a production generator.

## Analytical PS preimages and repeated evaluation

The first FPS candidate study only selects **objective-space** positions,
whereas ADR 0006 requires a native `ps()` in **decision space**. The
read-only follow-up verifies exact analytical preimages for
DTLZ1–4 (including positive simplex/hypersphere axes):

- DTLZ1: invert the cumulative objective-product mapping; on a
  degenerate boundary where the denominator vanishes choose zero for
  the non-identifiable position parameter. Set all distance variables
  to 0.5.
- DTLZ2/3: invert positive hypersphere directions to the angular
  coordinates of the standard DTLZ mapping. Set distance variables
  to 0.5. DTLZ3 shares its analytical front geometry with DTLZ2.
- DTLZ4: the same angular inverse with the additional 100th-root
  transformation. Boundary cases must be evaluated, not only
  checked by their norm.
- Verify each candidate by the **unmodified actual MOP evaluation**,
  requiring the computed objectives to match the proposed samples,
  while also checking decision-variable bounds and different M/K.

Run the multi-seed comparison:

```bash
python scripts/audit/gt_fps_preimage_study.py --objectives 3 5 \
    --budgets 128 512 --seeds 101 202 303 \
    --candidate-pool 4096 --reference-size 8192
python -m pytest tests/unit/test_gt_fps_preimage_study_issue119.py -q
```

The summary reports paired seeds for incumbent `mop.pf()` and the
validated FPS preimages, observed geometry, per-seed reference distances,
nearest-neighbor dispersion and elapsed generation times. Candidate
pool/reference seeds are independent. Reference distances are not
certified continuous-front worst-case errors. A fixed candidate pool
creates a ceiling on resolution; repeat with larger pool sizes and
objective dimensions before drawing conclusions. No FAIR metrics,
calibrated GT or historical regression/baseline data are touched.


## Repeated real-MOP measurements — 2026-11-19 (research only)

The repository-native CI experiment in
[PR #127](https://github.com/monacofj/moeabench/pull/127)
([Actions run 44583456192](https://github.com/monacofj/moeabench/actions/runs/44583456192))
executed `gt_fps_preimage_study.py` using the **actual MOP implementations**
with Python 3.12, DTLZ1–4, M=3/5, n=128/512/1024, candidate and reference
seeds 101/202/303, independent reference size 8192, and FPS candidate
pools 4096 and 8192. Each pool ran 72 comparisons (144 total); raw JSON
was uploaded as workflow artifacts with 30-day retention.

Representative **median of three paired seeds** for M=5, n=1024, pool=8192
(lower means better in all listed indicators):

| Geometry | Metric | Incumbent | FPS candidate |
|---|---|---:|---:|
| DTLZ1 simplex | Reference-to-front mean | 0.04533 | 0.02990 |
| DTLZ1 simplex | Finite-reference maximum | 0.17064 | 0.05463 |
| DTLZ1 simplex | Nearest-neighbor CV | 0.753 | 0.084 |
| DTLZ2/3/4 sphere | Reference-to-front mean | 0.10782 | 0.10055 |
| DTLZ2/3/4 sphere | Finite-reference maximum | 0.24306 | 0.18427 |
| DTLZ2/3/4 sphere | Nearest-neighbor CV | 0.306 | 0.069 |

The FPS candidate gave smaller *reference-to-front mean* distances
in all 72/72 paired pool=8192 comparisons and 72/72 paired pool=4096
comparisons. DTLZ2,3,4 share the same Pareto-objective geometry; their
identical reference statistics are **not independent physical geometries**.
Maximum observed objective-space preimage replay error: 1.07e-14
(pool=8192). Median wall times across all settings: incumbent `pf()`
0.0003 s, FPS including decision-space preimage and evaluation 0.077 s
(pool=8192); for pool=4096 the median was 0.034 s. These are approximate
runner-dependent diagnostics, not production performance guarantees.

**Interpretation and open questions:** the finite reference estimates
average and worst *observed* gaps; it cannot certify a continuous-front
worst-case covering radius. FPS dramatically reduces nearest-neighbor
variation, but its higher CPU cost and candidate-pool resolution ceiling
must be characterized before choosing a production sampler. Results
do not authorize changing MOP `ps()`/`pf()`, ADR 0018's runtime
budgets, `mop.calibrate()`, frozen regression tests or numeric baselines.
Future tests should include M>5 scaling, candidate-pool convergence,
nonparametric cost-quality tradeoff, independent geometric bounds and
indicator sensitivity on fixed MOEA output.


## Higher-dimensional stress test — 2026-11-19 (research only)

The independently rerunnable workflow
[PR #128](https://github.com/monacofj/moeabench/pull/128),
[Actions run 44585779672](https://github.com/monacofj/moeabench/actions/runs/44585779672),
tested the real, unmodified MOP `pf()` generator against FPS plus verified
decision-space preimages for **M=10/15**, budgets **128/512**, DTLZ1,
DTLZ2 and DTLZ4, three paired seeds (101/202/303), candidate pools
**2048 and 8192**, and an independent 16384-point finite reference.
Each pool produced 36 measured comparisons (72 in total). Raw JSON
artifacts were retained by the Actions workflow for 30 days.

Representative **medians of three seed pairs**, pool=8192 and
GT budget=512; smaller values are better:

| Problem | M | Reference→GT mean: current/FPS | Max finite-reference gap: current/FPS | NN-CV: current/FPS |
|---|---:|---:|---:|---:|
| DTLZ1 | 10 | 0.14987 / **0.07403** | 0.38256 / **0.11972** | 0.730 / **0.147** |
| DTLZ1 | 15 | 0.16409 / **0.08479** | 0.33747 / **0.13587** | 0.792 / **0.249** |
| DTLZ2 | 10 | **0.31697** / 0.32461 | 0.54728 / **0.46619** | 0.203 / **0.050** |
| DTLZ2 | 15 | **0.41751** / 0.43651 | 0.60722 / **0.59242** | 0.181 / **0.049** |
| DTLZ4 | 10 | **0.31697** / 0.32461 | 0.54728 / **0.46619** | 0.203 / **0.050** |
| DTLZ4 | 15 | **0.41751** / 0.43651 | 0.60722 / **0.59242** | 0.181 / **0.049** |

For both pool sizes, DTLZ1 improved mean distance in all paired seed
comparisons, whereas the positive hypersphere of DTLZ2/4 worsened its
mean distance in nearly all comparisons at M=10/15 (pool=8192: zero
improvements in the twelve spherical M×budget×seed configurations for
each of DTLZ2 and DTLZ4). FPS nevertheless reduced the *observed
maximum* and nearest-neighbor CV in these representative cases.
The geometry labels DTLZ2 and DTLZ4 are not independent front shapes;
DTLZ4 tests the additional objective-to-decision inverse.

Maximum forward-replay mismatch on the actual MOPs was
7.87e-15. Overall median exploratory generation times: 0.00041 sec
for the incumbent versus 0.04592 sec for FPS+inverse+evaluation
(pool=8192); with pool=2048, 0.00065 vs 0.02171 sec. Shared GitHub
runner timings are not a rigorous runtime benchmark.

**Interpretation.** The strong DTLZ1 gain may partially reflect the
candidate's mandatory inclusion of exact axes, unlike the incumbent's
random simplex sampling; a separate *axis-preserving incumbent*
control is needed before attributing every gain to FPS itself.
A larger FPS candidate pool does not necessarily decrease *mean*
reference-to-GT distance: FPS greedily targets maximum gaps, not
mean-square/mean-distance quantization. In high intrinsic dimensions,
a finite uniformly random reference also becomes sparse, so observed
maximum distances **are not** certified covering radii. These findings
rule out treating FPS as a universally superior default, and support
geometry-specific selection and evaluation of Riesz/CVT or hybrid
alternatives. No production change is authorized by this study.


## Anchored-incumbent control and approximate Voronoi relaxation (2026-11-19)

Research PR [#129](https://github.com/monacofj/moeabench/pull/129)
([Actions run 44587298913](https://github.com/monacofj/moeabench/actions/runs/44587298913))
compared five equally-sized objective-space GT sets generated or
inverse-validated against **unchanged actual DTLZ1/2 MOP evaluation**:
(i) existing `mop.pf()` sample, (ii) same with exact analytical
coordinate-axis extrema replacing the first M points, (iii) FPS from
a finite surface-uniform candidate pool seeded at the same axes,
(iv) projected Monte Carlo centroidal Voronoi relaxation initialized
from the anchored incumbent, and (v) FPS followed by the same
relaxation. Other controls: 12 fixed Lloyd-like iterations, independent
candidate and evaluation reference seeds, M=3/5/10, n=128/512,
seeds 101/202/303, candidate pool 2048/8192 and independent
8192-point evaluation reference. Each pool ran 36 MOP×M×budget×seed
comparisons, 72 total. Both new experimental jobs passed.

The approximate Voronoi operation assigns independent candidate
points to their closest current center, updates each center to
the assigned-cell arithmetic mean, then projects that mean to
the known simplex or positive hypersphere. Exact objective axes
remain anchored. These are **finite-sample, projected Lloyd
updates**, not a continuous-manifold CVT optimality theorem.

**Representative medians across three seeds**, M=5, n=512,
pool=8192 (smaller is better):

| Method | DTLZ1 ref→GT mean | DTLZ1 max observed | DTLZ2 ref→GT mean | DTLZ2 max observed |
|---|---:|---:|---:|---:|
| Incumbent | 0.05608 | 0.17816 | 0.12838 | 0.28459 |
| Incumbent with axes | 0.05592 | 0.16456 | 0.12838 | 0.28459 |
| FPS | 0.03627 | **0.06765** | 0.12110 | 0.22229 |
| Axes + projected MC-CVT | 0.04283 | 0.10841 | 0.11622 | 0.23066 |
| FPS + projected MC-CVT | **0.03401** | 0.06821 | **0.11581** | **0.21928** |

**M=10, n=512**, pool=8192: DTLZ1 incumbent 0.14976,
axes-only 0.14947, FPS 0.07400, axes+MC-CVT 0.07532,
FPS+MC-CVT **0.06358** for reference→GT mean. DTLZ2
incumbent **0.31720** versus pure FPS **0.32431**
(mean regresses), whereas axes+MC-CVT 0.29492 and
FPS+MC-CVT 0.29495. The latter is still accompanied by
a larger observed maximum gap than pure FPS, illustrating the
worst-gap / mean quantization tradeoff.

**Inference:** forcing exact axes in an otherwise unchanged DTLZ1
random sampler gives little improvement in mean distance; FPS
benefits arise primarily from distribution over the simplex.
Projected Voronoi refinement can improve mean quantization while
sometimes reducing worst-gap performance or spacing evenness.
High-dimensional DTLZ1 unbalanced/empty Voronoi cells and local
minima remain possible; the CV and exact point uniqueness must
be reviewed, especially in other component geometries.

**Cost caveat:** measured `fps` time is separate from
`fps_mc_cvt` *relaxation-only* time. End-to-end hybrid cost
is their sum **plus** decision-space inversion and evaluation
(which are performed subsequently and not included in these
wall-time fields). The reported `anchored=0` timing is merely
a placeholder for the near-zero axis replacement operation and
**excludes the incumbent generation cost**. Thus the current
timings are exploratory operation measurements, not apples-to-apples
complete runtime guarantees.

The independent evaluation sample is finite; none of these distances
certify the full continuous-front covering radius. Monte Carlo
candidate and evaluation measures are independent but based on
the same known analytical front geometry. The comparison is
**not** an acceptance test, and changes no accepted GT sampler,
calibration budget, historical regression test, baseline or ADR.
Next: optimize/review production-suitable variants separately,
measure complete end-to-end costs and seek mathematical coverage
bounds or robust reference-size convergence before selecting a
calibration default.


## Complete GT generation cost and reference-size sensitivity — 2026-11-19

Research-only PR [#130](https://github.com/monacofj/moeabench/pull/130),
[Actions run 44592287682](https://github.com/monacofj/moeabench/actions/runs/44592287682),
ran the actual DTLZ1/2 MOP implementation with M=3,5,10,
GT budgets n=128/512, candidate pool=4096, and paired seeds
101/202/303: 36 configuration/seed cases. For each case the **same
five generated GT arrays** were compared against independent
objective-space uniform validation references of sizes
2048, 8192 and 32768. Raw per-seed JSON data are uploaded to
the Action run as a 30-day research artifact.

For all alternative generation methods, wall times were measured
around full candidate generation (including objective-space pool),
inversion to decision-space Pareto-set points, and forward evaluation
with the unchanged original MOP function. Diagnostic reference
generation, cKDTree queries and output summarization were **not**
included in generation times. Unlike PR #129, the hybrid method's
timer includes *both* FPS and subsequent Voronoi relaxation.

**M=5, n=512, pool=4096; medians over three candidate/reference seeds**
(mean reference-to-GT distance uses an 8192-point reference):

| Method | DTLZ1 mean | DTLZ1 complete s | DTLZ2 mean | DTLZ2 complete s |
|---|---:|---:|---:|---:|
| Incumbent | 0.05608 | 0.00038 | 0.12838 | 0.00059 |
| Exact axes + incumbent | 0.05592 | 0.00071 | 0.12838 | 0.00093 |
| FPS | 0.03617 | 0.04218 | 0.12148 | 0.04277 |
| Anchored incumbent + MC-CVT | 0.04417 | 0.03366 | **0.11794** | **0.03292** |
| FPS + MC-CVT | **0.03460** | 0.07249 | 0.11811 | 0.07551 |

For the *same generated GTs*, increasing the independent
validation-reference size typically changed **mean** reference-to-GT
distance by a small amount (e.g. DTLZ2 M=10 n=512 incumbent:
0.31762 at R=2048, 0.31720 at R=8192 and 0.31713 at R=32768).
Finite-reference *observed maxima*, in contrast, often increased:
the same DTLZ2 example had maxima 0.50377, 0.52220, 0.54728,
respectively. An increased maximum is consistent with discovering
previously unobserved gaps, not deterioration of the fixed GT.
The diagnostic reference-size convergence is empirical; none of
these maxima is a certified upper bound on continuous-front
covering radius.

**Interpretation.** Improved geometrical quality comes with a
one-to-three-order-of-magnitude higher per-call generation cost
on this runner. For typical studied budgets even the hybrid cost
was below 0.2 seconds, but this observation is **not** a general
runtime limit for larger n, M, candidate pools or MOPs. The
choice of scientific-calibration budget cannot safely be inferred
from finite sampled reference distance alone: need a
geometrically justified absolute/relative resolution criterion,
a conservative time/memory bound and tests of sensitivity
of GT-dependent indicators on fixed MOEA outputs.

The experimental methods do not change any production MOP,
GT sidecar, ADR 0018 runtime point defaults, historical
regression or certification baseline. Their measured speed
is subject to GitHub Actions runner, cold-cache and numerical
library variability; there is no independent performance
certification claim.


## Sensitivity of actual indicators to GT discretization on fixed MOEA outputs

Research-only [PR #131](https://github.com/monacofj/moeabench/pull/131),
[Actions run 44595886190](https://github.com/monacofj/moeabench/actions/runs/44595886190)
runs **actual NSGA-II** once per MOP/seed on DTLZ1 and DTLZ2, M=3,
seeds 101 and 202, 80 individuals, 350 generations. Each final
objective-space population is fingerprinted with SHA256 and reused
**without regeneration or modification** for the three GT budgets
n=128/512/1024 and three generator methods: current `mop.pf()`,
objective-space FPS with verified PS inverse, and FPS followed by
projected MC-CVT. All GTs were generated in memory without
calling `calibrate()` or writing any sidecar.

This study calls MoeaBench's existing `GEN_igd`, `GEN_igdplus`
and `GEN_hypervolume` classes (no metric implementation changes).
For HV, it holds the analytical normalization bounds **fixed**
(ideal=0, nadir=0.5 for DTLZ1 or 1 for DTLZ2; reference=1.1 in
normalized space) and reports:
- `H_sol`: HV of the fixed MOEA front; **must not depend on GT**;
- `H_GT`: HV of the candidate discretized reference;
- `H_abs = H_sol / H_GT`: the same *ratio rule* as the public absolute
  HV mode, tested here in fixed analytical bounds, not by mutating
  MOP calibration. A preliminary 35-generation experiment produced
  `H_sol=0` on DTLZ1; the non-informative run was rejected and the
  instrument now requires positive solution HV. Do not use it as a
  scientific result.

For each fixed MOEA seed, differences between n=128 or 512 and the
**same method's n=1024** are recorded. Those are *finite-budget
comparisons*, not errors against continuous mathematical truth.
Selected medians of two fixed-MOEA seeds at n=512:

| MOP | GT generator | absolute change in IGD vs 1024 | absolute change in H_abs vs 1024 |
|---|---|---:|---:|
| DTLZ1 | incumbent | 0.00045 | 0.01092 |
| DTLZ1 | FPS | 0.00087 | 0.00417 |
| DTLZ1 | FPS + MC-CVT | 0.00065 | 0.00772 |
| DTLZ2 | incumbent | 0.00077 | 0.01497 |
| DTLZ2 | FPS | 0.00080 | 0.00872 |
| DTLZ2 | FPS + MC-CVT | 0.00061 | 0.01221 |

At n=128, absolute H_abs changes versus 1024 were substantially
larger (DTLZ1 incumbent 0.05733 vs FPS 0.02096; DTLZ2 incumbent
0.06678 vs FPS 0.04115). The paired sample is only two MOEA seeds;
these are descriptive results, not release acceptance thresholds or
statistical performance certification.

**Scientific caution:** IGD and IGD+ average distances over the
reference points, so changing the spatial distribution of GT points
also changes their *implicit reference measure*. MoeaBench's IGD
metric class currently uses `zero_to_one=True`, whose normalization
bounds can themselves depend on the finite reference. Therefore
neither absolute IGD change nor convergence to a higher-n sampled
GT alone proves accuracy to a continuous geometric target.
A defensible target must specify the intended surface measure,
normalization convention and independently validated reference.
Furthermore, improvements in geometrical spacing need not produce
smaller IGD for a particular fixed MOEA approximation: the
estimand is affected by reference weighting.

The dependence of `H_abs` on `H_GT` is direct and important:
a fixed `H_sol` and fixed normalization still give a changing ratio
as discretized GT hypervolume changes. The GT's full continuous
dominated volume and the numerical approximation error remain to
be bounded or characterized independently. HV in `scale='raw'`
with unchanged normalization bounds does not change when GT changes;
only the calibrated denominator in absolute mode does.

No production indicator, historical regression test, stored
MOP GT, accepted ADR, or calibration default was changed. Next:
geometry-derived reference measures and bounds (continuous
simplex/hypersphere where feasible), sensitivity across more
MOPs/seeds and explicit numerical error tolerance approval
before selecting automatic GT budgets.


## Exact continuous M=2 oracle: simplex segment and spherical arc (2026-11-19)

Research [PR #132](https://github.com/monacofj/moeabench/pull/132),
[Actions run 44599007083](https://github.com/monacofj/moeabench/actions/runs/44599007083),
introduces an **exact independent continuous-geometry reference** for
DTLZ1 and DTLZ2 at M=2 (and the shared DTLZ3/DTLZ4 positive-circle
geometry). With exactly two objectives:

- DTLZ1: `f1+f2=0.5` over a line segment of length
  `L = sqrt(2)/2`. Sort points along the line, integrate
  Euclidean nearest-point distance over each 1-D Voronoi cell
  exactly, with uniform arclength; the maximum distance occurs
  at an endpoint or a gap midpoint.
- DTLZ2/3/4: `f1²+f2²=1` on the positive quarter circle with
  angle 0..π/2. Voronoi cells are angular intervals. The
  chord distance `2 sin(|theta-theta_i|/2)` admits a closed-form
  cell integral. The maximum again occurs at one of the endpoints
  or at the midpoint between neighboring angular samples.
- The exact average distance and exact global *continuous*
  covering radius are computed from the actual sampled points
  with no dependence on the size or random realization of an
  evaluation reference.

The research script audits the actual `MOP.pf()` against FPS and
FPS+projected MC-CVT, using n=64/256/1024, independent seeds
101/202/303 and candidate pool 4096. The results below are
**medians over three seeds**, not certified general method rankings;
the continuous-geometry metrics themselves are exact up to
floating-point error.

| Geometry | n | Method | Exact continuous mean nearest distance | Exact continuous covering radius |
|---|---:|---|---:|---:|
| DTLZ1 line | 64 | incumbent | 0.0047588 | 0.0212769 |
| DTLZ1 line | 64 | FPS | 0.0028480 | 0.0109974 |
| DTLZ1 line | 64 | FPS+MC-CVT | **0.0028263** | **0.0068117** |
| DTLZ1 line | 256 | incumbent | 0.0012903 | 0.0092723 |
| DTLZ1 line | 256 | FPS | **0.0006975** | 0.0027336 |
| DTLZ1 line | 256 | FPS+MC-CVT | 0.0007074 | **0.0022324** |
| DTLZ2 arc | 64 | incumbent | **0.0062333** | **0.0124666** |
| DTLZ2 arc | 64 | FPS | 0.0063291 | 0.0246054 |
| DTLZ2 arc | 64 | FPS+MC-CVT | 0.0062815 | 0.0156567 |
| DTLZ2 arc | 256 | incumbent | **0.0015400** | **0.0030800** |
| DTLZ2 arc | 256 | FPS | 0.0015478 | 0.0057831 |
| DTLZ2 arc | 256 | FPS+MC-CVT | 0.0015718 | 0.0044910 |

**A crucial distinction:** the existing DTLZ2 M=2 generator
already uses deterministic equal-angle (uniform-arclength) spacing,
including the exact endpoints. On a 1D circular arc, that is
an excellent geometric design and FPS should **not** replace it.
The DTLZ1 M=2 generator instead samples uniform random decision
parameters, which yields uneven arc spacing despite a uniform
distribution in expectation; FPS can improve its finite-size
spacing. A future deterministic equal-arclength line grid is
also a natural independent candidate, to be compared before
selecting any replacement.

For these *specific* continuous curves, the finite-reference
observed maximum is mathematically a **lower bound** on the exact
covering radius. The study also measured how closely finite
random reference means approach the exact surface mean.
No analogous exact bound is claimed for arbitrary M>=3,
disconnected, constrained or mixed-dimensional Pareto fronts.
These analytical errors do not by themselves select a numeric
calibration tolerance; that choice is a scientific policy
requiring a separate explicit decision.

No production GT generation, accepted ADR, calibration default,
regression test, saved GT, or baseline was altered.


## Analytical optimal M=2 grids versus FPS (2026-11-19)

Research [PR #133](https://github.com/monacofj/moeabench/pull/133),
[Actions run 44600707914](https://github.com/monacofj/moeabench/actions/runs/44600707914)
compares the actual, unchanged `mop.pf()` for DTLZ1/2 M=2
against objective-space FPS and two deterministic analytical
arclength grids. The old exact continuous nearest-distance oracle
from PR #132 evaluates all methods; independent research tests
check the closed-form expressions and the numerical inversion
to genuine MOP decision vectors. Conditions: n=64/256/1024,
three FPS seeds, pool=4096. The same mathematical front is
sampled and no calibration sidecar is touched.

**Analytical optimization on a 1-D curve.** Restrict to
Euclidean nearest-point distances averaged uniformly in arclength,
and the maximum Euclidean nearest-point distance over the full
continuous segment or positive quarter-circle. With n sampled
points **constrained to contain both objective-axis endpoints**,
equal arclength spacing minimizes both criteria: the gaps are
equal because the sum of integrated nearest-distance
contributions is convex in gap length, and the maximum
distance is governed by the largest gap. With endpoint inclusion
relaxed, a midpoint-centered partition into n equal-arclength
cells gives the corresponding unconstrained optimum. This
latter design omits exact objective-axis extremes, potentially
undermining ideal/nadir provenance and other GT roles.
It is reported solely as a separate mathematical control.

For the DTLZ1 segment of length L=sqrt(2)/2, the exact
continuous mean and covering radius are L/(4k) and L/(2k),
where k=n-1 if both endpoints are prescribed, or k=n for
the equally spaced midpoint design. For the quarter-circle,
with total arclength A=pi/2 and cell/gap arclength g=A/k,
the exact mean chord distance is
`8 * (1 - cos(g/4)) / g` and exact radius is
`2 * sin(g/4)`. The formulas are **M=2 only**.

Representative results below use the exact continuous oracle
(median over the three FPS seeds, equal-grid values are fixed):

| MOP | n | Method | Exact mean distance | Exact covering radius | Median generation cost (s) |
|---|---:|---|---:|---:|---:|
| DTLZ1 | 256 | Current `pf()` | 0.00129034 | 0.00927234 | 0.000180 |
| DTLZ1 | 256 | FPS | 0.00069747 | 0.00273359 | 0.019000 |
| DTLZ1 | 256 | Equal arclength, **axes included** | **0.00069324** | **0.00138648** | 0.000220 |
| DTLZ1 | 256 | Equal arclength, midpoints | 0.00069053 | 0.00138107 | 0.000218 |
| DTLZ2 | 256 | Current `pf()` | **0.00154000** | **0.00307999** | 0.000052 |
| DTLZ2 | 256 | FPS | 0.00154785 | 0.00578306 | 0.017326 |
| DTLZ2 | 256 | Equal arclength, **axes included** | **0.00154000** | **0.00307999** | 0.000204 |
| DTLZ2 | 256 | Equal arclength, midpoints | 0.00153398 | 0.00306796 | 0.000184 |

For DTLZ1 n=1024, exact-axis equal spacing gave
mean 0.000172802 and radius 0.000345604 in ~0.000394 s,
whereas FPS gave mean 0.000182508 and radius 0.000836382
in ~0.074403 s. These execution times include candidate
generation, decision-space inversion and forward objective
evaluation where applicable, but are runner-dependent.

**Scientific implication:** For one-dimensional fronts whose
arclength parametrization is known, a small deterministic
analytical method is preferable to a costly finite-candidate
FPS or projected CVT algorithm. The existing DTLZ2 M=2
generator already implements the axes-preserving solution;
DTLZ1 M=2 could eventually receive the analogous simple
generator **only after explicit production-change review**.
The midpoint design makes a modest further improvement to
mean/radius but omits both axes and is not recommended as
the default scientific GT where exact objective extremes
must be retained.

This exact-grid result says nothing about global optimization
of discretization for M>=3 or disconnected/constrained fronts.
Uniform surface measure, objective-space normalization and
how GT discretization affects scientific indicators remain
separate obligations; no default budget or numerical tolerance
is established by these M2 results. No production generator,
existing regression tests, GT sidecars, baseline or accepted
ADR was changed.


## Analytically bounded continuous covering for M>=3 (2026-11-19)

Research [PR #134](https://github.com/monacofj/moeabench/pull/134),
[Actions run 44603135864](https://github.com/monacofj/moeabench/actions/runs/44603135864)
establishes a **provable, potentially conservative interval** for the
maximum Euclidean distance from the *continuous* DTLZ1 simplex or
DTLZ2/3/4 positive hypersphere to any fixed valid finite GT.
The method is independent of reference-front random sampling and
does not modify production samplers or calibration.

Construct all integer compositions of q into M coordinates, divide by q,
and map to the unit simplex. This grid has
`binomial(q+M-1,M-1)` points. Nearest integer rounding with largest
fractional remainder yields a covering net whose Euclidean radius
is at most `sqrt(M)/q` on the *unit* simplex. On the
DTLZ1 half-scale simplex, the bound becomes
`epsilon = sqrt(M)/(2q)`. For the positive unit hypersphere,
normalize nonnegative simplex vectors to unit norm.
Because simplex-vector norms are >= `1/sqrt(M)` and
normalization satisfies
`||u/||u|| - v/||v|||| <= ||u-v|| / sqrt(||u|| ||v||)`,
the spherical net radius is bounded by
`epsilon = M/q`. These are deliberately conservative
real-arithmetic bounds.

For any fixed sampled GT S, its objective-space nearest-set distance
`d(x,S)` is 1-Lipschitz on the continuous manifold. Therefore:
`D_q = max_{v in net} d(v,S) <= R(S) <= D_q + epsilon`,
where `R(S)` is the *true continuous covering radius*. The code
reports both ends and a small 1e-12 numerical roundoff allowance.
This is a mathematical inequality for ideal real arithmetic;
the implementation does **not** use interval arithmetic or
provide a formal machine-checked floating-point enclosure.
The net is **not** surface-uniform on the sphere, and the method
does **not** certify mean distance or IGD.

The isolated CI study used the **actual unchanged**
DTLZ1/2 MOP implementations (the DTLZ3/4 front geometries are
covered by the same mathematical spherical model),
M=3/5, GT budget=128/512, q=8/16/24, and independent FPS
candidate pool 4096 with seed 101. Selected bounds for
q=24 (lower/upper, continuous worst-distance units):

| MOP | M | GT points | Current PF interval | FPS interval | Separation certified? |
|---|---:|---:|---:|---:|---|
| DTLZ1 | 3 | 128 | [0.119892, 0.155976] | [0.040923, 0.077007] | **Yes, FPS smaller** |
| DTLZ1 | 3 | 512 | [0.057432, 0.093517] | [0.021293, 0.057377] | Yes, very narrow margin |
| DTLZ1 | 5 | 128 | [0.290215, 0.336800] | [0.109357, 0.155942] | **Yes, FPS smaller** |
| DTLZ1 | 5 | 512 | [0.231298, 0.277883] | [0.078527, 0.125112] | **Yes, FPS smaller** |
| DTLZ2 | 3 | 512 | [0.100794, 0.225794] | [0.044448, 0.169448] | No, intervals overlap |
| DTLZ2 | 5 | 512 | [0.310623, 0.518956] | [0.245180, 0.453513] | No, intervals overlap |

The DTLZ1 superiority at these tested configurations follows
from nonoverlapping *mathematical* intervals, not inferred from
Monte Carlo observations. It says nothing about other
budgets/seeds and does not prove FPS globally optimal. The
DTLZ2 empirical coverage improvements remain unproven as an
ordering of *continuous* worst-case coverage because the
conservative spherical bound is too loose.

A crucial cost limitation is combinatorial growth of net size,
particularly at high objective counts: 20,475 net points for
M=5 and q=24, and larger counts require more work. The
`max_points` guard stops unacceptable enumeration. A tighter
geometry-specific net or optimization with certified bounds
will be needed to scale these certificates to M>5 and to
establish quality-driven automatic calibration budgets.

Independent new research tests validate composition counts,
analytical geometry, randomized net distances, the rigorous
enclosure of previously derived *exact M=2* continuous
covering radii, and original-MOP FPS decision-space inversion.
No historical regression tests, calibrated GTs, current ADR 0018
runtime budgets or production code were modified.


## Tighter continuous covering certificate via dependent rounding (2026-11-19)

Research [PR #135](https://github.com/monacofj/moeabench/pull/135)
and [Actions run 44605082042](https://github.com/monacofj/moeabench/actions/runs/44605082042)
refine the original complete barycentric lattice proof from PR #134.
They make **no change** to net points, original MOPs, the accepted GT
budget defaults, original tests or the reference sets.

For any nonnegative simplex vector `u` of coordinate sum 1,
write `q*u = floor(q*u)+r` with fractional coordinates
`0 <= r_i < 1`. Their sum is an integer `k`.
Systematic *dependent rounding* supplies a `k`-hot binary
random vector `B` such that `sum(B)=k` and `E(B_i)=r_i`:
take an offset T uniformly over [0,1), apply the difference of
successive floors of the cumulative fractional sums plus T.
Consequently,

`E[||B-r||²] = sum_i r_i(1-r_i) <= M/4`.

Therefore at least one q-level integer simplex lattice point
lies within `sqrt(M)/(2q)` of every unit simplex point.
This improves the previous `sqrt(M)/q` bound **by 2×**.
For DTLZ1 half-scale simplex, the epsilon-net bound is now
`sqrt(M)/(4q)`. For DTLZ2/3/4 positive hypersphere, using
the same previously proved normalization inequality, epsilon
is now `M/(2q)` instead of `M/q`.

At any q, both methods use precisely the same net.
The observed net maximum, a lower bound on the full
continuous covering radius, remains unchanged.
The mathematically valid upper bound is tightened by half
the previous epsilon; each interval width is exactly half
its former mathematical width. This is a **real-arithmetic
theorem**; the computational implementation adds 1e-12 for
numerical roundoff and does not constitute formally verified
floating-point interval arithmetic.

The experiment used actual unchanged DTLZ1/2 MOPs, M=3/5,
GT budgets 128/512, three candidate seeds 101/202/303,
q=8/16/24, FPS candidate pool 4096 (24 total comparison
cases, each with two candidate samplers). The table below
shows seed 101 at q=24; distances are in objective-space
Euclidean units:

| MOP | M | GT n | Current continuous radius [lower, upper] | FPS continuous radius [lower, upper] | Ordering established? |
|---|---:|---:|---|---|---|
| DTLZ1 | 3 | 128 | [0.119892, 0.137934] | [0.040923, 0.058965] | FPS lower |
| DTLZ1 | 3 | 512 | [0.057432, 0.075475] | [0.021293, 0.039335] | FPS lower |
| DTLZ1 | 5 | 128 | [0.290215, 0.313507] | [0.109357, 0.132650] | FPS lower |
| DTLZ1 | 5 | 512 | [0.231298, 0.254591] | [0.078527, 0.101820] | FPS lower |
| DTLZ2 | 3 | 128 | [0.153799, 0.216299] | [0.088589, 0.151089] | **FPS lower; new certified separation** |
| DTLZ2 | 3 | 512 | [0.100794, 0.163294] | [0.044448, 0.106948] | Undecided (overlap) |
| DTLZ2 | 5 | 128 | [0.476373, 0.580540] | [0.338942, 0.443109] | **FPS lower; new certified separation** |
| DTLZ2 | 5 | 512 | [0.310623, 0.414790] | [0.245180, 0.349346] | Undecided (overlap) |

These results are geometry-/budget-/seed-specific. They
establish **continuous worst-gap ordering**, not a mean-distance
ordering, IGD improvement, universal FPS superiority, or
a production GT replacement. In particular, the M=2
deterministic DTLZ2 generator remains preferable to FPS.

A separate new test verifies the systematic rounding
construction with fixed-sum fractions, empirical marginals
against deterministic offset integration, exact M=2
continuous-reference enclosure, lattice bounds and actual
MOP inverse/forward consistency. No existing regression
test is modified. To improve these certificates further,
the next mathematical research should exploit local
norms/angles or adaptive geometric nets rather than
unboundedly increasing the combinatorial q-lattice.


## Adaptive angular certification for positive hyperspheres (2026-11-19)

Research [PR #136](https://github.com/monacofj/moeabench/pull/136)
([Actions run 44606943654](https://github.com/monacofj/moeabench/actions/runs/44606943654))
investigates a second mathematical enclosure for the *same* actual
DTLZ2 GT sets already assessed by the q=24 dependent-rounding
barycentric lattice of PR #135.

The positive unit hypersphere with M objectives is parameterized
by M−1 angles in [0,π/2]. The Euclidean norm of each angular
partial derivative is at most one and those tangent vectors are
orthogonal. The coordinate map from angle space to the actual
unit hypersphere is therefore **1-Lipschitz**. For any rectangular
angular cell C with midpoint c and half-diagonal r,
`max_{theta in C} dist(f(theta),S) <= dist(f(c),S)+r`.
The distance at any evaluated midpoint is a valid **lower**
bound on the global worst continuous-front distance. An adaptive
priority queue repeatedly bisects the angular rectangle with
the largest upper bound, preserving the parent's upper bound
as a cap for its children. Stopping occurs at the requested
interval-width target or maximum active-cell budget.

The independent q=24 barycentric bounds and adaptive angular
bounds remain mathematically valid even if either method is
loose. Their **interval intersection** is valid and never
wider than either component interval. The code uses
floating-point nearest-neighbor queries, not formally verified
interval-arithmetic bounds, and the optional 1e-12 allowance
is not a formal proof of machine-rounding containment.

Real-MOP measurements used DTLZ2 M=3/5, GT budgets 128/512,
two FPS seeds 101/202, 4096-point FPS candidate pool,
4000 maximum active angular cells and target width 0.01.
The q=24 barycentric net contains 325 points for M=3 and
20,475 for M=5. The results below represent seed 101:

| M | GT n | Method | Barycentric radius [lower, upper] | Angular radius [lower, upper] | Angular cells |
|---|---:|---|---|---|---:|
| 3 | 128 | Incumbent | [0.153799, 0.216299] | **[0.176407, 0.186231]** | 158 |
| 3 | 128 | FPS | [0.088589, 0.151089] | **[0.102749, 0.112706]** | 567 |
| 3 | 512 | Incumbent | [0.100794, 0.163294] | **[0.102010, 0.111090]** | 334 |
| 3 | 512 | FPS | [0.044448, 0.106948] | **[0.054836, 0.064819]** | 1,254 |
| 5 | 128 | Incumbent | [0.476373, 0.580540] | **[0.481130, 0.490018]** | 3,381 |
| 5 | 128 | FPS | [0.338942, 0.443109] | [0.322715, 0.426362] | 4,000 |
| 5 | 512 | Incumbent | [0.310623, 0.414790] | [0.306369, 0.382807] | 4,000 |
| 5 | 512 | FPS | [0.245180, 0.349346] | [0.227010, 0.357881] | 4,000 |

For **M=3, n=512**, the angular interval proves
`R(FPS) < R(incumbent)` by nonoverlapping upper/lower bounds,
whereas the original barycentric intervals overlapped.
For M=5 the method is **not uniformly tighter**: the
fixed active-cell budget can bind before reaching target
width, especially on a higher-dimensional FPS configuration.
The combined intersection automatically adopts whichever
method provides the stronger lower and upper bounds.

At M=3 the adaptive certificates reached width below 0.01
with roughly 158–1562 active cells. At M=5, some cases
reached width below 0.01 while others remained above 0.1
after 4,000 cells. This is a clear dimensional limitation,
not a failure of the underlying mathematical inequality.
The reported timings are exploratory and depend on the CI runner.

The method certifies only Euclidean **continuous worst
covering distance**, not surface-weighted mean error,
IGD, HV convergence, an arbitrary DTLZ/DPF manifold, or a
default `calibrate()` point budget. The positive hypersphere
geometry also applies mathematically to DTLZ3/4, but only
DTLZ2 was executed in this measured matrix. No production
generator, accepted ADR, baseline, regression test or saved
GT was changed.


## Local-Jacobian angular bounds for higher-dimensional positive spheres (2026-11-19)

Independent research [PR #137](https://github.com/monacofj/moeabench/pull/137),
[Actions run 44608758755](https://github.com/monacofj/moeabench/actions/runs/44608758755),
tested a local improvement to the adaptive angular certificate of PR #136.
The original routine, its tests, and the barycentric certificate remain
**unaltered**. The new read-only script evaluates **the same GT point sets**
and intersects three independently valid intervals: original angular,
locally weighted angular, and barycentric.

**Mathematical derivation:** the M-objective positive hypersphere has
orthogonal angular tangent vectors whose norm for coordinate j is
`prod(cos(theta_i), i<j)`. On angular hyperrectangle [lo,hi],
their norm is everywhere at most
`w_j = prod(cos(lo_i), i<j)`, because the cosines are decreasing
through [0,π/2]. Along the straight parameter-space segment
joining cell center c and any theta in the cell, its instantaneous
objective-space speed is bounded by
`sqrt(sum_j (w_j * (theta_j-c_j))**2)`.
By integration the Euclidean chord between f(c) and f(theta)
is therefore at most
`r_local = sqrt(sum_j (w_j*(hi_j-lo_j)/2)**2)`.
This replaces the previous unweighted angular half-diagonal and
is **never greater** than that previous per-cell bound.
The cell with the largest admissible continuous covering upper
bound is split along the coordinate with largest weighted width;
parent upper bounds are inherited by child cells.

Every cell center supplies a lower bound. Intersecting the
real-arithmetic enclosing intervals from the old and new angular
methods and the barycentric net yields a valid combined interval,
never wider than any constituent interval. A different adaptive
split ordering may mean the locally weighted method is not
uniformly better *by itself* at the same active-cell limit. These
statements concern ideal real arithmetic; the floating-point
implementation adds 1e-12 but is **not interval-verified**.

The CI experiment used actual unmodified DTLZ2 MOPs with M=3/5,
GT budgets n=128/512, FPS pool 4096, seeds 101/202, q=24,
target width 0.01 and max 4000 active angular cells.
Selected seed-101 observations:

| M | GT n | Method | Old angular width | Local angular width | Combined width | Old/local cells |
|---|---:|---|---:|---:|---:|---|
| 3 | 128 | Incumbent | 0.009824 | 0.009568 | **0.007486** | 158 / 113 |
| 3 | 128 | FPS | 0.009958 | 0.009492 | 0.009492 | 567 / 371 |
| 3 | 512 | Incumbent | 0.009080 | 0.008325 | 0.008325 | 334 / 219 |
| 3 | 512 | FPS | 0.009982 | 0.009981 | 0.009500 | 1254 / 813 |
| 5 | 128 | Incumbent | 0.008888 | 0.009087 | **0.008863** | 3381 / 550 |
| 5 | 128 | FPS | 0.103647 | **0.016915** | **0.015898** | 4000 / 4000 |
| 5 | 512 | Incumbent | 0.076438 | **0.009415** | 0.009415 | 4000 / 2029 |
| 5 | 512 | FPS | 0.130871 | **0.055481** | **0.037966** | 4000 / 4000 |

The second seed yielded similar qualitative improvements for M=5,
with remaining combined interval widths 0.022276 (FPS n=128)
and 0.027451 (FPS n=512). The locally weighted certificate
therefore improves practical high-dimensional covering guarantees
without raising the 4000-cell budget. The FPS case can still
exhaust that budget, so this is not proof of convergence at
arbitrary M or point cardinality.

Independent *new* tests check upper Jacobian norm bounds, chords
inside randomly generated angle cells, exact M=2 continuous-radius
enclosure, monotonic adaptive refinement and intersection with
the original spherical and barycentric intervals. These experiments
certify worst-case continuous **geometric coverage only**, not
mean integration error, IGD/HV stability or a universally suitable
scientific GT density. Costs are runner dependent and are not
production performance contracts.

The accepted ADR 0018 runtime GT defaults remain intact, as
do all production MOP generators, archived GTs, original
regression tests, and numerical baselines. Choosing
`mop.calibrate()` adaptive budgets still requires explicit
review and approval under ADR 0057.

## Extension and acceptance criteria

For DTLZ7/DPF2 use the *independently derived exact admissible Pareto
regions*, then assess per-region coverage/spacing. For DPF3/4/5, do not
treat filtering nondominated points **within the sample** as proof of
true global Pareto optimality. For DTLZ8/9, compare against analytical
feasibility and geometry including disconnected or constrained
boundaries. For original DTLZ5/6 M>=4, **#65 must first characterize
the complete PF including `g>0`** before a purported uniform sample
can be certified. A provisional subset is not a GT oracle.

Before promoting a new generator: present per-MOP independent
membership/completeness evidence; cover all known components/extremes;
report finite-budget spacing/covering and convergence on multiple
references; compare IGD/IGD+/GT-based HV using fixed fronts; verify
determinism and operational cost; document regression and certification
impact. A single favorable summary statistic is insufficient.

**ADRs and test safety:** ADR 0006 governs native analytical PS/PF and
ADR 0018 currently fixes default calibration densities at 2,000 DTLZ /
10,000 DPF. This pilot changes neither. Any incompatible modification
to an accepted ADR requires explicit maintainer authorization. Warn
and obtain authorization before editing existing regression tests,
canonical GT files, numerical baselines or certification records.
Historical data remain traceable; mathematically incorrect GTs should
ultimately be replaced as current reference data after review.
