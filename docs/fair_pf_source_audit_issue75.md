# FAIR Closeness: independent Pareto-front source audit (issue #75)

**2026-11-11 — Screening evidence, NOT scientific baseline certification.**

The separate correction PR #77 uses the existing ADR 0037 definition of
`ideal_res`: a median distance to frozen GT from a genuinely fresh
analytical Pareto-front sample, divided by frozen resolution `s_K`.
It does **not** adopt the proposed uniform normalized-objective-area law
studied separately in PR #76.

## Scope and method

The provisional `stage_closeness_adr0037.py` initially found an independent
analytical source with acceptable *near-GT geometry* for **12 of 14** archived
MOPs and staged **528 = 12 × 44** corrected K-wise Closeness records.
**Those tests did not prove that the sources actually sample global
non-dominated Pareto-optimal points or have an approved probability law.**

The independent `audit_pf_sources_issue75.py` therefore additionally
examines the *actual* `MOP.ps()/pf()` code paths and the original frozen
GT (full, exact SHA256 match), on bounded 1,024-point analytical samples:
unique objective vectors; within-sample Pareto nondominance using pymoo;
constrained feasibility when applicable; and conditional/degenerate
source geometry. The fresh analytical-reference pool is also screened.
The audit records the source-code and GT hashes. **Within-sample
nondominance is necessary, not a proof of global Pareto optimality**;
near-GT density is not evidence that two probability laws agree.

[Verified GitHub Actions #43073491535](https://github.com/monacofj/moeabench/actions/runs/43073491535):
six focused tests and the full **310/310 unit suite passed**, with unchanged
official baseline/GT and canonical `audit_report`. Its
`fair-75-fourteen-mop-pf-source-audit-not-certified` JSON artifact
(ID **13878664242**) contains per-MOP SHA256, cardinality, sampled
nondominance/feasibility, geometry and source-law indicators.

## Results, M=3, 1,024 sampled points

| MOP | Analytical sampling mechanism | Unique sample non-dominated | Audit interpretation |
|---|---|---:|---|
| DPF1 | Linear base decision-parameter grid, 1D degenerate curve | 100% | Conditional; parameter-grid law, not objective-area uniformity |
| DPF2 | Linear base grid mapped by DTLZ7-like nonmonotonic objectives | **47.95%** | **Hold:** analytical source contains dominated points |
| DPF3 | Linear base grid followed by non-dominated filtering (variable cardinality) | 100% after filter; ~55.13% before | **Unsupported:** independent analytic sample not compatible with packaged GT |
| DPF4 | Linear base grid, chaotic piecewise projection | 100% | Conditional; objective-space law not independently justified |
| DPF5 | Uniform 2D decision grid; different objective formulas at `x1 > 1/3` | 100% among distinct points | **Hold:** dimensional collapse and objective-deduplication change sampling weights |
| DTLZ1 | Uniform random Pareto-set decision parameters (fixed seed) | 100% | Conditional; pushforward on simplex not uniform objective-area measure |
| DTLZ2 | Positive-Gaussian sphere directions; special axis extrema | 100% | Conditional; known physical sphere law, frozen objective normalization differs |
| DTLZ3 | Inherited DTLZ2 positive-sphere sampling | 100% | Conditional; known physical sphere law |
| DTLZ4 | Inverse-power parameterization of positive-sphere directions | 100% | Conditional; front law versus normalized geometry requires review |
| DTLZ5 | Uniform random decision parameters, degenerate PF curve | 100% | Conditional; extra decision dimensions do not increase PF dimension |
| DTLZ6 | Uniform random decision parameters, degenerate PF curve | 100% | Conditional; PF density and degenerate geometry require review |
| DTLZ7 | Uniform random decision parameters, no PF-region filter | **33.50%** | **Hold:** analytical source contains dominated points |
| DTLZ8 | Static optimal-X CSV (M3) or guided solver | 100% among sampled points | **Unsupported:** no compatible new analytical sample independent of frozen GT |
| DTLZ9 | Linear `f_M` grid mapped back to feasible decision variables | 100% | Conditional; 1D curve, parameter-uniform not necessarily arc-uniform |

**DPF5 quantitative finding.** For the actual 32×32 M=3 decision grid
(`n_points=1024`), the branch `x1>1/3` contains **672/1024 = 65.625%**
of all decision samples. In that branch the objective function depends only
on `x1` and not on `x2`, leaving **21 unique objective points**;
the other branch has **352 unique objective points**. After objective
deduplication (which `independent_analytical_reference()` performs), the
branch has only **21/(352+21) = 5.630%** of unique-objective mass, a **11.656×**
relative change. Its underlying PF combines a 2D patch with a collapsed
1D branch. Neither relative weighting was selected as a documented normative
measure; closeness to GT cannot resolve this ambiguity.

**DPF2/DTLZ7 quantitative finding.** Their current analytical generators
produce dominated sample points, even after removing duplicate objective
vectors; these are not simply uncertainty over which uniform distribution
to adopt. The observed dominated fractions are approximately **52.05%**
(DPF2) and **66.50%** (DTLZ7), respectively. An ideal-residue reference
must not be certified from those samplers as they stand.

## Decision for the Closeness migration

- **9 MOPs remain conditional** after this necessary screening:
  DPF1, DPF4, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ9.
  Their **9 × 44 = 396** K records are *candidates for further scientific
  review*, **not approved or installed**. Their source sampling laws still
  require an explicit interpretation, even if they pass sampled ND.
- **3 MOPs are on hold**: DPF2 and DTLZ7 require analytical PF samplers
  that exclude dominated regions; DPF5 requires a declared probability
  weighting across its branches and an independent consistency test.
  Their **132** previously staged records must **not** be promoted.
- **2 MOPs are unsupported** by the independent finite-GT screen: DPF3
  and DTLZ8; their **88** records were not staged and their unavailable
  correction must not masquerade as a measured `ideal_res=0`.

**All 528 previously generated records remain exploratory**, and
the official baseline v0.14.0 and `calibration/audit_report.html` remain
untouched. The source version 0.14.0 versus consumer library 0.16.1
is provenance, **not** a reason to recalibrate.

## Next work

Resolve the mathematical definition and domain of the sampling law for
the nine conditional cases, separating genuinely Pareto-optimal source
samplers from distributions that may be biased by nonlinear parameterization.
Repair or independently replace the demonstrably invalid DPF2/DTLZ7 PF
samplers, and define the intended branch-weighting law for DPF5.
Only then revise the per-MOP/K eligible Closeness/ECDF values with an
explicit comparison of real clinical Q-scores and no retrospective drift
tolerance. Do not silently introduce the experimental normalized-area law
or publish the 528-record preview as a certified baseline.
