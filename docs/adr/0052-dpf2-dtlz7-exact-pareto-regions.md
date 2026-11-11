# ADR 0052 — Exact Pareto-region sampling for DPF2 and DTLZ7

**Status:** Proposed (source generator correction verified; archive compatibility under audit)  
**Date:** 2026-11-11  
**Issue:** [#78](https://github.com/monacofj/moeabench/issues/78)  
**Branch:** `fix/78-dpf2-dtlz7-pareto-sources`  
**PR:** [#79](https://github.com/monacofj/moeabench/pull/79)  
**Upstream FAIR investigation:** [#75](https://github.com/monacofj/moeabench/issues/75), [PR #77](https://github.com/monacofj/moeabench/pull/77), and [ADR 0051](https://github.com/monacofj/moeabench/blob/fix/75-fair-core/docs/adr/0051-fair-conformance-audit.md)

## Context and verified failure

The 14-MOP FAIR source audit found that `DPF2.pf()` and `DTLZ7.pf()`
contain **dominated points** despite being documented as analytical
Pareto-front sources. In the original source, DPF2's inherited
`BaseDPF.ps()` linearly sampled its DTLZ7-like base parameter over [0,1],
and DTLZ7's `ps()` generated an unrestricted `RandomState(42)` uniform
cube. With 1,024 sampled points, **47.9492%** (DPF2) and **33.4961%**
(DTLZ7) of unique objectives were within-sample nondominated. Their
remaining points were unsuitable for estimating an *ideal* finite-GT
Closeness residue.

This issue concerns the mathematical domain of the source PF,
not whether normalized-objective surface-area sampling should be
adopted (the latter remains experimental in [PR #76](https://github.com/monacofj/moeabench/pull/76)).

## Analytical derivation

For DTLZ7 at the Pareto-optimal distance variable `g=1`,

```text
f_i = x_i, i=1,...,M-1
f_M = 2 M - Σ_i a(x_i),     a(x)=x (1+sin(3πx)).
```

For any parameter `x_i`, if there exists `t < x_i` with
`a(t) >= a(x_i)`, replacing `x_i` by `t` does not worsen any
objective and strictly improves `f_i`. Thus the original solution is
dominated. Conversely, if every `x_i` is a **strict running maximum**
of `a` over all preceding `t<x_i`, every decrease of an `x_i`
strictly decreases its contribution `a(x_i)` and therefore worsens
`f_M`; no alternative with all `f_i` no worse can dominate it.
Accordingly this strict-record property is a necessary *and sufficient*
characterization of the global Pareto-set parameter domain.

On `[0,1]` the strict-record regions are

```text
[0, α] ∪ (β, γ]
α = 0.2514118360889171   (first local maximum)
β = 0.6316265307000614   (second-region equal-record reentry)
γ = 0.8594008566447239   (second local maximum).
```

The boundaries are derived numerically from the **analytical**
`a'(x)=1+sin(3πx)+3πx cos(3πx)`, not chosen by threshold tuning.
The number of disconnected components is `2^(M-1)`.
The limiting intervals agree with the independently published DTLZ7
formulation (PlatEMO's `[0,.251412,.631627,.859401]`; see
https://gitlab.inf.elte.hu/CBMA-Darts-Playing-Robot/darts-playing-robot/-/blob/master/Problem%20integration%20with%20PlatEMO/Problems/Multi-objective%20optimization/DTLZ/DTLZ7.m).

DPF2 has an analogous `D`-objective DTLZ7-like base for
`g=1`. Its added objectives are nonnegative coordinatewise-monotone
(squared positive weighted sums of its base objectives) and
retain all `D` base objectives unchanged. Therefore they **cannot
create or remove** Pareto dominance relative to the base: the same
strict-record restriction applies independently to its `D-1`
position parameters.

## Decision

Introduce a small, shared source helper
`moeabench/mops/_dtlz7_pareto_regions.py` that computes the analytical
roots once (bounded Brent roots) and maps uniform [0,1] draws to the
two allowed intervals, weighted by **their decision-parameter lengths**.

- DTLZ7 keeps a private `RandomState(42)` and transforms all
  `M-1` position parameters to the true Pareto-record domain.
- DPF2 at `D=2` keeps a deterministic, ordered decision grid
  but maps it to the disconnected domain; `D>2` uses the same
  `RandomState(42)` parameter-law family as the old base.
- Only analytical `ps()` changes. Neither `evaluation()` nor
  the optimization landscape is changed.
- The distribution is **uniform in the allowed decision parameters**,
  not uniform in physical or normalized objective-space surface
  measure. The discrete grid/seed remain reproducible.
- Include all disconnected components, test sampled non-dominance
  at M=2,3,4 and DPF2 D=2,3, verify strict-record inequalities,
  exact boundary roots, deterministic seeds and global NumPy RNG isolation.

## Consequences and separation of concerns

The archived v0.14.0 GT and clinical baselines were produced under the
**previous unrestricted PF source**, which included dominated points.
Correcting `ps()` **does not retroactively repair** those archived GTs,
their frozen normalizations, or any Closeness/Q-score. The appropriate
next action is an explicit comparison of the corrected source with the
old GT, and a separately reviewed GT/calibration decision. Do not
relabel the old baseline with a later library version or silently
overwrite immutable frozen references.

DPF5's mixed-dimensional branch-weighting problem and DPF3/DTLZ8's
independent GT incompatibility remain separate.

## Chronological implementation and validation log

### 2026-11-11 — Derivation, isolated implementation and first CI

- Verified source `DTLZ7.py`, `DPF2.py`, `BaseDPF` and
  original unit-PF sampling. Independently checked numerical
  derivative roots with SciPy and corresponding published DTLZ7
  region endpoints; no arbitrary ND filtering alone was accepted.
- Created `fix/78-dpf2-dtlz7-pareto-sources` **from main**,
  independent of the larger PR #77 core FAIR changes. Added the shared
  region helper, corrected DPF2/DTLZ7 `ps()` and deterministic regression
  tests. Created [Draft PR #79](https://github.com/monacofj/moeabench/pull/79).
- **PASS — [GitHub Actions #43077969491](https://github.com/monacofj/moeabench/actions/runs/43077969491):**
  **9/9 issue-focused tests and complete 279/279 unit suite passed.**
  The tests cover exact numerical endpoints, strict prefix-record property,
  original-source dominated-point counterexamples, all `2^(M-1)`
  components, DPF2 higher D, deterministic PRNG isolation, exact
  objective evaluation and no official-baseline changes.
- Added `calibration/scripts/audit_archived_pareto_sources_issue78.py`
  to report frozen-GT vs corrected-PF compatibility *without*
  rerunning/replacing baselines. A new CI run must confirm the
  quantitative legacy-GT evidence before declaring the source
  correction ready for review; no claim of an updated clinical Q
  baseline is made.

**Release gate:** Do not merge a modified GT, source calibration or
canonical `audit_report` as part of this isolated source-PF correction.
Scientific review of the legacy GT and subsequent `ideal_res`/ECDF
compatibility remains pending.


### 2026-11-11 — Frozen-GT compatibility measured, WITHOUT modifying historic references

- **PASS — [CI 43078550239](https://github.com/monacofj/moeabench/actions/runs/43078550239):** Repeated **9/9 focused and 279/279 unit tests**, then ran `audit_archived_pareto_sources_issue78.py` using 1,024 sampled frozen-GT points (seed 1337) and the exact 1,024-point newly corrected analytical source for each MOP. Verified original v0.14.0 JSON GT equals packaged NPZ GT byte-for-byte in numeric arrays; archived frozen baseline and GT package unchanged. Read-only evidence artifact `issue78-frozen-gt-versus-corrected-pf-not-certified` ID **13883601215**.
- **DPF2:** Among 1,024 distinct original archived GT points, **504 (49.21875%)** are nondominated *within that sample*. The 1,024 unique corrected PF points are **100%** within-sample nondominated. Corrected source to frozen 10k GT median nearest-neighbor distance = **0.236421** frozen-GT local spacings (95th percentile 1.050716). Proximity to the old GT does not certify that GT's dominated points as valid Pareto references.
- **DTLZ7:** Among 1,024 distinct original archived GT points, only **311 (30.37109%)** are nondominated *within that sample*. The 1,024 unique corrected PF points are **100%** within-sample nondominated. Corrected-to-frozen-GT nearest distance median **1.032925** local spacings (95th percentile 2.304114). This large discrepancy indicates that recalibration against the unchanged old GT would need careful scientific evaluation; it cannot simply be assumed valid.
- **IMPORTANT:** All fractions above describe a **bounded sample** and do not claim a full 10k-point Pareto classification. A 100% within-sample ND rate also is not a substitute for the global prefix-record proof above. Do not run `stage_closeness_adr0037.py` and promote the DPF2/DTLZ7 `ideal_res` against old GT without an explicit reference-GT review.
- **NEXT ENGINEERING GATE:** Basic integration with real regenerated fixture experiments has been added to temporary branch workflow; its CI result is pending. Afterwards remove the temporary workflow and update PR/issue evidence before merge.
