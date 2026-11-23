<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Built-in MOP Ground Truth — Post-correction Scientific Status (#119)

> **Historical snapshot — not the current certification status.**
> The table below records evidence before the final #119/#65
> closures and before subsequent source-bound certifications.
> Both issues have since been closed/completed. Original
> DTLZ5/6 M>=4 full-front sampling was corrected through PR #159,
> and the DTLZ8 production hybrid was merged through PR #146.
> Preserve the experimental findings and pending qualifications
> **as of that checkpoint**, but obtain present-day normative
> status from [ADR 0057](../../adr/0057-proposed-purpose-specific-gt-budgets.md)
> and [the GT specification](../../gt.md).
> [Proposed ADR 0059](../../adr/0059-gt-quality-measurement-and-acceptance.md)
> separately defines a future quantitative API/test gate, not
> a retroactive conclusion that every historical measurement passed.

**Snapshot:** 2026-11-21, following mainline PRs #140, #143, #146,
#147, #148, #149 and #150. This is a research **status record**, not
an assertion that the final issue #119 release certification is complete.

**Interpretation:** Analytically valid Pareto membership, finite
reference cardinality, component occupancy, spatial resolution,
regularity and a canonical reference measure are separate scientific
properties. Unit/CI success establishes the tested contracts only.
A point can be mathematically on-front even if it is a poor finite
representative of that front. For mixed-dimensional references, no
natural probability weight for components is implied by their mere
existence.

## Current MOP assessment

| MOP | Published-front / evaluator status | PS/PF correction in main | Tested positive invariants | Remaining scientific qualification |
| --- | --- | --- | --- | --- |
| DTLZ1 | Nonnegative 0.5-simplex | #148: exact simplex vertices and original seeded interior | Extreme membership, budget/replay, seeded tail | Independent dense-simplex covering and IGD/IGD+ sensitivity across M/budgets |
| DTLZ2 | Positive unit hypersphere | Previously validated analytical inverse sphere sampling | Positive unit norm, exact axes, replay | Consolidate measured covering/cost at release revision |
| DTLZ3 | Same PF, multimodal off-front g | Previously validated same sphere geometry | Positive unit norm, exact axes, replay | As DTLZ2; no arbitrary distinct sampler required |
| DTLZ4 | Same sphere, angular x^100 inverse mapping | Existing MOP-specific inverse sphere sampler | Positive unit norm, axes, decision inversion | Quantify inverse precision at higher M |
| DTLZ7 | 2^(M-1) exact disconnected record components | #149: component occupancy before repetition | Record interval membership, min(budget, component count) occupancy and replay | Within-component spatial discrepancy, cost when components grow faster than budget |
| DTLZ8 | M=2 line; M>=3 central line plus surface patches | #146: on-front analytical reference; M=4 preserving completion | Independent analytical membership/feasibility, exact unique counts, component coverage, N%M compatibility | Explicit line-versus-patch weighting, per-patch regularity/indicator sensitivity, bounded-cost policy |
| DTLZ9 | 1D constrained objective curve | #150: equal objective-space arc-length sampling | Independent quadrature, curve, endpoints, budget/replay | Strict float64 feasibility near equality is unchanged; measure runtime, numerical sensitivity and candidate cost |
| DPF1 | Nonnegative linear projection of essential 0.5-simplex | #148: exact D>2 essential vertices; D=2 grid unchanged | Projected essential vertex membership, repeatability, seeded tail | Quantify coverage in complete projected M-objective space |
| DPF2 | DTLZ7-like essential record components, nonnegative redundant projections | #149: stratify component identities for D>2 | Occupancy min(budget, 2^(D-1)), record membership, replay | Within-component spacing in projected objective space; scaling with D/M |
| DPF3 | Published sorted chaotic thresholds and essential angular geometry | #140: evaluator/sampler correction | Independent published equation and Pareto tests | Include final high-M calibration sensitivity/cost in release report |
| DPF4 | Sorted chaotic thresholds, squared projections | #143: evaluator precision and threshold correction | Independent published equation, stable small-angle behavior, deterministic chaos pool | Repeat indicator sensitivity tests on final release revision |
| DPF5 | Published P/D spherical geometry | #140: evaluator and separate branch sampling | Independent P/D geometry, numeric comparisons | Final release verification of branch weighting/precision |
| DTLZ5/6 (M>=4) | Original full front includes g>0 Pareto solutions | Not in #119 implementation scope | Current g=0-only warning | **Open #65:** mathematical completion and independently certified sampling |

## Status of regression protection

The original fixed-coordinate DTLZ8 and DTLZ9 geometric reference
assertions encoded their previous finite sampling. These were
replaced **only for their named cases**, with independent published
geometry, deterministic replay and DTLZ9 equal-arc criteria under
the explicit maintainer authorization. Historical GT JSON files,
numeric stability certificates and performance references were
not replaced. Other regression and dependency sentinels remain
active. Original research evidence, including negative findings,
is archived in this directory and prior PRs.

ADR 0057 expresses current scientific GT decisions; historical
addenda have been moved into this directory. ADR 0056 still controls
any new versioned numerical/performance/certification reference.

## Remaining acceptance gates for issue #119

1. **Final revision matrix:** exercise each supported MOP at
   representative M=2/3/5/10 where mathematically defined, relevant
   intrinsic D values, and budgets 128, 512 and the currently effective
   calibration defaults (2,000 DTLZ / 10,000 DPF). Capture actual
   cardinality, unique objectives, on-front residual, extrema,
   disconnected component occupancy, deterministic replay and cost.
2. **Independent geometric comparisons:** compare against separate
   mathematically correct dense probes or certified bounds; record
   directed covering gaps, within-component nearest-neighbor spacing
   and discrepancy at matched budgets. For disconnected/mixed-
   dimensional fronts, do not silently average unlike components.
3. **Indicator migration sensitivity:** hold MOEA populations fixed
   and measure the effect of original versus corrected finite GT on
   IGD, IGD+, and GT-dependent HV normalization. Report reversals,
   if any, without assuming old reference scores remain unchanged.
4. **Reproducible release certification:** run the complete unit,
   integration, numeric and performance suites on the designated
   source revision and approved environment, retaining certified
   reference provenance. Resolve or explicitly document all
   differences according to ADR 0056.
5. **Separate #65:** DTLZ5/6 M>=4 full-front reconstruction is
   deliberately excluded from declaring #119's other-MOP protocol
   complete. The shared methodology must be applied there later.

The scientific findings in this document support the revised
generators. **They do not license declaring issue #119 complete
before the remaining quantitative and release gates above.**

## Evidence

- [Issue #119](https://github.com/monacofj/moeabench/issues/119)
- [Read-only ten-MOP audit #142](https://github.com/monacofj/moeabench/pull/142)
- [DTLZ8 independent research #144](https://github.com/monacofj/moeabench/pull/144)
- [DPF3/DPF5 correction #140](https://github.com/monacofj/moeabench/pull/140)
- [DPF4 correction #143](https://github.com/monacofj/moeabench/pull/143)
- [DTLZ8 correction #146](https://github.com/monacofj/moeabench/pull/146)
- [DTLZ1/DPF1 correction #148](https://github.com/monacofj/moeabench/pull/148)
- [DTLZ7/DPF2 correction #149](https://github.com/monacofj/moeabench/pull/149)
- [DTLZ9 correction #150](https://github.com/monacofj/moeabench/pull/150)
