<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF5 fixed-total effective budget P/D allocation study — issue #119

**Status: exploratory read-only evidence, no reference law adopted.** [Research CI #44730592452](https://github.com/monacofj/moeabench/actions/runs/44730592452) **PASS, 75 independent tests**. Source: `scripts/audit/gt_dpf5_allocation_issue119.py`, new isolated tests `tests/unit/test_gt_dpf5_allocation_issue119.py`; machine-readable JSON stored in that workflow's artifact.

## Design and interpretation

For each requested budget 128 or 512 and DPF5 (M,D)=(3,2),(5,2),(6,3),(6,4), the current unmodified PS decisions are evaluated using the independently reconstructed **published** equations. The *total* number of objective vectors, counting unique objectives separately per P and D branch, is frozen. Three counterfactual allocations are compared: original branch counts, 50% in P, and approximately 2/3 in P. Candidate samples use deterministic scrambled Sobol with limited boundary anchors, generated at each of sampler seeds 71/72/73. Every comparison is measured against the same three independent 256-point per-branch probe sets with seeds 201/202/203. The criterion shown here is the largest observed conditional nearest-neighbor distance from either branch across all three probe sets (worst-seed observed distance). We report each sampler seed separately and summarize the **maximum over sampler seeds**, a conservative descriptive quantity.

The current baseline repeats exactly across sampler seeds because it is deterministic and unchanged; it is not independent replication. Exact global uniqueness (after concatenation of branch samples) is recorded separately from fixed branchwise unique counts. In the tested configurations the global exact-unique count was constant for each scenario (equal to the branchwise total), but this is an observation, not a general theorem. **No probability measure, branch weighting or universal geometry budget is selected** by this max-conditional-distance choice. The candidate timing values in JSON are single wall-clock observations and not benchmark certificates. The anchors cover selected first-angle endpoints and angular corners, **not** a certified full boundary or complete set of extrema.

## Worst observed conditional nearest-point distance at requested 512

| M,D | Incumbent | Half P | Two-thirds P | Winning among tested allocations |
|---|---:|---:|---:|---|
| 3,2 | **0.07705** | 0.10917 | 0.08702 | incumbent |
| 5,2 | 0.41784 | 0.34187 | **0.33189** | 2/3 P |
| 6,3 | 0.45091 | 0.42947 | **0.38824** | 2/3 P |
| 6,4 | 0.45091 | 0.42947 | **0.38824** | 2/3 P |

At requested 128:
- (3,2) incumbent 0.10695, half P up to 0.23817, 2/3 P up to 0.19169 — incumbent remains best.
- (5,2) incumbent 0.48220, half P up to 0.55323, 2/3 P up to 0.47100 — minor advantage to 2/3 P.
- (6,3) and (6,4) incumbent 0.88305, half P up to 0.75362, 2/3 P up to 0.58406 — clear reduction in measured conditional gap with more P points.

## Scientific interpretation and limits

The DPF5 P branch is an (M−1)-dimensional spherical patch while the D branch is a (D−1)-dimensional subset. Increasing P's share at fixed total can be beneficial when P's covering dimension is much larger; allocating a uniform proportion to P across all configurations is not justified by these numbers. For M3/D2 the incumbent's 154 P / 15 D *unique* budget at requested 512 already assigns ~91% to P. The 2/3 P candidate therefore **takes points away from P**, which can hurt. This matters more than just the numerical labels 'incumbent/half/two-thirds'.

A max-conditional-distance diagnostic is **not** an objective-space Hausdorff or certified worst-case covering radius. Further investigations should independently derive rigorous geometric covering controls and explicit precision/cost criteria, and compare deterministic candidates without assuming a unique mixed-dimensional uniform probability law. Evaluator correctness and sampler correctness must remain separate from the automatic budget decision under accepted ADR0057.

No production MOP evaluator, PS/PF, calibrate, original regression, accepted ADR, archived GT, sidecar, baseline, or FAIR research changed. PR #139 remains Draft and unmerged.
