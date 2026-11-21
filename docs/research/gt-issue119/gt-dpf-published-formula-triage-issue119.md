<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Published-formulation discrepancy triage — DPF3 and DPF5

**Issue #119 / PR #139, research-only.** This note compares the published formulas, not merely outputs of a sampled PF. It is **not** authorization to edit the MOP implementation, regression tests, reference data, or an accepted ADR.

Primary reference: Zhen, Li, Cheng, Peng and Yao, *Multiobjective Test Problems with Degenerate Pareto Fronts*, arXiv:1806.02706v2 (2022), §IV-C, §IV-E, §IV-F, equations (15)–(16), (19)–(21), https://arxiv.org/abs/1806.02706 and author-hosted https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf.

## DPF3: projection order (source-level discrepancy)

The original article requires **η₁ < η₂ < … < η_(m−d)** and explicitly says (§IV-F) that logistic chaotic outputs used for η in DPF3/4 are **sorted increasingly**.

Current `BaseDPF._generate_chaos()` yields the logistic values in generation order, without sorting. Current `DPF3.evaluation()` obtains `chaos = self._chaos_pool[:U]` and uses successive entries directly for the min/max projections. Therefore **the sorted-η prerequisite is not enforced**; for U≥2 it may be violated. This is a source-level discrepancy with the published parameter construction. It can create a nonmonotone or collapsed projection arrangement and affects whether the resulting problem is the prescribed DPF3. The M=3 case has U=1, where ordering is vacuous; its 128→23 / 512→88 cardinality reduction cannot be attributed to parameter misordering alone.

**Required next verification:** derive the exact piecewise projection with the original sorted thresholds and independently evaluate matched decision points with current vs mathematically prescribed projection, for M=4/5/10, without changing the original evaluator. Measure dominance pairs and whether sorting resolves any violations; do not use regenerated PF to mask evaluator differences. A separate mathematical proof of global Pareto membership remains required.

## DPF5: branch threshold (source-level discrepancy)

The published equation (19) assigns the β branch for **x₁ < 1/3** and γ otherwise. The current evaluator selects γ/D when `x1 > 1/3`, i.e. assigns β/P at equality. This is a different mathematical definition on the threshold boundary. Its effect depends on exact boundary values and branch continuity and should be tested independently; random samplers generally miss a measure-zero boundary, but structured grids can include it.

The published DPF5 formulation explicitly defines a partial nondegenerate `(m−1)`-dimensional PF segment and a degenerate `(d−1)`-dimensional segment. The reported 462/512 internally dominated M5 vectors require a **term-by-term objective-kernel check** against the published β and γ functions, including indexing of the m−d+1 first functions, before inferring a sampling-only fault. The text extraction of the paper's piecewise typeset equations is insufficient for a confident index-by-index sign-off; verify against the original typeset page (or authors' reference source).

**Required next verification:** create fresh independent equality/left/right boundary probes; derive the β/γ objective-vector formulas for representative m,d; compare every component to the existing evaluator for mathematically valid PS (g=0). Distinguish *within-P*, *within-D* and *cross-branch* strict dominance. Compare valid finite supports at equal **unique objective-vector count**, preserving all components and extremes.

## Current reproducible observational evidence

[PR #139](https://github.com/monacofj/moeabench/pull/139), [CI #44617511658](https://github.com/monacofj/moeabench/actions/runs/44617511658) has 6 passing new tests and separate green regression and stability workflows, and [JSON artifact](https://github.com/monacofj/moeabench/actions/runs/44617511658/artifacts/14576056220). Its finite-sample counts are in `docs/gt-dpf3-dpf5-audit-issue119.md`. Those are **diagnostics**, not complete mathematical validation of either PF.

Accepted ADR 0006/0018/0057 are preserved. No existing regression test, production evaluator, `ps()`, `pf()`, `calibrate()`, GT sidecar, frozen baseline, or other study is changed.
