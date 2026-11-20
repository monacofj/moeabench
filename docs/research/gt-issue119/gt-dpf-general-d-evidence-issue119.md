<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Published DPF5 equation reconstruction for general D — issue #119

**Research-only; no MOP or reference mutation.** This study derives DPF5's original beta/gamma objective functions for general M>D>=2 from the source publication: Zhen et al., *Multiobjective Test Problems with Degenerate Pareto Fronts*, arXiv:1806.02706v2, §IV-E, eqs. (19)–(20), https://liangli-zhen.github.io/assets/pdf/arxiv2022_DPF.pdf. Distinct from the previous D=2 study, it checks intrinsic D=3/4 at g=0 and g=0.04. Numerical evidence is in `scripts/audit/gt_dpf5_general_d_issue119.py`, its independent new tests and the PR #139 Actions artifact.

**Independent mathematical reconstruction:** Let H_q be the q-dimensional standard nonnegative hyperspherical map with q−1 angles and unit Euclidean norm. For beta/P, the published formula is precisely H_M of all M−1 position angles; for gamma/D, take H_D on the first D−1 angles, repeat its first coordinate **M−D+1** times divided by sqrt(M−D+1), then append the other D−1 spherical coordinates. Finally multiply by (1+g) and choose beta when x1 < 1/3, gamma otherwise. Both branches therefore satisfy sum f_i²=(1+g)² exactly in real arithmetic. This establishes a *necessary* mathematical identity of these published formulas, not a proof of global Pareto completeness.

**Observed source issue at D>2:** Direct comparison with the actual unchanged DPF5 evaluator shows incorrect angle reuse/indexing for some D>2 parameter combinations, affecting the P branch and, at D=4, the gamma/D branch as well. For the deterministic 128-point, strictly interior branch sample with g=0:
- (M=6,D=3) current P max scaled squared-norm error ~0.171273; reconstructed P and current D are at numerical precision
- (M=8,D=3) current P error ~0.451484; reconstructed P/current D near numerical precision
- (M=6,D=4) current P error ~0.314780 **and current D error ~0.321684**, while both reconstructed branches are near numerical precision
- (M=8,D=4) current P error ~0.316033 and D error ~0.321684, with reconstructed branches near numerical precision

The first CI research job passed 44 new tests, but its initial reporting script accidentally overwrote the numeric `D` field with the degenerate-branch diagnostic. This was a **report-schema defect**, not a production change. Corrected field names `intrinsic_D`, `branch_P`, and `branch_D` have been committed, plus an explicit new independent assertion for the D=4 gamma discrepancy; all of the new checks must pass again on this corrected revision before the results are final.

**Limitations:** These fixed-box and off-PF probes test the *evaluation formulas*, not global Pareto membership, complete PF support, GT cardinality/coverage, or indicator error; the original PS/PF generators and all reference data remain untouched. Before any production correction, review ADRs 0006/0018/0057 and identify the affected existing numerical regression and archived GT references. Warn and obtain explicit approval before an ADR-conflicting edit or change to existing regressions.
