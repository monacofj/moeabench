<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench Documentation Index

This index explains the purpose of MoeaBench's main documents and suggests where to start reading. For a feature, consult the appropriate guide or technical reference; for **why** a particular architecture was chosen, consult the Architecture Decision Records (ADRs); for scientific evidence, hypotheses, mathematical proofs, and experimental results, consult the specialized research reports.

**Reference documentation** explains concepts and contracts that should remain useful independently of any issue's progress. **Research reports** preserve reproducible evidence and experimental reasoning. **ADRs** record decisions, their status, and subsequent supersession; a proposed ADR must never be treated as an approved contract.

## General Documentation and API

| Document | Contents | When to read |
|---|---|---|
| [User Guide](userguide.md) | Installation, first experiments, and MoeaBench workflows | Learning how to use the software |
| [API Reference](reference.md) | Interfaces, classes, methods, and documented parameters | Checking programming contracts |
| [API Sheet](api_sheet.md) | Concise map of common operations and workflows | Finding an API entry point quickly |
| [Ground Truth](gt.md) | PS/PF/GT concepts, MOP-owned reference generation, DPF3/DPF5 mathematics, sampling quality, numerical precision, indicators, and historical references | Understanding how to define and validate a scientific reference |
| [Contributing](CONTRIBUTING.md) | Development and contribution guidelines | Making or reviewing code contributions |
| [ADR Index](adr/README.md) | Architectural decisions, accepted/proposed status, and supersession lineage | Checking architectural rules before changing contracts |

The repository's [main README](../README.md) contains an overview, examples, citation guidance, and bibliographical information. This **index.md** organizes technical documentation; it is neither a bibliography nor a replacement for the repository README.

## Ground Truth: From Theory to Reproducible Evidence

The permanent, self-contained technical reference is **[gt.md](gt.md)**. The files below are **specialized reports from [issue #119](https://github.com/monacofj/moeabench/issues/119) and [PR #139](https://github.com/monacofj/moeabench/pull/139)**. They preserve detailed proofs, experimental designs, data, negative results, comparisons against the audited implementation, and traceability. They are grouped under **[research/gt-issue119](research/gt-issue119/README.md)**, rather than scattered at the top level of \`docs/\`.

### Mathematical Foundations and Proofs

| Document | Scope |
|---|---|
| [DPF3 global Pareto set/front proof](research/gt-issue119/gt-dpf3-global-pareto-proof-issue119.md) | Ordered chaotic thresholds, projection-order preservation, and proof that published PS equals the set where g=0 |
| [DPF5 global Pareto set/front proof](research/gt-issue119/gt-dpf5-global-pareto-proof-issue119.md) | Hyperspherical P/D branches, global membership, completeness, and cross-branch nondominance |
| [DPF5 componentwise angular covering](research/gt-issue119/gt-dpf5-angular-cover-candidate-issue119.md) | Conditional real-arithmetic continuous covering bounds for complete P/D grids and explicit float64/cost limitations |
| [DPF3 angular covering at D=2](research/gt-issue119/gt-dpf3-d2-angular-cover-proof-issue119.md) | Finite angular construction, exact-real-arithmetic Hausdorff bound, and float64 distinctions |
| [DPF3 general-dimensional angular covering](research/gt-issue119/gt-dpf3-nd-cover-and-acceptance-issue119.md) | Higher-D proof, sample degeneracies, tensor cost, roundoff measurements, and scientific quality criteria |

### Equation and Numerical Audits

| Document | Scope |
|---|---|
| [Initial PS/PF diagnostic](research/gt-issue119/gt-dpf3-dpf5-audit-issue119.md) | Requested/actual counts, duplicates, and sample-relative dominance in audited evaluators |
| [Published-formula triage](research/gt-issue119/gt-dpf-published-formula-triage-issue119.md) | Initial identification of DPF3 threshold ordering and the DPF5 P/D equality-boundary discrepancy |
| [General-D DPF5 reconstruction](research/gt-issue119/gt-dpf-general-d-evidence-issue119.md) | Independent P/beta and D/gamma reconstruction for higher intrinsic dimensions, g=0 and g>0 |
| [DPF3/DPF5 numerical kernels](research/gt-issue119/gt-dpf-numeric-kernel-evidence-issue119.md) | 1−cos cancellation, stable trigonometric expressions, spherical invariants, and evaluator differences |
| [High-precision numerical sweep](research/gt-issue119/gt-dpf-high-precision-sweep-issue119.md) | High-precision reference values, underflow, representability limits, and dimensional checks |
| [DPF5 off-front audit](research/gt-issue119/gt-dpf5-offpareto-issue119.md) | g>0 evaluations, scaled published equations, and optional research dependency isolation |

### Finite Sampling, Sampler Comparisons, and Metrics

| Document | Scope |
|---|---|
| [DPF5 branchwise finite coverage](research/gt-issue119/gt-dpf5-finite-sampler-coverage-issue119.md) | Independent P/D coverage probes, effective counts, and differences between implemented and published boundaries |
| [Fixed-unique-budget Sobol comparison](research/gt-issue119/gt-dpf5-branch-sobol-comparison-issue119.md) | Controlled branchwise experiment, showing both improvements and degradations |
| [P/D point-budget allocation](research/gt-issue119/gt-dpf5-branch-allocation-issue119.md) | Incumbent versus 50%/approximately 67% P allocation, across seeds with equal total effective budgets |
| [Impact on IGD, IGD+, and HV](research/gt-issue119/gt-dpf5-paired-indicator-impact-issue119.md) | Isolation of evaluator and GT-reference effects on indicators with fixed populations |

### Change Management and Reproducibility

| Document | Scope |
|---|---|
| [Production impact inventory](research/gt-issue119/gt-dpf-production-impact-inventory-issue119.md) | Source files, consumers, tests, historical references, and potential correction impact |
| [Correction and versioned migration proposal](research/gt-issue119/gt-dpf-correction-migration-proposal-issue119.md) | Proposed sequence and approval requirements for evaluator fixes, samplers, frozen references, and calibration budgets |
| [DPF3/DPF5 historical regression migration review](research/gt-issue119/gt-correction-test-migration-issue119.md) | Exact pre-existing negative-diagnostic test failures, unchanged DPF5 frozen baseline, and explicit test-approval gates |
| [PR #139 pre-merge review](research/gt-issue119/gt-dpf-premerge-review-issue119.md) | Research-only scope, safeguards, changed-file inventory, and CI verification |

Earlier reports may contain a historical statement such as "global membership has not yet been established." **The consolidated results are given in [gt.md](gt.md) and the subsequent mathematical proofs.** Preliminary observations must not be mistaken for the final conclusions.

## Architecture Decision Records Relevant to GT

The [full ADR index](adr/README.md) provides each decision's status, history, and supersession lineage. The following are the most relevant entry points for GT work:

| ADR | Subject | Relevance |
|---|---|---|
| [0006](adr/0006-benchmarks-reimplementation.md) | Benchmark reimplementation | MOP ownership of mathematical evaluation and analytical PS/PF |
| [0016](adr/0016-evolution-of-scientific-validation-and-guided-geometric-solvers.md) | Guided geometrical solvers | Analytical and geometry-aware reference construction |
| [0017](adr/0017-comprehensive-calibration-and-testing-framework.md) | Scientific calibration and testing | Analytical references and historical baseline roles |
| [0018](adr/0018-consolidated-scientific-report-and-gt-density-refinement.md) | Historical GT density | Legacy budgets, partially superseded in principle by 0057 |
| [0021](adr/0021-numerical-integrity-and-validation-pipeline-float64-policy.md) | Numerical integrity | float64 representation and validation rules |
| [0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md) | Versioned certifications | Historical preservation and separate certification lifecycles |
| [0057](adr/0057-proposed-purpose-specific-gt-budgets.md) | Purpose-specific GT budgets | MOP-owned on-demand generation and calibration; numerical implementation distinct from in-principle decisions |

[ADR 0053](adr/0053-dpf5-mixed-dimensional-reference-measure.md) records **proposed, noncertified research** concerning mixed-dimensional distributions and historical FAIR studies. It is **not** an adopted production GT policy.

## How to Navigate

To **understand** Ground Truth science, read [gt.md](gt.md). To **implement** algorithms or change APIs, inspect the applicable [ADRs](adr/README.md) and [API reference](reference.md). To **verify a proof or reproduce an experiment**, follow the specialized report and its research script/tests in PR #139. To understand **why** an architectural choice was made, consult the ADR's full history and current status, including later amendments.

This index organizes knowledge; it neither turns an experimental finding into an accepted architectural decision nor turns a historical record into a current runtime contract.
