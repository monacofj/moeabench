<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Architecture Decision Records (ADR)

This directory contains records of the key architectural decisions made during the evolution of moeabench.

| ID | Title | Status | Date |
|---|---|---|---|
| [0001](0001-oo-refactoring.md) | Object-Oriented Refactoring | Accepted | 2026-01-14 |
| [0002](0002-algorithm-wrappers.md) | Algorithm Wrapper Strategy | Accepted | 2026-01-14 |
| [0003](0003-vectorized-performance.md) | NumPy Vectorized Performance | Accepted | 2026-01-14 |
| [0004](0004-analysis-specialization.md) | Analysis-First Specialization | Accepted | 2026-01-14 |
| [0005](0005-parallelism-removal.md) | Parallelism Removal | Accepted | 2026-01-14 |
| [0006](0006-benchmarks-reimplementation.md) | Benchmark Reimplementation Narrative | Accepted | 2026-01-14 |
| [0007](0007-vectorized-nds.md) | Vectorized Non-Dominated Sorting | Accepted | 2026-01-14 |
| [0008](0008-population-stratification.md) | Population Strata (Rank Distribution) | Accepted | 2026-01-14 |
| [0009](0009-tier-nomenclature.md) | Tier-based Competitive Analysis | Accepted | 2026-01-16 |
| [0010](0010-rich-stats-results.md) | Rich Result Objects and Narrative Reporting | Accepted | 2026-01-14 |
| [0011](0011-selective-persistence.md) | Selective Persistence (Save/Load) | Accepted | 2026-01-17 |
| [0012](0012-manager-mode.md) | Manager Mode: Experiment Delegation Semantics | Accepted | 2026-01-18 |
| [0013](0013-visual-identity.md) | Centralized Visual Identity (Ocean Palette) | Accepted | 2026-01-18 |
| [0014](0014-semantic-taxonomy.md) | Semantic Taxonomy (Performance vs. Topology) | Accepted | 2026-01-19 |
| [0015](0015-scientific-ground-truth-and-numerical-integrity.md) | Ground Truth Integrity | Superseded by 0016 | 2026-01-20 |
| [0016](0016-evolution-of-scientific-validation-and-guided-geometric-solvers.md) | Guided Geometric Solvers | Accepted | 2026-01-25 |
| [0017](0017-comprehensive-calibration-and-testing-framework.md) | Calibration & Testing Framework | Accepted | 2026-02-01 |
| [0018](0018-consolidated-scientific-report-and-gt-density-refinement.md) | Consolidated Report & GT Density | Accepted | 2026-02-02 |
| [0019](0019-tripartite-hypervolume-metrics-nomenclature.md) | Tripartite Hypervolume Metrics | Partially Superseded by 0047 (`rel`) and 0058 (`abs`) | 2026-02-03 |
| [0020](0020-standardized-reference-point-strategy-for-normalization.md) | Standardized Reference Point Strategy | Accepted | 2026-02-04 |
| [0021](0021-numerical-integrity-and-validation-pipeline-float64-policy.md) | Numerical Integrity (Float64 Policy) | Accepted | 2026-02-05 |
| [0022](0022-emd-as-primary-diagnostic-for-degenerate-manifolds.md) | EMD as Primary Diagnostic | Accepted | 2026-02-06 |
| [0023](0023-caste-visualization-overhaul.md) | Caste Visualization Overhaul | Accepted | 2026-02-07 |
| [0024](0024-automated-diagnostics.md) | Automated Diagnostics (Pathology) | Accepted | 2026-02-08 |
| [0025](0025-comprehensive-pathology-table.md) | Comprehensive Algorithmic Pathology | Accepted | 2026-02-09 |
| [0026](0026-clinical-metrology.md) | Clinical Metrology and Calibration | Accepted | 2026-02-10 |
| [0027](0027-calibration-report-consistency.md) | Calibration Report ↔ Clinical Payload | Accepted | 2026-02-11 |
| [0028](0028-refined-clinical-diagnostics-v0.9.1.md) | Refined Clinical Diagnostics (v0.9.1) | Accepted | 2026-02-17 |
| [0029](0029-headway-nomenclature.md) | Headway Nomenclature | Partially Superseded by 0037 | 2026-02-18 |
| [0030](0030-clinical-instrument-architecture.md) | Clinical Instrument Architecture | Partially Superseded by 0039 | 2026-02-20 |
| [0031](0031-numerical-integrity-verification-tier.md) | Numerical Integrity Hub (v0.9+) | Accepted | 2026-02-21 |
| [0032](0032-enhanced-persistence-metadata-and-diagnostic-simplification.md) | Diagnostic Simplification | Accepted | 2026-02-22 |
| [0033](0033-flexible-reference-management-for-longitudinal-auditing.md) | Flexible Reference Management | Accepted | 2026-02-23 |
| [0034](0034-unified-metric-analysis-api.md) | Unified Metric Analysis API | Accepted | 2026-02-24 |
| [0035](0035-triple-mode-hypervolume-reporting.md) | Triple-Mode Hypervolume Reporting | Partially Superseded by 0047 (`rel`) and 0058 (`abs`) | 2026-02-25 |
| [0036](0036-half-normal-closeness-v0.12.0.md) | Half-Normal Projection (v0.12.0) | Partially Superseded by 0037 | 2026-02-25 |
| [0037](0037-plausible-q1-and-search-drive-headway-v0.13.1.md) | Plausible Q1 and Search-Drive Headway (v0.13.1) | Partially Superseded by 0049 | 2026-03-08 |
| [0038](0038-runtime-output-and-sidecar-hygiene-v0.13.3.md) | Runtime Output Control and Sidecar Hygiene (v0.13.3) | Accepted | 2026-03-11 |
| [0039](0039-canonical-api-and-compare-semantics-v0.14.0.md) | Canonical API and Compare Semantics (v0.14.0) | Partially Superseded by 0048 | 2026-03-11 |
| [0040](0040-canonical-view-inputs-and-no-compatibility-shims.md) | Canonical View Inputs and No Compatibility Shims | Accepted | 2026-03-13 |
| [0041](0041-layer-ontology-for-structural-analysis.md) | Layer Ontology for Structural Analysis | Accepted | 2026-03-13 |
| [0042](0042-hypervolume-bbox-diagnostics.md) | Diagnostic-Only Hypervolume Bounding Boxes | Accepted | 2026-08-12 |
| [0043](0043-hypervolume-engine-selection.md) | Hypervolume Engine Selection and Reproducible Monte Carlo | Accepted | 2026-08-13 |
| [0044](0044-ordinal-hypervolume.md) | Ordinal Hypervolume | Accepted | 2026-08-19 |
| [0045](0045-component-seed-allocation.md) | Deterministic Component Seed Allocation | Accepted | 2026-08-21 |
| [0046](0046-relative-ordinal-hypervolume.md) | Relative Scaling for Ordinal Hypervolume | Accepted | 2026-08-21 |
| [0047](0047-conventional-hypervolume-relative-scaling.md) | Conventional Hypervolume Relative Scaling | Accepted | 2026-08-27 |
| [0048](0048-public-clinical-metrics-and-audit-context-parity.md) | Public Clinical Metrics and Audit Context Parity | Accepted | 2026-08-27 |
| [0049](0049-longitudinal-headway-definition.md) | Longitudinal Headway Definition | Accepted | 2026-08-27 |

| [0050](0050-certified-environment-verification-and-provenance.md) | Certified Environment Verification and Experiment Provenance | Accepted | 2026-10-15 |
| [0051](0051-fair-conformance-audit.md) | FAIR Conformance Audit, Scientific Oracles, and Validation Log | Proposed (investigation in progress) | 2026-10-21 |
| [0052](0052-dpf2-dtlz7-exact-pareto-regions.md) | Exact Pareto-region sampling for DPF2 and DTLZ7 (issue #78) | Proposed | 2026-11-11 |
| [0053](0053-dpf5-mixed-dimensional-reference-measure.md) | DPF5: mixed-dimensional ideal reference and negative ECDF law (scientific research, issue #81) | Proposed / noncertified | 2026-11-12 |
| [0054](0054-dominance-constrained-one-sided-gt-blur.md) | Dominated-side one-sided GT half-normal blur (issue #83) | Proposed | 2026-11-12 |
| [0055](0055-dpf5-closeness-reference-protocol.md) | DPF5 Closeness ideal and unfavorable reference laws; multi-seed validation contract (issue #81) | Proposed / noncertified | 2026-11-12 |

| [0056](0056-versioned-certification-and-performance-reference-lifecycle.md) | Versioned Numerical and Performance Certification | Accepted (read final addenda) | 2026-11-18 |
| [0057](0057-proposed-purpose-specific-gt-budgets.md) | On-Demand, Geometry-Aware Ground Truth | Accepted; MOP-specific GT implemented; automatic budget policy pending | 2026-11-19 |
| [0058](0058-scalar-absolute-hv-certified-reference.md) | Scalar Absolute HV: Certified or Sampled Reference | Accepted and implemented (#89, PR #170) | 2026-11-23 |
| [0059](0059-gt-quality-measurement-and-acceptance.md) | Measurable GT Quality and Independent Acceptance | Accepted; initial API/report implemented, further MOP oracles pending | 2026-11-23 |

### Quick component lookup (check lineage and final dated addenda)

This list is a navigation shortcut, **not a replacement** for reading each
applicable ADR and its later supersessions. Accepted-in-principle ADRs govern direction; unimplemented details are not runtime contracts. Proposed ADRs are not binding.

| Work area | Start with accepted decisions | Current pending research |
|---|---|---|
| MOP formulas, Pareto set/front | 0006, 0016 (supersedes 0015), 0017, **0057** | #119 and #65 completed; 0059 proposed GT-quality API and certification matrix |
| GT calibration and point budgets | 0017, **0018 (current 2,000/10,000 defaults)**, **0057** | Automatic adaptive budget remains unimplemented; 0059 proposed diagnostics; #54 empirical GT independent |
| View and topology | 0014, 0040; follow API lineage 0039 → 0048 | #119 only for GT source/visual decimation considerations |
| Save/load, provenance | 0011, 0032, 0050 | proposed GT archive integration, undecided |
| Conventional Hypervolume | 0019 → 0035 → **0047** (`rel`) and **0058** (`abs`), plus 0020, 0042–0043 | [GT specification](../gt.md#92-current-scalar-absolute-hv-reference-specification-adr-0058); [#89 research memorial](../research/hv-issue89-research-memorial.md) |
| Numerical and stochastic certification | **0050 and 0056 (read through final addenda)** | #63 completed; source-bound certification still required for future changes |
| FAIR historical research | accepted API lineage where still applicable | proposed 0051–0055; FAIR not a core GT dependency |

## Decision Lineage

The status column records whether a historical ADR remains wholly normative or has been superseded in a specific part. Superseded ADRs are retained as historical records rather than rewritten to match later behavior.

### Ground Truth

- **Current MOP-owned GT source and calibration requirements:**
  [ADR 0057](0057-proposed-purpose-specific-gt-budgets.md),
  implemented analytical samplers and the
  [GT specification](../gt.md).
  #119 and #65 are closed/completed. The automatic, geometry-aware
  calibration **budget** remains an accepted direction without a
  deployed universal selection algorithm (ADR 0018 defaults remain).
- **GT validity, coverage and regularity as measurable API/test data:**
  [ADR 0059](0059-gt-quality-measurement-and-acceptance.md) is
  **accepted, with initial implementation**. The opt-in inspection API and report are implemented; remaining scientific validators and acceptance thresholds are not yet certified.
  Any future accepted GT-quality contract must use independent
  mathematical validators and distinguish error, finite-sample
  adequacy and unverified claims.
- **Historical evidence** belongs under
  [research/gt-issue119](../research/gt-issue119/);
  its old statuses are snapshots and do not override current
  normative ADRs or production source.

### Hypervolume

- Nomenclature and reporting: [0019](0019-tripartite-hypervolume-metrics-nomenclature.md) → [0035](0035-triple-mode-hypervolume-reporting.md) → [0047](0047-conventional-hypervolume-relative-scaling.md) for `rel`; [0058](0058-scalar-absolute-hv-certified-reference.md) governs `abs` denominator provenance and certification.
- Conventional geometry and computation: [0020](0020-standardized-reference-point-strategy-for-normalization.md) → [0042](0042-hypervolume-bbox-diagnostics.md) / [0043](0043-hypervolume-engine-selection.md), with current relative scaling in [0047](0047-conventional-hypervolume-relative-scaling.md).
- Absolute-GT reference policy: [0058](0058-scalar-absolute-hv-certified-reference.md) is normative; [the GT specification](../gt.md) defines the current operational behavior; [the issue #89 research memorial](../research/hv-issue89-research-memorial.md) preserves historical experiments and rejected alternatives.
- Ordinal branch: [0044](0044-ordinal-hypervolume.md) → [0046](0046-relative-ordinal-hypervolume.md).

### Environment, numerical stability and regression validation

- [0050](0050-certified-environment-verification-and-provenance.md) records the
  numerical/runtime certification contract and the 2026-11-17
  maintainer-authorized separation between deterministic numerical replay
  and optional statistical performance assessment.
- Its 2026-11-18 addendum documents the completed verification work
  (PRs #88, #90, #91, #99, #100), the confirmed NumPy/OpenBLAS profile
  that reproduces the historical NSGA-III×DPF1 result (issue #92, PR #101),
  and the distinction between passing CI and an approved release manifest.
- The current operational classification, commands and reference safeguards
  are in the [testing manual](../test.md); historical decisions remain
  preserved in their original ADR sections.
- **Implemented versioned certification architecture:**
  [ADR 0056](0056-versioned-certification-and-performance-reference-lifecycle.md)
  governs three independent commands (`--certify-numeric`,
  `--record-performance-baseline`, `--certify-performance`), a compact
  immutable numerical certificate, and the approved 0.18.0 performance
  reference with six prespecified MOP–MOEA cases and 64 seeds per case.
  A future candidate must pass its own numerical certification before
  statistical performance certification; the reference is never
  automatically regenerated. See the [testing tutorial](../test.md)
  for the implemented CLI and the latest operational rules.

### Clinical diagnostics

- Metrology and physical/quality layers: [0026](0026-clinical-metrology.md) → [0028](0028-refined-clinical-diagnostics-v0.9.1.md) → [0029](0029-headway-nomenclature.md) / [0036](0036-half-normal-closeness-v0.12.0.md) → [0037](0037-plausible-q1-and-search-drive-headway-v0.13.1.md) → [0049](0049-longitudinal-headway-definition.md) for the current HEADWAY definition.
- FAIR implementation and pending calibration science: [0051](0051-fair-conformance-audit.md) records the completed issue #75 implementation audit; [0052](0052-dpf2-dtlz7-exact-pareto-regions.md) records source PF corrections; [0053](0053-dpf5-mixed-dimensional-reference-measure.md) records the DPF5 probability-measure and negative-ECDF experiments, **proposed only**, with scientific calibration release in [issue #81](https://github.com/monacofj/moeabench/issues/81).
- Public API and result/view contract: [0030](0030-clinical-instrument-architecture.md) → [0039](0039-canonical-api-and-compare-semantics-v0.14.0.md) → [0040](0040-canonical-view-inputs-and-no-compatibility-shims.md) → [0048](0048-public-clinical-metrics-and-audit-context-parity.md).

---
*For more information on the ADR process, see [Architecture Decision Records](https://adr.github.io/).*
