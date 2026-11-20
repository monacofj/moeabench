<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3/DPF5 production correction: nonmutating impact inventory — issue #119

**Status:** preparation for maintainer review, not approval to modify production, ADRs or historical tests. Reviewed accepted ADR 0006 (native mathematical PS/PF contract), 0018 (historical DPF 10,000-point calibration density), 0057 (MOP-owned explicit calibrate, purpose-specific budget direction but numerical policy pending), and 0056 (versioned, immutable numerical/performance certification).

## Verified scientific defects in present evaluator

- DPF3: unsorted projected chaotic eta for M−D>=2 versus published ordered thresholds; extreme theta=(pi/2)x^100 cancellation in 1−cos(theta) and representability limits. Historical PF post-filter has effective cardinality far below request.
- DPF5: beta/P formula and gamma/D indexing errors depending on M,D. Current math violates published scaled sphere equation `sum_i F_i²=(1+g)²` in documented configurations. Published branch condition `x1<1/3` conflicts with current P-at-equality. M3 grid cardinality and duplicate objective vectors are independent sampling issues.

## Specific affected existing artifacts to inspect (no edits authorized)

1. **Production evaluation and generation:** `moeabench/mops/DPF3.py`, `DPF5.py`, `base_dpf.py`; original PS/PF and public calibrate contract under ADR 0006/0057. Distinguish changing problem evaluation from replacement of objective-space sampling and requested-point semantics.
2. **Frozen numerical/geometry evidence:** `tests/stability_reference_gt.json` includes a DPF5 basic reference; `tests/stability_gt_geometry_reference.json` embeds historical three-objective DPF5 GT coordinates. These must remain version-traceable and are not to be silently regenerated or replaced. Inspect `tests/test_stability_basic.py` before assigning a precise expected numerical difference.
3. **Historical DPF3 regression:** `tests/unit/test_dpf3_emd_regression.py` consumes `DPF3().pf(n_points=250)` for metric identity/translation laws; these assertions may remain valid even if sampled coordinates change. Do not presume all DPF3 tests require edits.
4. **Calibration and archived reference provenance:** the configured calibrated sidecar, stored reference arrays, empirical external GT and any historical basic numerical certifications must be inventoried by configuration (M,D,N,K, source version, geometry) before migration. Existing official GT references do not become correct through recalibration alone.
5. **Metrics and studies:** IGD, IGD+ and GT-relative absolute HV can change even for a fixed MOEA population after reference replacement; MOEA results themselves can also change if evaluation equations change. Hold MOEA random inputs/decision points fixed to isolate evaluator effects, then hold evaluated objective populations fixed to isolate GT effects. Relative HV semantics and separate FAIR research are outside this repair.
6. **Release certification:** ADR 0056 prohibits modifying frozen reference values merely to pass CI; keep original tests/baselines for comparison and version candidate replacements with explicit maintainer approval. New independent scientific tests are permitted; changes to existing regression tests require prior warning/approval.

## Proposed review sequence

A. Publish exact term-by-term published DPF3/DPF5 equation matrix, including nonzero g, all supported dimensions, and strict branch threshold, against independent calculations (research tests already give partial evidence, including D=2/3/4).
B. Complete mathematical Pareto membership/completeness verification, constrained by M,D and mixed-dimensional component support. Separate this from finite GT regularity and sample-size choice.
C. Produce a fixed-decision paired current/proposed evaluator study and fixed-objective-population paired reference/indicator study. Record deltas, counts, provenance and performance; never rewrite frozen references in the study.
D. Submit a precise change plan listing every production contract, accepted ADR and existing regression file that would be altered and **request authorization before any incompatible ADR or regression change**. Keep historical references accessible and correctly identified by version.
E. After approval, implement separately reviewed corrections and certify the final candidate; budget-policy default changes under ADR 0057 are a separate decision.

This research audit does not authorize production or historical numerical changes and PR #139 remains Draft.
