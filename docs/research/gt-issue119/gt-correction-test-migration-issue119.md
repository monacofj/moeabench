<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF3/DPF5 Correction: Pre-existing Research Assertions and Numerical Baseline Migration

**Review document for Draft PR #140, issue #119, umbrella #122.** This inventory records incompatibilities revealed by the mathematically corrected implementations; **no existing regression test, numerical fixture, calibration sidecar, or baseline has been changed**. It separates proof of a source correction from the governance decision required for a versioned historical-reference migration.

## 1. Why the previously passing assertions can now fail

Research PR #139 initially introduced **negative-diagnostic assertions** that deliberately required the then-current DPF3/DPF5 source to exhibit known defects. Examples included demanding a nonzero difference from published DPF5 equations or demanding that a DPF5 sample request for 128 points return only 121 decisions. These were valuable *before* a correction, to prevent the investigation from claiming an absent defect.

After the evaluator and MOP-specific samplers are corrected in PR #140, those negative-diagnostic assertions must not remain as the **scientific correctness criteria for the corrected runtime**. A passing corrected implementation should instead reproduce the published formulas and global Pareto geometry, while the **historical investigation** remains reproducible using the frozen pre-correction source revision.

This is a **test contract transition**, not permission to weaken a regression test until it passes.

## 2. Exact pre-existing failures observed after evaluator and initial sampler corrections

The complete unit suite run on correction-branch commit \`2e535d864f0447758a23b971e3852a5e739da039\` reported the following **15 failures in 8 pre-existing research test files**. Each asserts a property of the previously defective implementation that a corrected source should no longer exhibit.

| Existing test file | Failed cases | Former asserted behavior | Scientific successor assertion |
|---|---:|---|---|
| \`tests/unit/test_gt_dpf5_allocation_issue119.py\` | 1 | Old M3/D2 effective P/D allocation (and 121 decisions for requested 128) | Explicit-budget sample count and separately declared geometrical P/D allocation |
| \`tests/unit/test_gt_dpf5_finite_sampler_issue119.py\` | 1 | M3/D2 128-request square grid returned 121 rows with objective duplicates | Exact numerical cardinality and branchwise coverage for corrected sampling |
| \`tests/unit/test_gt_dpf5_fixed_impact_issue119.py\` | 3 | Current evaluator **disagrees** with published DPF5 objectives | Current evaluator agrees with independent published equations on identical fixed decisions |
| \`tests/unit/test_gt_dpf5_general_d_issue119.py\` | 2 | Higher-M P and D=4 gamma **disagree** with the publication | Correct P and D match published equations across dimensions |
| \`tests/unit/test_gt_dpf5_offpareto_issue119.py\` | 3 | Off-PS P branch **differs** from the published spherical invariant | Radial \`(1+g)\` model and spherical invariant also hold for \`g>0\` |
| \`tests/unit/test_gt_dpf_branch_diagnostics_issue119.py\` | 1 | Nonsquare M3/D2 request shrinks from 48 to 36 | Request returns stated/effective objective count with preserved branches |
| \`tests/unit/test_gt_dpf_equation_probes_issue119.py\` | 2 | DPF3 thresholds are **unsorted**; DPF5 equality belongs to P | Sorted DPF3 thresholds and DPF5 equality belongs to D |
| \`tests/unit/test_gt_dpf_numerics_kernel_issue119.py\` | 2 | DPF3 first objective collapses under direct \`1-cos\`; DPF5 P norm is defective | Stable positive DPF3 complement and correct DPF5 unit-sphere invariant |

The new, separately named tests in PR #140 already check the **positive mathematical and geometry-specific assertions** independently. None replaces or modifies the eight historical files listed above.

## 3. Distinct numerical-stability reference incompatibility

The **pre-existing** \`tests/test_stability_basic.py::test_gt_reference_reproducibility[DPF5]\` compares a DPF5 reference result to historical thresholds in \`tests/stability_reference_gt.json\`. Under the corrected DPF5 **evaluator but still using the old sampler**, a previous run observed \`backward_max=0.0421024913123726\` against the frozen historical bound \`0.03429322324694801\`. The rest of that 41-case basic stability suite passed in that run.

The geometry-specific sampler subsequently changed, so those **numbers are not automatically the expected outcome of the final correction branch**. The final-branch stability measurement must be taken from its own pinned-environment CI.

This is a scientific reference-version transition: old behavior can remain historically reproducible, but it should not be treated as the canonical published-model GT. The correct path is to **preserve the old baseline**, independently validate a **new versioned reference**, document old-versus-new indicator effects, and adopt a carefully reviewed **successor** test contract. Do **not** simply increase \`backward_max\`, re-record the JSON in place, or relax the test to suppress the signal.

## 4. Approval boundary and recommended order

1. Independently validate all new DPF3/DPF5 equations and finite sampler properties, including the **conditional complete-angular-grid theorem**, numerical cardinality, branch extrema, time/cost, and reproducibility; distinguish exact-real covering from the float64 rounding envelope.
2. Record **the exact final-branch failing cases**, compare them against the inventory above and identify any genuinely new mathematical or numerical defect that is not explained by the correction.
3. Agree a clear research-history preservation mechanism: either freeze the negative-diagnostic scripts/tests against their **original source revision**, or archive the dated negative observations as immutable reports and replace the live contract with positive mathematical checks. These are different approaches; choose explicitly.
4. Develop a **new versioned numerical reference** for the corrected model under the pinned certified environment, with identity, formula version, geometry, sample budget, and measured numerical sensitivity.
5. **Warn and obtain explicit maintainer authorization before editing or replacing any existing test, frozen fixture, or numerical baseline.** This authorization is separate from the maintainer's approval of the MOP-specific correction work in the dated ADR 0057 addendum.
6. Rerun unit, complete numerical stability, and scientific reference workflows on the same revision, then certify the successor reference and request code review/merge.

The current Draft PR #140 does not assert that historically frozen tests already agree with the corrected mathematical model. No other umbrella #122 workstream is implicitly combined with this migration.
