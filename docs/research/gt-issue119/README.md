<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Ground Truth Scientific Audit: Issue #119

This directory contains the specialized, reproducible scientific evidence produced by the DPF3/DPF5 Ground Truth investigation, [issue #119](https://github.com/monacofj/moeabench/issues/119), [PR #139](https://github.com/monacofj/moeabench/pull/139).

For the **permanent, self-contained scientific reference**, consult [docs/gt.md](../../gt.md). For an annotated inventory of all reports and architectural decisions, consult [docs/index.md](../../index.md).

These reports preserve independent mathematical proofs, numerical experiments, reproducibility artifacts, negative results, dated hypotheses, and migration planning. Early provisional observations are part of the research record, not competing final specifications.

## Mathematical proofs

- `gt-dpf3-global-pareto-proof-issue119.md` — complete published DPF3 PS/PF proof.
- `gt-dpf5-global-pareto-proof-issue119.md` — complete published DPF5 P/D PS/PF proof.
- `gt-dpf3-d2-angular-cover-proof-issue119.md` — DPF3 angular coverage at D=2.
- `gt-dpf3-nd-cover-and-acceptance-issue119.md` — DPF3 general-D covering certificate and precision limits.

## Independent mathematical and numerical audits

- `gt-dpf3-dpf5-audit-issue119.md`
- `gt-dpf-published-formula-triage-issue119.md`
- `gt-dpf-general-d-evidence-issue119.md`
- `gt-dpf-numeric-kernel-evidence-issue119.md`
- `gt-dpf-high-precision-sweep-issue119.md`
- `gt-dpf5-offpareto-issue119.md`

## Finite-sampling and indicator experiments

- `gt-dpf5-finite-sampler-coverage-issue119.md`
- `gt-dpf5-branch-sobol-comparison-issue119.md`
- `gt-dpf5-branch-allocation-issue119.md`
- `gt-dpf5-paired-indicator-impact-issue119.md`

## Migration and quality review

- `gt-dpf-production-impact-inventory-issue119.md`
- `gt-dpf-correction-migration-proposal-issue119.md`
- `gt-dpf-premerge-review-issue119.md`

The independent research scripts are in `scripts/audit/`, with new tests under `tests/unit/`. The reports' relocation changes **no production source, accepted ADR, existing regression test, or historical reference**.
