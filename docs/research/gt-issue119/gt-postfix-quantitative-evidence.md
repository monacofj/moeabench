<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Post-fix quantitative GT checkpoint — issue #119

**Status:** reproducible finite-sample research, not release certification.
**Source:** PR #152; initial all-MOP evidence from
`a7d837314803f2f82137290d508d0654a0c8db88`,
independent finite-probe extension from
`1a796041490cb518bcfe266bbe3fcf90eec29b25`.
GitHub Actions run artifacts contain the machine-readable JSON.
Measurements are platform-dependent elapsed wall time, not
comparably certified per-MOP performance.

## Cardinality and deterministic replay

The first GitHub Actions execution completed all 72 combinations:
twelve MOPs, M=3 and M=5, and budgets 128/512 plus currently
effective calibration budgets (2,000 for DTLZ, 10,000 for DPF).
All 72 returned their requested point count, had no duplicate
objective rows, and replayed `ps()` identically. Independent
analytical residual checks, where available for DTLZ1–4/8/9,
were below `1e-10`; unsupported residual checks are explicitly
`null`, not interpreted as proof of correctness. The report
also records objective hashes, sampled-region counts and
component-wise nearest-neighbor spacing for DTLZ7, DTLZ8
and DPF2. Positive-report contract tests: 17 passed.

## Independent finite-probe distance evidence

A second execution used independent, deterministic decision-space
constructs for DTLZ1–4/7/9 and DPF1/2 (not their production
`ps()` routines), with 512 probe points for three independent
seeds 11941–11943. For every one of 16 problem/dimension
combinations, the **worst across the three probes** finite
maximum Euclidean gap was lower at budget 512 than 128.
Representative values:

| Problem | M | Maximum gap at n=128 | Maximum gap at n=512 |
|---|---:|---:|---:|
| DTLZ1 | 5 | 0.1853 | 0.1317 |
| DTLZ2 | 5 | 0.3990 | 0.2894 |
| DTLZ7 | 5 | 0.6437 | 0.5015 |
| DTLZ9 | 5 | 0.0095 | 0.0024 |
| DPF2 | 5 | 0.8858 | 0.3531 |

These are **not dimensionless** measures and should not be
compared across MOPs without accounting for their scales and
objective geometry. DTLZ7 and DPF2 component results are retained
separately, rather than treating a pooled nearest-neighbor CV
as a valid disconnected-front regularity statistic.

A strict limitation: the independent-probe experiment does not
cover DTLZ8 or DPF3/4/5. Those have dedicated mathematical
tests and historical research evidence, but cannot legitimately
inherit this finite-probe result without dedicated reference
generators. No continuous covering radius is certified.

## Indicator-reference sensitivity

The third experiment holds synthetic **analytically correct**
approximation populations fixed and varies GT budgets. It computes
raw Euclidean IGD and positive-part IGD+, without the public
indicator's reference normalization. All **48 fixed-population
comparisons** (eight MOPs, M=3/5, three seeds) showed a nonzero
IGD+ change between n=128 and the current full calibration budget,
with **36 new research tests passing**. This is a measurable
reference-resolution effect, not necessarily deterioration or
an algorithmic regression.

For example, for the *same* synthetic population with M=5,
seed=11941, raw IGD+ at 128 points versus the full budget was:

| Problem | GT n=128 | GT full |
|---|---:|---:|
| DTLZ1 | 0.034999 | 0.033085 |
| DTLZ2 | 0.210979 | 0.219204 |
| DTLZ4 | 0.519218 | 0.482486 |
| DTLZ7 | 0.128167 | 0.127164 |
| DTLZ9 | 0.007244 | 0.006793 |
| DPF2 | 0.045382 | 0.042228 |

The direction can change with the benchmark: comparing
indicator scores across versions requires *the same explicitly
identified GT*. The probe experiment is **not** a stochastic MOEA
performance-certification surrogate and does not demonstrate
MOEA rank inversions. GT-dependent HV and fixed-stochastic-MOEA-
front sensitivity remain separate tasks; neither is considered
complete here.

## Reproduction

```sh
python -m pytest -q tests/unit/test_gt_postfix_quantitative_issue119.py
python -m scripts.audit.gt_postfix_quantitative_issue119 --objectives 3 5 --budgets 128 512 --large-budget > gt-postfix-quantitative.json
python -m scripts.audit.gt_independent_probes_issue119 --objectives 3 5 --budgets 128 512 --seeds 11941 11942 11943 --probes 512 > gt-postfix-finite-probes.json
python -m scripts.audit.gt_reference_sensitivity_issue119 --objectives 3 5 --budgets 128 512 2000 --seeds 11941 11942 11943 --population 64 > gt-reference-sensitivity.json
```

These commands do not write calibration caches, frozen GT files,
performance references or versioned certificates. No accepted ADR
or existing regression test is modified by this evidence. Final
source-revision/environment-bound certification remains governed
by ADR 0056 and its executable workflow, and the completion of
original DTLZ5/6 fronts remains a separate issue #65.
