<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Explicit statistical performance stability

Performance stability and numerical reproducibility answer different questions.
Numerical stability verifies fixed inputs, deterministic seeded trajectories and
saved numerical references. A single execution is appropriate when checking a
fixed seed against its frozen output. Statistical performance stability asks
whether the *distribution* of solver quality has deteriorated relative to a
previously reviewed result. It requires multiple runs.

The performance test is **optional and separately requested**. It does not run
as part of `--basic`, `--certify`, `mb.system.verify_stability()`, or
automatically after a numerical failure.

```bash
python test.py --numeric-stability --basic
python test.py --performance-stability --performance-reference /path/to/reviewed.json
```

There is **no bundled statistical baseline** for the obsolete deep test:
`tests/baselines_v0.8.0.csv` is missing. `--stability --deep` and the
all-scopes `--deep` invocation fail explicitly rather than treating the
historical skipped test as a successful certification.

## Reference artifact: schema v1

A reviewable external JSON reference must have `schema` equal to
`moeabench_performance_reference_v1`, a nonempty `cases` array, and a
`provenance` object with these nonempty strings: `source_revision`,
`reference_origin`, and `reviewed_by`.

Every case requires:

| Field | Interpretation |
|---|---|
| `id` | Unique human-readable case ID |
| `mop` | Object with a public class `name` and optional constructor `kwargs` |
| `moea` | Object with a public class `name` and constructor `kwargs` (without `seed`) |
| `metric` | One minimized quality indicator: `igd`, `igdplus`, `gd`, `gdplus`, or `emd` |
| `reference_front` | Fixed 2D objective matrix, independent of the current solver output |
| `seeds` | Explicit, distinct integer seeds; at least three |
| `baseline_values` | Prior version's finite quality measurement for each seed, in identical order |
| `max_relative_degradation` | Preregistered nonnegative practical degradation threshold as a fraction |
| `max_absolute_degradation` | Preregistered nonnegative absolute degradation threshold |
| `alpha` | Optional one-sided significance level, default 0.05 |

**No example baseline numbers are bundled**: publishing synthetic targets as a
reference would create the same self-certification problem that the old deep
suite had. A reviewer must establish baseline measurements from an identified
prior release or independent implementation, fix the seed schedule and
reference front, record the exact dependency environment and code revision
in `reference_origin` or an accompanying audit, and approve the artifact
before enabling it in CI. This schema validates that review metadata was
supplied; it cannot itself prove that an external review occurred.

For each case the runner performs exactly the listed number of solver runs
(no global 30-run default). It computes paired differences
`current_metric(seed) - baseline_metric(seed)` and reports a one-sided paired
t-test (increase is worse), plus the practical degradation threshold:

`max(max_absolute_degradation, max_relative_degradation * abs(baseline_mean))`

A case is marked `FAILED` only if the mean increase exceeds that threshold
and the one-sided p-value is below `alpha`. Other outcomes are `PASSED` for
this **specific regression rule**, rather than proof that performance has
not changed. This protocol tests worsening mean quality, not general
distribution equality, and the small-sample power and multiple-comparison
policy must be considered when selecting cases. An inconclusive result
should not be interpreted as evidence of performance equivalence.

Only the explicitly listed algorithms and problems are run. A missing or
invalid reference aborts before optimization. Neither this runner nor
`--certify` regenerates, changes or overwrites a scientific baseline.

## Reference preparation and decision gate

The previous `0.16.1` release and independently computed reference outputs
can be assessed as candidate sources, but raw target values are not yet
approved for the FAIR-free 1.0.0 preparation. Before making performance
regression a release gate, maintainers must review provenance, selected
quality metrics, paired seed design, practical effect thresholds and
statistical power. Reference renewal is a separate audited decision;
failed tests must not be made green by changing the target values.
