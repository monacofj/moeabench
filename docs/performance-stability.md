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


## Reference artifact v2 — three-way statistical decision (issue #98)

The original `moeabench_performance_reference_v1` interface remains available
for compatibility, but its `PASSED` status merely means that a particular
regression rule did not fire. It is **not** a noninferiority result.

The opt-in `moeabench_performance_reference_v2` schema is an additive,
more conservative protocol for externally approved reference artifacts.
It is **not** a mandatory release gate in the FAIR-free 1.0.0 preparation:
no previous-release numerical performance baseline has yet been independently
approved and no baseline values are distributed. The explicit issue #98
release decision is therefore to **defer statistical performance certification**
until provenance, paired pilot data and a case shortlist receive review.
The same command and system API accept either version, but report the schema.

### V2 reference requirements

A v2 JSON document contains top-level `schema`, `provenance`,
`family_alpha`, `target_power` and a nonempty `cases` list.
`provenance` requires `source_revision`, `reference_origin`,
`reviewed_by`, `environment_fingerprint`, `measurement_protocol` and
`pilot_origin`: these are audit metadata, **not** cryptographic proof
of independent review. Cases retain v1's explicit `mop`/`moea` constructor
specifications, minimized `metric`, fixed `reference_front`, distinct
`seeds`, `baseline_values` indexed by seed, and preregistered
`max_absolute_degradation`/`max_relative_degradation`.
Every case additionally supplies `pilot_standard_deviation` (of
**paired** quality differences, computed externally) and a nonempty
`sample_size_rationale`. The overall reference supplies the familywise
significance bound `family_alpha` and desired `target_power`
(0.80–0.99). Per-case `alpha` is rejected in v2.

Run counts are fixed by the provided seed lists; there is no automatic
30-run matrix, early stop, adaptive repetition or baseline renewal.
The runner requires at least eight seeds and validates that the number
meets an **a priori** normal-difference, noncentral Student-t power calculation.
Its planned effect is distance `δ` from the practical degradation boundary:
under unchanged mean quality, `μ_delta=0`, and under material deterioration,
`μ_delta=2δ`. The power is the chance to classify those respective
situations, given the externally obtained pilot paired standard deviation.
If the required runs exceed 5,000, the reference is refused; the experiment
must be redesigned rather than implicitly expanded. A pilot standard
deviation of zero is allowed for exactly deterministic comparisons, with the
same minimum eight paired seeds.

### Familywise evidence and outcomes

For each case, define `d_i = candidate(seed_i) − baseline(seed_i)`,
where a **larger** minimized score is worse, and preregister the positive
margin

`δ = max(max_absolute_degradation, baseline_mean × max_relative_degradation)`.

With `m` cases in the single reference, v2 uses a simultaneous
two-sided Student-t confidence interval for `mean(d_i)` with tail level
`family_alpha/(2m)`. This conservative Bonferroni adjustment bounds
the familywise probability of at least one false directional conclusion
under the Student-t model. No reclassification is allowed after inspecting
outcomes.

- `PASSED`: the **upper** simultaneous confidence bound is strictly below
  `δ`, providing evidence against meaningful mean performance degradation.
- `FAILED`: the **lower** bound is strictly above `δ`, providing evidence
  of material mean performance degradation.
- `INCONCLUSIVE`: the bounds overlap or touch `δ`; the available
  replications do not justify either claim.

The overall result is `FAILED` if any case fails; otherwise
`INCONCLUSIVE` if any case is inconclusive; otherwise `PASSED`.
The CLI exits **0** only on `PASSED`, **1** on `FAILED` or
`INCONCLUSIVE`, and **2** on an invalid or unavailable reference.
Numerical stability is never started automatically.

**Scientific limits:** these statements concern *mean paired score change*,
not equality of complete distributions or behavior in all MOP/MOEA settings.
The paired t interval assumes the seed-wise differences are independent and
approximately normal, with comparable environments and an honest external
pilot variance estimate. Pairing by seed may be useful, but variance reduction
is not guaranteed merely by sharing seed integers; it must be checked in
the pilot. Skewed/heavy-tailed outcomes, multiplicity across separate
reference files and exploratory case selection require a separate method
review. CI's noninference state is explicit rather than interpreted as a pass.

### Acceptance and cost before using a v2 reference

The maintainer/reviewer must select representative cases, verify the
reference revision and numerical environment, examine seed-wise paired
pilot variability, set practical margins based on scientific consequences,
check the required run count and approve the reference artifact **before**
running comparisons. A fixed budget may motivate a smaller case shortlist
or remaining with optional diagnostics; the test must not silently reduce
sample sizes to obtain a favorable result.

The test suite includes synthetic deterministic **unit tests of the
decision procedure**, not certified performance observations. Producing an
official baseline from a historical release is a separate reviewed artifact
and was intentionally **not performed** in PR #100 or any subsequent
verification work. No official prior-release performance baseline is
bundled as of this documentation update.
