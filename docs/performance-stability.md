<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Scientific performance certification and historical baselines

This is the current experimental specification for MoeaBench release
certification. See [ADR 0056](adr/0056-versioned-certification-and-performance-reference-lifecycle.md)
and [the testing manual](test.md). A **performance reference** is a frozen
sample of results measured with a previous certified release; a
**performance certificate** is the verdict for a later candidate.
The numerical certificate is independent of both.

## Operations

```bash
# Certify the current developer numerical environment; never changes a
# historical solver front, mathematical oracle or performance baseline.
python test.py --certify-numeric

# One-time, explicitly authorized recording of the inaugural 0.18.0
# historical reference. The output is CANDIDATE until independently reviewed.
python test.py --record-performance-baseline \
  --output moeabench/resources/certifications/0.18.0/performance-reference.json

# On a later candidate version, compare with the approved historical
# reference. The output is CERTIFIED or NOT_CERTIFIED, and is immutable.
python test.py --certify-performance \
  --reference moeabench/resources/certifications/0.18.0/performance-reference.json \
  --output moeabench/resources/certifications/1.0.0/performance-vs-0.18.0.json
```

The three operations are independent. **No command automatically renews
or substitutes an approved reference.** Creating a candidate reference
does not certify its performance; no earlier release exists for that
comparison. An independently reviewed candidate is explicitly marked
`APPROVED` and committed. Failed validation leaves the reference
unchanged. The new user-facing spelling has no permanent legacy alias
requirement; regression-contract migration is tracked in issue #108.

## Fixed family of six experiments

| Case | MOP | MOEA |
|---|---|---|
| dtlz2-nsga2 | DTLZ2(M=3) | NSGA2 |
| dtlz2-nsga3 | DTLZ2(M=3) | NSGA3 |
| dtlz2-moead | DTLZ2(M=3) | MOEAD |
| dtlz7-nsga2 | DTLZ7(M=3) | NSGA2 |
| dtlz7-nsga3 | DTLZ7(M=3) | NSGA3 |
| dpf1-nsga3 | DPF1(M=3,D=2) | NSGA3 |

Each case uses population 48, 60 generations, and the 64 prespecified
seeds `424242 + 104729*i`, with `i=0,…,63`. The first reference
costs 384 algorithm runs. Exactly one 512-point objective-space
reference is frozen per MOP and bundled with its SHA-256; later
candidate versions must use those **same coordinates**, not recalculate
them from potentially changed MOP code. This is a performance
measurement input, not a new numerical-regression baseline.

The decisive indicator is IGD+ calculated by the existing public
reference-normalized implementation (smaller is better). GD+ and EMD
are stored as diagnostics and do not create extra significance tests.
Each historical observation retains the seed, objective-front size,
three scores, configurations and software/runtime provenance. Complete
solver populations are not duplicated for every certified version.

## Statistical decision

We evaluate a future candidate using the **same seed for each paired
reference and candidate score**. The first eight seeds form a pilot,
used exclusively to estimate the standard deviation of paired
differences; these measurements are excluded from the final inference.
The remaining 56 seeds form a prespecified confirmation pool.

For case `j`, define the mean degradation
`d_i = IGD+_candidate(seed_i) - IGD+_reference(seed_i)`.
The maximum materiality tolerance is fixed in advance to

```text
delta_j = max(1e-12, 0.001 * historical_mean_IGDplus_j)
```

The 0.1% relative allowance is an explicit conservative release
engineering tolerance, **not** evidence that a 10% change is
acceptable, nor a universal precision floor for unrelated scientific
metrics. The `1e-12` term addresses numerical zero.

Statistical protection is **familywise 1%** over the six IGD+ cases,
using two-sided Student-t intervals with Bonferroni correction;
each tail gets `0.01/(2*6)`. A successful numerical certification
and fixed run seeds are prerequisites, not substitutes for this
statistical check.

The separate eight-pair pilot supports a noncentral Student-t
sample-size calculation targeting **95% probability of establishing
noninferiority when the true mean change is zero**. The selected number
of confirmatory pairs is between eight and 56. If 56 pairs cannot
meet the planning target, the outcome is `NOT_CERTIFIED` with reason
`INSUFFICIENT_POWER`. A pilot estimate is uncertain, so the 95%
figure is a model-based planning target, not a guarantee.

For each case with adequate planned sample size, compute the
simultaneous confidence interval `[L,U]` on the mean paired change:

| Statistical evidence | Case decision |
|---|---|
| `U < delta` | `CERTIFIED` (evidence of acceptable performance) |
| `L > delta` | `NOT_CERTIFIED` (`REGRESSION_DETECTED`) |
| Otherwise | `NOT_CERTIFIED` (`INSUFFICIENT_EVIDENCE`) |

A release receives overall **`CERTIFIED` only if all six cases
pass**. Any other outcome is **`NOT_CERTIFIED`**, with the exact
cause recorded for corrective action. Statistical uncertainty
never silently permits release acceptance. An inconclusive
scientific interval is a **negative operational gate**, not a third
publication decision.

## Historical records and numerical references

The numerical certification archives a small per-version JSON
containing the version/environment snapshot, source revision,
the **hashes** of existing frozen regression fixtures, and the
certification evidence. There is **no copied population/front archive
per version**. The canonical numerical regression tests and their
original fixtures are preserved.

The versioned directory `moeabench/resources/certifications/<version>/`
contains the compact numerical certificate and, where approved,
the historical performance reference or a performance comparison
certificate. `mb.system.certification_history()` lists the records.
`mb.system.matching_certified_environments()` identifies identical
*dependency-version fingerprints*; it does not claim numerical
equivalence without an actual numerical comparison.

For the first historical reference, the algorithm source revision is
the previously numerically certified 0.18.0 commit. Reference
creation and its approval are separately auditable Git events.
A future result that differs from the current version cannot be
declared numerically equal to an older version based on hashes or
version numbers alone.

## When a test fails

A numerical replay mismatch blocks `--certify-numeric`. Preserve the
original expected results, inspect the runtime preflight, oracle,
component seeds and governing ADRs; **never refresh a historical
regression fixture just to make CI pass**.

A performance failure means either a statistically established
material mean regression or insufficient evidence/power to
demonstrate acceptable performance. Both block certification;
the detailed reason determines whether to debug an actual change
or investigate the study's precision and variability. A
`NOT_CERTIFIED` result must not regenerate or overwrite its
reference. A genuinely new scientific baseline requires a
separate explicitly authorized and reviewed decision.

The historical implementation and provisional statistical interfaces
remain documented in Git history; this document specifies the
approved unified protocol for issue #108.
