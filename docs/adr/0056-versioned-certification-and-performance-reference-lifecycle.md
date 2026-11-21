<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0056: Versioned Numerical and Performance Certification

## Status

Accepted — approved by the maintainer on 2026-11-18 (issue #108).

**Current normative implementation:** read the dated refinement and
completion records at the **end of this ADR**. The earlier 24-run, 10%
degradation, 5% familywise, and three-way *release verdict* proposals are
superseded; they remain in the file only as architectural decision
history. The operative study is 64 reference seeds, eight independent
pilot pairs, up to 56 confirmatory pairs, 0.1% materiality, familywise
1%, and a binary certification gate.

This decision specifies the **target architecture and frozen experimental
design**. Acceptance of the protocol does **not** imply that the new CLI,
certification archive or performance baseline have already been generated.

## Context

The completed numerical certification of MoeaBench 0.18.0 demonstrates the
unit, integration and deterministic numerical contracts in its approved
execution environment, and writes the dependency version manifest.
Stochastic performance requires a different experiment: the distribution
of quality across *different* run seeds, measured against an independent
historical reference. These operations previously had overlapping
terminology and did not maintain an explicit per-version certificate index.

A requirement for a *prior* standard deviation of **paired changes between
the reference and the candidate** is impossible to satisfy when creating
the very first historical reference: the future candidate does not yet
exist. Fabricating a variance from the identical reference runs would
give a false zero estimate. An a priori, bounded seed budget combined with
a conservative, simultaneous interval and a genuine `INCONCLUSIVE`
outcome avoids this circular dependency.

The maintainer has authorized replacing pre-1.0 provisional CLI and
statistical schemas and updating accepted ADRs. This ADR supersedes the
*provisional* performance-protocol details of ADR 0050 and the historical
sampling prescriptions of ADR 0017 **for release certification only**.
The older ADRs remain intact as historical records. The public
`mop.calibrate()` GT contract, component seeds of ADR 0045, and the
mathematical/numerical regression values remain unchanged.

## Decision: three explicit commands, independent responsibilities

| Command | Reads | Does | Writes |
|---|---|---|---|
| `python test.py --certify-numeric` | Current source, the full basic suite and **unchanged** numerical references | Preflight of required reference environment, unit + integration + numerical stability | Versioned numerical certificate for the candidate plus current manifest |
| `python test.py --record-performance-baseline` | An already numerically certified version and a reviewed, fixed experimental plan | Runs the baseline study, validates its reproducibility and provenance | New **candidate** historical performance reference; explicit approval freezes it |
| `python test.py --certify-performance` | A frozen historical reference from an **earlier** accepted version | Runs the same cases/seeds on the candidate and performs simultaneous statistical comparisons | Versioned performance certificate with `PASSED`, `FAILED` or `INCONCLUSIVE`; never rewrites the reference |

No automatic dependency or chaining connects these operations: numerical
certification does **not** launch or update performance calibration; a
performance failure does **not** regenerate a baseline; numeric failure
does **not** automatically launch a performance investigation.
When a historical reference exists, both certificates are separately
reviewable for a release decision. No successful performance certificate
can be issued on `INCONCLUSIVE`.

For pre-1.0 development, the old user-facing `--certify` and
`--performance-stability` spellings and the v1/v2 user-facing schema
distinction are to be replaced, **without compatibility aliases**.
This is an authorized API migration, not an instruction to change
historical regression assertions. Existing tests must be audited and
the maintainer warned/asked for permission before any historical
regression test is changed.

### What changes at each certification

- A *numeric certificate* identifies the source revision, package and
  dependency versions, OS/CPU profile and effective NumPy/BLAS dispatch,
  the exact suite and pass/fail evidence. It **does not** create or
  replace a GT, solver-front signature or mathematical metric target.
- A *performance baseline* is an immutable, externally reviewed
  experimental sample on a frozen, numerically certified source
  revision. Recording a baseline is an exceptional, explicitly
  authorized operation; its first use cannot claim comparison-based
  performance certification of itself.
- A *performance certificate* records the executed candidate's
  seed-wise results and familywise statistical decisions relative to
  the **identified earlier baseline**, including uncertainty. It never
  makes the candidate the new baseline automatically.

## Initial historical performance reference: 0.18.0

The initial reference source is the certified **0.18.0 candidate**
(commit [0d3e6680](https://github.com/monacofj/moeabench/commit/0d3e66804351e23b93fb483ff25e178e06ef6e12)),
using its approved Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1,
pymoo 0.6.2, DEAP 1.4.4, moocore 0.3.2, pandas 3.0.6 and native
NumPy/BLAS reference profile. **The baseline is not yet generated** by
acceptance of this ADR.

### Scientifically chosen sentinel cases

All objectives are minimized; all MOPs use `M=3`. The selected cases
cover smooth continuous, disconnected and degenerate front geometries,
and two independent underlying algorithm libraries.

| Case ID | MOP constructor | MOEA constructor | Rationale |
|---|---|---|---|
| `dtlz2-nsga2` | `DTLZ2(M=3)` | `NSGA2` | Smooth positive sphere; DEAP engine |
| `dtlz2-nsga3` | `DTLZ2(M=3)` | `NSGA3` | Same geometry; reference-direction algorithm |
| `dtlz2-moead` | `DTLZ2(M=3)` | `MOEAD` | Same geometry; decomposition algorithm |
| `dtlz7-nsga2` | `DTLZ7(M=3)` | `NSGA2` | Disconnected Pareto front in DEAP |
| `dtlz7-nsga3` | `DTLZ7(M=3)` | `NSGA3` | Disconnected front and reference directions |
| `dpf1-nsga3` | `DPF1(M=3, D=2)` | `NSGA3` | Degenerate projected front, historically sensitive |

Every MOEA receives `population=48` and `generations=60`,
with its own run seed as specified below. This is a **release
regression sentinel**, not a high-budget scientific competition among
different MOEAs. The comparison is always *the same algorithm/problem
configuration across versions*, so native differences in evaluation
count across wrappers are reported but never assumed equal.
The base experiment budget is
`6 cases × 24 seeds = 144 solver runs per version` with nominal
`48 × 60 = 2,880` evolutionary evaluations per run (plus the
wrapper-specific initial population and bookkeeping). It is opt-in
and **never** part of routine unit/numerical CI.

### Seeds and fixed geometric reference

All six cases use the **same** list of 24 distinct integer seeds
`seed_i = 424242 + 104729*i`, for `i = 0, …, 23`.
The list, its generation rule and all effective component seeds
must be stored. NSGA-III direction seeds are derived as in ADR 0045;
no shared global RNG stream is used to draw seeds during baseline
creation.

For each MOP, the 0.18.0 source computes one deterministic analytical
or reference Pareto front using `mop.pf(n_points=512)`. Before
recording quality, the points, ordering, dtype, objective dimension,
derivation and SHA-256 are frozen into an **external reference artifact**
for that MOP. All 24 observations of the case, and every later
candidate, use the same frozen objective-space points. Future MOP
implementations must not silently regenerate the reference from the
candidate's code. A review validates the reference geometry against
the MOP's established scientific model. There is no GT
recalibration or historical numerical-fixture replacement.

### Measurements and family of hypotheses

For each completed run, store its final non-dominated front,
objective summary and the quality indicators:

- **Primary certification metric:** IGD+ using the frozen front,
  the existing public reference-normalized metric semantics and
  lower-is-better interpretation.
- **Diagnostic-only metrics:** GD+ and EMD, evaluated from the same
  observed final front and frozen reference. They are stored for
  investigation, without being promoted to additional release
  significance tests after examining their values.

Only the six **IGD+** case means belong to the inferential family.
No new cases, metrics or run seeds can be added after looking at
the candidate results without a separately approved, versioned
protocol revision.

### Predeclared comparison and bounded cost

A future candidate repeats **exactly 24** complete seed-paired runs
per case. Define `d_i = IGD+_candidate(seed_i) -
IGD+_reference(seed_i)`, where positive values indicate worsening.
The scientifically declared practical mean-degradation threshold is

`delta = max(0.01, 0.10 * mean(IGD+_reference))`.

The absolute floor `0.01` is in the reference-normalized indicator
space, while the 10% term represents a relative loss from the
historical mean. Neither term is tuned using candidate outcomes.
These are proposed operational materiality thresholds for *release
regression*, not assertions that all scientific studies find such
changes important.

For each case, form the Student-t **two-sided** confidence interval
for `mean(d)`, using a Bonferroni simultaneous familywise bound of
`alpha_family=0.05` across the six prespecified cases (each tail
gets `0.05/(2*6)`). Classify:

- **`PASSED`** iff the upper interval bound is *strictly below*
  `delta`: evidence against material mean worsening.
- **`FAILED`** iff the lower interval bound is *strictly above*
  `delta`: evidence of material mean worsening.
- **`INCONCLUSIVE`** otherwise, including equality at the boundary.

Overall status is `FAILED` if any case fails, otherwise
`INCONCLUSIVE` if any case is inconclusive, otherwise `PASSED`.
All 24 seed pairs must be available; missing/invalid values are
errors, never omitted or silently imputed. No optional stopping,
post-hoc seed selection or rebaselining is permitted. The model
assumes independent paired seed observations and an adequate
approximation of paired-mean Student-t inference; diagnostics
must report unusually skewed or high-variance differences.

**Power and pilot interpretation:** The first historical reference
does not contain paired changes against a future version, and
therefore cannot yield a genuine empirical *future paired-difference*
standard deviation. The earlier provisional 80%-power sample-size
validator is **superseded** for this bounded release sentinel:
24 paired observations are predetermined by the cost envelope.
The baseline stores its across-seed variability and a prospective
sensitivity analysis; later certificates report observed interval
width and uncertainty. There is **no blanket claim of 80% statistical
power**. When the chosen budget is insufficient, the correct outcome
is `INCONCLUSIVE` and the reference remains frozen. A more
powerful investigation may be designed independently, with
preregistered revisions, but it cannot silently enlarge this
accepted family or cherry-pick a passing result.

## Historical registry, not only current fingerprint

The registry must keep independently versioned artifacts and a
manifest of their relationships, for example:

```text
moeabench/resources/certifications/index.json
moeabench/resources/certifications/0.18.0/environment.json
moeabench/resources/certifications/0.18.0/numeric-results.json
moeabench/resources/certifications/0.18.0/performance-reference.json
moeabench/resources/certifications/1.0.0/environment.json
moeabench/resources/certifications/1.0.0/numeric-results.json
moeabench/resources/certifications/1.0.0/performance-vs-0.18.0.json
```

The *existing* `moeabench/resources/certified_environment.json`
remains the current version-identity entry point until replacement
by a complete indexed reader. A successful certification registers
a **new immutable version entry** with source hash, dates,
artifact hashes, full observed outcomes, provenance and parent
baseline ID. It must refuse overwriting an approved historical
entry; an intentional correction is a new audited revision.

Historical records may include previous environment manifests
(e.g. 0.16.1) even if they lack comparable numeric/performance
outcomes. Missing measurements are explicitly `UNKNOWN`, not PASS.
A user-facing diagnostic may truthfully say "the **observed**
numeric signature matches version X" or "the measured performance
is statistically compatible with the reference for version Y"
**only if the corresponding reference check actually ran with
documented suitable inputs and environment**. A package-version
fingerprint match alone is never proof of matching numeric results
or performance distributions.

Baseline creation is *not* the next step automatically after
numeric certification. The normal life cycle is:

```text
Version A:    certify-numeric
              [separate, approved decision to establish a reference]
              record-performance-baseline(A)

Version B:    certify-numeric
              certify-performance(B vs archived A)
              [optional separate, approved decision to create baseline B]
```

No certificate may silently revise existing baseline values.
The historical 0.18.0 performance reference is to be recorded
once and then reused for future versions until the maintainer
explicitly approves a replacement.

## Consequences and safeguards

- Numerical replay and performance comparison remain scientifically
  separate; neither automatically changes or launches the other.
- The statistical case family is small and reproducible, and its
  maximum compute cost is explicit.
- Honest `INCONCLUSIVE` results take priority over unsupportable
  declarations of statistical power.
- A per-version index can later identify an **observed** match with
  an earlier certified version, preserving traceability rather
  than inferring scientific agreement from software versions alone.
- Accepted ADRs, mathematical GT and historical regression fixtures
  are immutable absent specific prior authorization. If implementing
  the new public names requires changing existing regression tests,
  **the maintainer must be warned and explicitly authorize those
  test changes first**.
- Implementations must not claim that this ADR alone executes runs,
  certifies performance or creates a historical reference.

## Related records

[ADR 0017](0017-comprehensive-calibration-and-testing-framework.md) —
earlier research calibration process and its historical high-budget
baseline, distinct from this bounded release-regression sentinel.

[ADR 0045](0045-component-seed-allocation.md) — fixed, namespaced
internal component seeds.

[ADR 0050](0050-certified-environment-verification-and-provenance.md) —
prior numerical/runtime certification, including successful 0.18.0
environment evidence.

[Issue #108](https://github.com/monacofj/moeabench/issues/108) —
implementation and acceptance tracking.


## Initial executable compatibility evidence (2026-11-18)

The controlled [six-case probe](https://github.com/monacofj/moeabench/actions/runs/44365881085)
passed on Python 3.14.4 with the pinned reference numerical libraries
and effective NumPy/OpenBLAS preflight. It verified each exact public
MOP/MOEA constructor above, its deterministic 512×3 objective reference,
and two distinct run seeds `424242` and `528971` using a **three-generation
compatibility smoke**, with finite nonnegative IGD+ for all 12 runs.

These checks **verify interface feasibility only**. They do not
represent the eventual 60-generation/24-seed baseline, estimate
its statistical precision, approve the geometric sampling for a
new release or certify any candidate's performance. In particular,
the DPF1 case has a markedly different raw IGD+ scale, supporting
the predetermined combined absolute/relative materiality rule.
No MOP/MOEA source, numerical regression test, GT or historical
fixture was modified during this probe.


## Maintainer-directed refinement — 2026-11-18, implementation of issue #108

This dated amendment **supersedes**, for implementation, the numerical
archive duplication, 24-seed fixed-budget design, 10% mean-degradation
margin, 0.01 absolute floor, and 5% familywise alpha stated earlier in
this ADR. The earlier text is retained as decision history.

**No new numerical-front archive per release.** Retain the current
canonical frozen numerical regression fixtures once. A numerical
certificate stores a compact manifest: software versions, effective
runtime profile, numerical fixture hashes, test counts, evidence URL,
and source revision. Existing 0.18.0 formal certification remains valid.
Historical Git versions provide earlier fixtures when explicitly
needed for actual comparison. Geometric reference fronts for a
*statistical performance experiment*, however, are kept as an
integral, frozen input to that experiment so future code cannot
regenerate different fronts while claiming a comparable score.

**Statistical risk policy.** For six fixed IGD+ MOP/MOEA cases,
use familywise two-sided Student-t simultaneous intervals at
`alpha=0.01` and a conservatively small, scale-relative
materiality margin of **0.1% of each case's historical mean IGD+**,
with only `1e-12` as a numerical zero floor. The 0.1% threshold is
an explicit, reviewable **engineering tolerance** for the release
sentinel, not a universal claim of scientific equivalence. No 10%
or 0.01 floor is permissible. A future scientific decision changing
the threshold must be authorized *before* candidate results are seen.

**Prospective design, no invented pilot variance.** Record **64
prespecified distinct seeds** per case for the first reference (six
cases, 384 total solver runs, at 48 individuals and 60 generations).
When comparing a candidate, run the first **eight seed pairs as a
pilot** to estimate the *paired difference* standard deviation;
these eight outcomes are **excluded from confirmatory analysis**.
Determine the smallest confirmatory sample size using a noncentral
Student-t design for **95% planned probability of concluding
noninferiority at true zero mean difference** with the declared margin,
case family and alpha, requiring at least eight and at most **56**
new pairs per case. Missing power within this ceiling means no
certificate, with reason `INSUFFICIENT_POWER`, rather than inventing
variance or silently increasing the budget. When 56 or fewer are
enough, run exactly that many new prespecified pairs from the remaining
56 seeds. A predeclared family of six hypotheses stays fixed; no
adaptive margin and no optional stopping. The calculation's model
assumptions and realized confidence interval are reported, not
misrepresented as guarantees.

**Binary operational gate.** A release certificate is
`CERTIFIED` only when all six simultaneous upper confidence
bounds lie strictly below their respective allowable margins;
otherwise `NOT_CERTIFIED`. Record detailed reasons:
`REGRESSION_DETECTED`, `INSUFFICIENT_EVIDENCE`,
`INSUFFICIENT_POWER` or `INVALID_REFERENCE`.
These reasons do not create a third publish/accept decision.
A failing or insufficient study **never updates its reference**.

**Legacy code/tests.** No regression test is authorized for
modification by this amendment. Existing synthetic unit tests
asserting old CLI/v1/v2 behavior are to be audited before their
replacement; the maintainer must receive specific advance warning
and authorize any regression-test modifications. New regression
tests may be added. A branch may implement the new protocol
without deleting the old tested interface until that approval.

**Prior audit data.** `tests/calibration_reference_audit_v0.13.2.json`
is unrelated to versioned performance benchmarks. Replacing its
use in existing numerical regression tests is a separate,
test-affecting change requiring prior authorization, and is not
a prerequisite to the statistical baseline.

**No automatic baseline renewal.** Completing a numeric
certificate does not invoke baseline recording. The
`--record-performance-baseline` operation is a separately
approved one-time creation of the 0.18.0 historical sample;
`--certify-performance` always consumes a previously frozen
sample, never writes or replaces it.


## Completion and audit record — issue #108, 2026-11-18

- The exact preapproved 0.18.0 source revision
  `0d3e66804351e23b93fb483ff25e178e06ef6e12` passed the
  [compact numerical certification](https://github.com/monacofj/moeabench/actions/runs/44403553093),
  independently of performance baseline production. It stores
  package/dependency version and native runtime information, regression
  hashes and source revision. Historical numerical front targets are
  **not duplicated per version**.
- The source revision generated all **384 empirical performance
  observations** in
  [run 44401850675](https://github.com/monacofj/moeabench/actions/runs/44401850675).
  The original candidate artifact SHA-256 was
  `8e635a027520b22fc14aa703ba85d105207b708cf05c885407b69ecb22212730`.
  An independent
  [technical audit](https://github.com/monacofj/moeabench/actions/runs/44409693329)
  reproduced that exact digest, validated the 64-seed sets and frozen
  objective-reference hashes and independently replayed **three
  solver runs per case** (18 total), all matching their stored IGD+
  values. The subsequent `APPROVED` record states **automated
  technical review**; it does not represent human inspection of
  every experiment.
- The old standalone statistical reference protocols, old public
  performance-certification names and source modules were retired
  before 1.0.0, with equivalent or stronger regression guards in the
  migrated tests. The three public commands are independently
  dispatched; no action may automatically renew performance baselines.
- The previously retired diagnostic GT-audit artifact was found
  referenced by numerical-stability tests. The exact 1000-point
  geometry reference for DTLZ8, DTLZ9 and DPF5, plus exact sample
  counts for DTLZ2/3/4, were copied into the dedicated
  `tests/stability_gt_geometry_reference.json` fixture. The
  **mathematical regression comparisons and numerical tolerances were
  preserved**; the unrelated audit dataset was removed from the
  active branch. No MOP or MOEA implementation changed.
- The definitive public release gate is `CERTIFIED` only if **all**
  six hypotheses establish noninferiority under the accepted
  simultaneous interval. Any observed material regression or
  insufficient evidence/power yields `NOT_CERTIFIED` and a
  specific diagnostic cause, without editing the frozen reference.
- The 0.18.0 performance artifact is the inaugural historical
  reference, **not** a claim that a future 1.0.0 (or changed
  0.18.0 source revision) is already certified. The version must
  advance before certifying materially changed source code.


## Maintainer-authorized inaugural DPF5 GT calibration — 2026-11-20 (#119/#140)

The maintainer explicitly authorized correction of the pre-existing regression
tests and **replacement of the provisional DPF5 geometric GT fixture as the
first scientifically valid calibration** for the restored published DPF5
equations. The existing `tests/stability_gt_geometry_reference.json` DPF5
points and `tests/stability_reference_gt.json` DPF5 tolerance were generated
for an evaluator/sampler subsequently shown to diverge from the DPF5 paper.
They were *provisional implementation behavior*, not an accepted
scientific calibration of the published model.

**Narrow approved exception to the immutable-reference safeguard:**
replace **only the DPF5 geometry and numerical-tolerance entries** with
a first independently validated, reproducible corrected-model calibration,
and update the fifteen preexisting defect-observation tests to assert
correct published mathematical behavior. The former entries remain
recoverable through Git history but are **not** a separate scientific GT
version or a predecessor calibration artifact. This express approval
supersedes the prior no-test-edit/no-baseline-rewrite restriction **solely
for this initial correction**; it is not a general permission to rewrite
accepted numerical references whenever software output changes.

Inaugural DPF5 GT acceptance requirements:

1. Use the corrected native `DPF5(M=3,D=2)` reference generator at the
   same **1,000-point** stability-reference geometry budget.
2. Independently reconstruct the published DPF5 equations, verify all
   reference objectives against that oracle, verify both P/D branches,
   `g=0`, sphere identity, effective cardinality and deterministic replay.
3. Generate under the pinned reference numerical runtime, recording the
   source identifier, calibration method and SHA-256 of the actual GT.
   Numerical stability comparisons must remain **meaningfully strict**
   against this reference; do not merely increase the old tolerance.
4. Leave DTLZ references and the distinct release-performance reference,
   statistical certification policy and other numerical baseline contracts
   **untouched**.
5. Rerun complete unit, reference numerical stability, and independent
   DPF3/DPF5 GT workflows. Fix actual defects rather than weakening the
   new verification.

The immutable reference rule of this ADR remains fully applicable
**after** accepting this inaugural DPF5 calibration. This amendment has
no effect on automatic `mop.calibrate()` budget policy, FAIR research,
or other MOPs.
