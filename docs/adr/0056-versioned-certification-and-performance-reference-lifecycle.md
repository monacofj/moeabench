<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0056: Versioned Numerical and Performance Certification

## Status

Accepted — approved by the maintainer on 2026-11-18 (issue #108).
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
