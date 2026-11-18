<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0050: Certified Environment Verification and Experiment Provenance

## Status

Accepted

## Context

Scientific comparisons require evidence that the software environment can
reproduce the reference numerical behavior. Matching dependency versions is
helpful, but it does not establish that independent operating systems, Python
patch releases, native numerical libraries, or architectures yield identical
optimization trajectories. Conversely, a changed version does not necessarily
imply changed results.

Prior numerical regression tests protected reference metrics and fronts, but
did not provide a single user-facing distinction between dependency identity,
observed numerical behavior, and the execution environment of a persisted
experiment. Automatically running exhaustive tests during routine experiment
execution would also be unexpectedly expensive.

## Decision

MoeaBench uses three complementary, explicitly separate checks:

1. **Version identity.** `mb.system.verify_version()` compares the runtime
   fingerprint with a developer-certified, versioned manifest shipped with the
   package. The first `exp.run()` performs this inexpensive check and caches
   it. The outcome is `MATCH`, `MISMATCH`, or `UNCONFIGURED`. It is not a
   numerical equivalence test.
2. **Numerical stability.** `mb.system.verify_stability()` is explicitly
   invoked and tests exact PRNG signatures plus representative reference
   optimization fronts under stated numerical tolerances. Its outcome is
   `PASSED` or `FAILED`; success establishes only the tested behaviors.
   It does not modify the manifest or reference fixtures.
3. **Experiment provenance.** Execution-time environment versions travel with
   saved experiments. `mb.system.verify_experiment(exp)` compares the current
   runtime with that provenance; loading an archive performs this comparison
   when the metadata exists. Legacy archives without it yield `UNKNOWN`
   rather than a false match.

`exp.report()` displays cached verification states and does not trigger
numerical checks. Version mismatch and unconfigured references yield a
nonrepeating advisory at execution time; experiments remain executable.
Changing the environment invalidates the in-session stability status.
Experiment provenance mismatches remain visible even if the current runtime
passes a separate numerical check.

The reference manifest is generated only by a successful
`python test.py --certify` run in the designated development environment.
That command executes the complete `--deep` suite before writing
`moeabench/diagnostics/resources/certified_environment.json`. Failed tests
must not be bypassed by relaxing thresholds or silently regenerating
reference outputs.

The manifest tracks Python, MoeaBench, and dependencies relevant to scientific
calculations (NumPy, SciPy, pymoo, pandas, moocore, DEAP). Matplotlib remains
outside *numerical* certification; its API is protected by functional
integration tests.

## Evidence and limitations

During PR #68, the `NSGA3 × DPF1` stability fixture yielded a three-point
front in the certified Python 3.14.4 environment but a two-point front in
other CI runs. Controlled investigation found reference-direction coordinates
differing by at most approximately `8.88e-16`; subsequent NSGA-III selection
amplified these perturbations into distinct trajectories. In a diagnostic
experiment only, rounding generated directions to 13 decimal places caused
identical trajectories across the tested Python 3.12.15, 3.14.4, and 3.14.8
environments. No such rounding was adopted in the production algorithm.

The exact origin of the floating-point perturbation in the reference-direction
generator was not established, and a matching runtime fingerprint does not
guarantee bitwise identity in arbitrary future deployments. Canonicalized
cross-platform reference directions require a separate design decision.

## Consequences

- Users can distinguish `version match`, `numerical checks passed`, and
  `experiment provenance match` without conflating their guarantees.
- Release certification remains an explicit, reviewable operation.
- A single smoke-suite pass cannot certify all optimization problems,
  configurations, or hardware combinations.
- Diagnostics can expose meaningful environmental drift without preventing
  research runs solely because versions differ.

## Alternatives considered

- **Infer certified versions from whichever machine runs the tests:** rejected
  because it makes the reference self-validating.
- **Automatically run full stability checks on every experiment:** rejected
  because of execution cost and unexpected side effects.
- **Treat matching versions as proof of identical results:** rejected based on
  the distinction between identity and actual numerical behavior.
- **Round all NSGA-III reference directions in production:** deferred; doing so
  changes algorithm behavior and requires separate justification and baselines.

## Related decisions and documentation

- [ADR 0031](0031-numerical-integrity-verification-tier.md): numerical integrity tests.
- [ADR 0045](0045-component-seed-allocation.md): deterministic component seed allocation.
- [Environment verification](../environment-verification.md): user/developer commands.
- [Test architecture](../test.md): testing scope and levels.
- [Reproducibility](../reproducibility.md): research interpretation.

## Maintainer-authorized testing-policy amendment — 2026-11-17

The release validation plan separates two kinds of stability assessment:

- **Numerical stability** verifies deterministic reference signatures, fixed
  input-to-output indicator values, and representative seeded solver fronts.
  One execution per frozen seeded configuration suffices for replay comparison;
  it does not establish a statistical performance distribution.
- **Performance stability** evaluates repeated stochastic experiments to
  detect changes in solver performance distributions. It is a separately
  invoked, explicitly scoped diagnostic and must not be triggered implicitly
  when numerical stability fails. Its repetition count and statistical
  reference policy are selected for the particular study, rather than
  automatically multiplying a fixed 30-run battery across every MOP/MOEA.

For the FAIR-free recovery and release workflow, `basic` numerical
stability in the designated certified environment is the required numerical
validation tier. The historical `deep` statistical test is not a
certification prerequisite while its missing legacy baseline makes it
non-operative. A successful but skipped `deep` test must never be reported as
evidence of performance stability. The `--certify` mechanism must be brought
into conformity with this amendment **before it is used to update** the
environment manifest. It may neither regenerate numerical fixtures nor
accept failing checks. The reference environment and provenance separation
specified above remain intact.

This amendment supersedes the earlier requirement in this ADR to execute a
complete `--deep` suite as a prerequisite for `--certify`. The original
historical wording is retained for traceability; the CLI/documentation
transition is part of the next implementation stage. No change to a regression
test or frozen baseline is authorized by this amendment.

## Implementation record — draft PR #91 (2026-11-18)

The code implements the 2026-11-17 separation as two explicitly selected
paths:

- `python test.py --numeric-stability --smoke|--basic` uses the existing
  deterministic `scope_stability` pytest regressions. `--stability` remains an
  alias. The historical `--stability --deep` and no-scope `--deep` fail
  explicitly because the required old statistical artifact is missing.
- `python test.py --performance-stability --performance-reference PATH` and
  `mb.system.verify_performance_stability(PATH)` use an externally supplied
  versioned reference. No default statistical run count or reference is
  fabricated, and a numerical failure never triggers performance testing.
- `python test.py --certify` uses the complete `basic` development suite
  before any change to the manifest. This retains the requirement to review
  a new environment fingerprint before committing it.

The initial **candidate** performance-reference protocol records source
revision, reference origin, reviewer, fixed objective-space front, explicit
MOP/MOEA parameters, selected seeds and seed-paired previous values. The
runner compares minimized metrics with a one-sided paired t-test and requires
both statistical significance and a predeclared practical effect. Case
selection, minimum useful replication count, external source credibility,
multiplicity treatment, and statistical power still require scientific review
before this protocol becomes a release certification gate. The protocol's
`PASSED` status means only that its specified regression rule did not fire.

PR #91's CI reproduced the previously documented `NSGA3 x DPF1` two-versus-
three point discrepancy even with Python 3.14.4 and the pinned NumPy/SciPy/
pymoo/DEAP/moocore/pandas versions. This is an unresolved numerical stability
failure, **not** a justification to regenerate the frozen front or alter the
existing regression test. The repository's manifest still identifies MoeaBench
0.16.1 and must not be refreshed until the full currently required suite passes.

No accepted numerical tolerance, reference front, seed contract, existing
regression test, or scientific baseline was changed in this stage.

The pre-existing NSGA-III/DPF1 cross-environment discrepancy and its exploratory
investigation are tracked separately in [issue #92](https://github.com/monacofj/moeabench/issues/92).
The test architecture retains the original regression as an active gate; no
regression fixture, tolerance, test, or production solver was changed here.


## Performance-protocol scientific review — issue #98 (2026-11-18)

The previous v1 statistical performance reference remains an explicitly
requested **diagnostic** whose `PASSED` status is limited to its old
p-value/effect threshold rule. It is not used for release certification.
An additive v2 reference protocol is available for future independently
approved benchmark artifacts: it reports `PASSED` for an upper simultaneous
mean-change confidence bound below a prespecified practical margin,
`FAILED` for a lower bound above that margin, and `INCONCLUSIVE` otherwise.
The familywise adjustment is Bonferroni over the specified case family, and
a preregistered paired-variance pilot determines the minimum seed count
under a Student-t power model. No automatic 30-run matrix is introduced;
no reference is regenerated from the candidate execution.

**Release decision:** an externally reviewed, versioned prior-release
statistical reference and pilot currently do not exist. Accordingly,
statistical performance remains **opt-in, outside `--basic` and
`--certify`**, until a reference and its scientific sampling assumptions
are approved in a separate decision. Matching the current release's
measurements to self-generated data would not constitute independent
certification. This is consistent with, and does not replace, the 2026-11-17
maintainer-authorized separation of numerical and performance stability.
The existing certified numerical manifest, reference fronts, tests and
historical provenance semantics remain unmodified.

Protocol details and limits are described in
[Performance stability](../performance-stability.md).


## Completed three-stage recovery and numerical reference profile — 2026-11-18 (issue #102)

**Scope and precedence.** This is a dated implementation/evidence addendum to
the accepted ADR. It does not replace the historical accounts of PRs #68/#91,
alter the RNG contract of ADR 0045, relax numerical tolerances, or authorize
regeneration of reference outputs. Read the earlier unresolved-drift and
`--deep` statements in their *historical* context: the 2026-11-17
maintainer-authorized amendment replaced the deep release prerequisite with
the complete basic suite; the investigation below subsequently established a
tested reference environment. A successful CI run by itself does not update
the certified manifest.

### Sequenced verification work

- **Unit correctness and core API**: PR [#88](https://github.com/monacofj/moeabench/pull/88)
  recovered FAIR-independent GT calibration and historical
  conventional-HV contracts; PR [#90](https://github.com/monacofj/moeabench/pull/90)
  consolidated unit contracts and independent mathematical oracles.
  `mop.calibrate()` and `mop.calibrated_gt` belong to the MOP irrespective
  of any FAIR diagnostic. In `H_abs`, the MOP's calibrated GT defines
  the denominator; optional `ref` selects external normalization
  geometry, not the GT source. The temporary availability exception in
  ADR 0048 concerns FAIR/clinical APIs only.
- **Numerical and performance tests**: PR [#91](https://github.com/monacofj/moeabench/pull/91)
  separated deterministic replay from opt-in statistical comparison.
  PR [#99](https://github.com/monacofj/moeabench/pull/99) added 23
  fixed-input conventional-metric stability tests, with 21 analytical
  reference values cross-checked by an implementation using the
  Python standard library alone. PR
  [#100](https://github.com/monacofj/moeabench/pull/100) added the
  optional three-way performance protocol v2, with explicit
  `INCONCLUSIVE`, familywise adjustment and a priori sample-size
  planning. The v1 interface remains available but its nonfailure
  result is not evidence of equivalence.
- **Failures detected by the tests**: issue
  [#92](https://github.com/monacofj/moeabench/issues/92) investigated
  the historical NSGA-III×DPF1 front discrepancy **after** implementing
  the test architecture. PR
  [#101](https://github.com/monacofj/moeabench/pull/101) applied a
  controlled *CI reference runtime* rather than modifying the solver
  or its approved reference. The issue and PR are closed/merged.

### NSGA-III × DPF1: experimentally identified environmental drift

With Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, pymoo 0.6.2, DEAP
1.4.4, moocore 0.3.2 and pandas 3.0.6, different hosted CI processors
produced different fronts **despite matching version metadata and seeds**.
The controlled [issue #92 experiment](https://github.com/monacofj/moeabench/actions/runs/44340186293)
isolated two relevant numerical execution choices:

1. NumPy float64 `power` SIMD dispatch during SBX crossover. With
   the default `X86_V4` dispatch the resulting final front contained
   two points. With `NPY_DISABLE_CPU_FEATURES=X86_V4`, the SBX
   output matched its historical signature, but a different
   three-point front could still emerge.
2. The OpenBLAS CPU kernel used along the NSGA-III
   survival/normalization path. Setting `OPENBLAS_CORETYPE=HASWELL`,
   while keeping the NumPy `X86_V4` mask and single-threaded BLAS,
   reproduced the **exact historical three-point front** (SHA256
   prefix `77dc14fc...`) under both Ubuntu 22.04 and 24.04.

The complete *unaltered* numerical basic suite passed **18/18** on
both images in this combination. After the independent conventional
metric references were added, the two regular CI workflows
([unit contract](https://github.com/monacofj/moeabench/actions/runs/44341867393)
and [stability separation](https://github.com/monacofj/moeabench/actions/runs/44341867318))
each passed **41/41** basic numerical checks; the unit/integration
and smoke jobs also passed. Those are observed results for the
tested profiles, not proof of numerical portability on all hosts.

### Effective runtime policy and verification

Only the two `reference-numeric-stability` CI jobs use these pinned
Linux/x86_64 numerical environment variables:

```bash
NPY_DISABLE_CPU_FEATURES=X86_V4
OPENBLAS_CORETYPE=HASWELL
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

The read-only
[`scripts/check_numerical_reference_runtime.py`](../../scripts/check_numerical_reference_runtime.py)
verifies the **actual** NumPy float64 `power` dispatch
(`baseline(X86_V2)`), the loaded NumPy and SciPy OpenBLAS CPU kernels
(`Haswell`) and single-thread operation. Failure of this preflight
invalidates claims of parity with the tested reference runtime.
This is a scoped CI verification arrangement, not a new requirement for
all user experiments or an unconditional cross-platform reproducibility
guarantee. No production algorithm, seed protocol, frozen front, or
historical test was modified by PR #101.

### Formal release certification remains separate

The manifest
`moeabench/resources/certified_environment.json` still records a
previously certified MoeaBench **0.16.1** version snapshot, including
its legacy description. It was intentionally **not rewritten** by
the CI repair or documentation work. When certifying a new release,
developers must establish the designated runtime, run
`python test.py --basic`, and then run
`python test.py --certify`, which repeats the full basic suite
without inherited `PYTEST_ADDOPTS` selection filters and updates
the manifest **only after success**. The standalone `--certify` runner
currently tests the suite result but does not itself inspect the
effective CPU/BLAS profile; that preflight is explicit in CI and
must be run when manually preparing a certified reference environment.

There is **no approved external statistical performance baseline** for
the forthcoming FAIR-free release. Statistical comparisons remain
opt-in and outside `--basic`/`--certify` (2026-11-17 amendment and
issue #98). Formal manifest certification and a future approved
performance reference require their own review; neither was implied
by closing issues #92, #97 and #98.

For the up-to-date command and classification details see the
[complete test architecture manual](../test.md),
[environment verification](../environment-verification.md) and
[performance stability protocol](../performance-stability.md).
