<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench test architecture and verification manual

This document is the operational reference for the **current FAIR-free**
MoeaBench test runner. It describes what is actually implemented in
[`test.py`](../test.py), [`tests/conftest.py`](../tests/conftest.py),
the regression fixtures, the continuous-integration workflows and the
optional statistical comparison API. It distinguishes three questions:

1. **Is a component correct for known inputs?** Unit tests and independently
   derived mathematical oracles.
2. **Does the same numerical computation replay its approved reference?**
   Deterministic numerical-stability tests of indicators, ground truth (GT),
   random streams and seeded solver trajectories.
3. **Has stochastic optimization performance materially deteriorated across
   repetitions?** An explicitly requested, *reference-backed* statistical
   assessment with an independent prior baseline.

These questions have different evidence, costs and conclusions. A passing
unit test does not demonstrate statistical performance stability; a
one-seed replay test does not characterize a distribution. Conversely, a
numerical-regression failure must remain visible, even when the
implementation passes its mathematical unit tests.

## 1. Test runner and command selection

The repository root contains the executable orchestration script
`test.py`. It starts pytest in a subprocess for each selected scope, prints
the collected/passed/failed/error/skipped counts and returns a nonzero exit
status on a failed batch. Its default command is:

```bash
python test.py
# Equivalent to:
python test.py --smoke
```

With **no scope**, the runner executes the three pytest scopes, in order:
unit, integration and numerical stability. With an explicit scope, it runs
**only** that scope. The CLI has mutually exclusive scope selectors and
mutually exclusive level selectors.

| Command | Effective selection |
|---|---|
| `python test.py` | All unit tests + smoke integration + smoke numerical stability |
| `python test.py --basic` | All unit tests + smoke/basic integration + smoke/basic numerical stability |
| `python test.py --unit` | **All** unit tests |
| `python test.py --unit --smoke` | **All** unit tests (level ignored for unit execution) |
| `python test.py --unit --basic` | **All** unit tests |
| `python test.py --unit --deep` | **All** unit tests; this is not the legacy statistical suite |
| `python test.py --integration --smoke` | Smoke integration cases |
| `python test.py --integration --basic` | Smoke and basic integration cases |
| `python test.py --numeric-stability --smoke` | Smoke numerical stability cases |
| `python test.py --numeric-stability --basic` | Smoke and basic numerical stability cases |
| `python test.py --stability --basic` | Alias for the preceding numerical selection |
| `python test.py --performance-stability --performance-reference PATH` | A separate opt-in statistical comparison; **no pytest tier** |
| `python test.py --list` | List the collected official pytest tests and their exact classifications |
| `python test.py --certify` | Run a new complete basic battery and, only after success, update the version manifest |

The `--stability` spelling is a compatibility alias of
`--numeric-stability`. The performance reference flag is only valid with
`--performance-stability`. That statistical scope cannot be combined with
`--list` or `--smoke`/`--basic`/`--deep`.

**Unsupported configurations fail visibly.** The legacy
`python test.py --stability --deep` and all-scopes
`python test.py --deep` fail with a usage error: the obsolete statistical
baseline is absent. `--integration --deep` is also rejected. The legacy
test file remains in the repository for historical examination; it is not a
substitute for an approved external performance reference. The runner's
`--certify` option cannot be combined with any other selector.

### Listing versus running

`--list` collects pytest node IDs without executing their test functions.
List filters select the **exact declared level**, whereas execution is
*inclusive* up to the requested level for integration/stability. This
distinction matters particularly for unit tests:

```bash
python test.py --list
python test.py --list --unit
python test.py --list --integration --smoke
python test.py --list --numeric-stability --basic
```

For example, `--list --unit --smoke` displays unit tests classified as
smoke, but `--unit --smoke` **executes every unit test**. Do not infer an
execution subset from a listing filter. Collection failures (including
unclassified test paths) are errors, not silently skipped tests.

## 2. The scope × level ontology

Each official pytest test has exactly one scope and one level. The
classification code lives in `tests/conftest.py`, and markers are applied
at collection time according to the file path. This preserves a uniform
vocabulary while allowing different operational meanings across scopes.

| Scope | Purpose | Effective levels |
|---|---|---|
| `unit` | Local contracts, API semantics and independently testable mathematics | **Total only**; all flags execute the full unit scope |
| `integration` | Composition of public APIs and end-to-end workflows | `smoke` then `basic` (inclusive) |
| `stability` | Deterministic numerical references and seeded solver replay | `smoke` then `basic` (inclusive) |
| Statistical performance | Repeated, seed-paired quality comparison with a reviewed historical reference | Separate command/API, **not** a pytest level |

The actual pytest marker vocabulary is `scope_unit`,
`scope_integration`, `scope_stability`, `level_smoke`,
`level_basic`, and `level_deep`. The runner constructs the expressions:

```text
unit:                scope_unit
integration smoke:   scope_integration and (level_smoke)
integration basic:   scope_integration and (level_smoke or level_basic)
stability smoke:     scope_stability and (level_smoke)
stability basic:     scope_stability and (level_smoke or level_basic)
```

The marker `level_deep` identifies the retained historical file; it is
excluded from active numerical certification. The runner does not execute
all six markers as a single undifferentiated test level.

### 2.1 Unit — independent correctness and public contracts

Unit tests are collected from `tests/unit/` and
`tests/test_light_tier.py`. Their coverage includes the public API,
numerical primitives, MOP-owned GT calibration, metric inputs and outputs,
float64 behavior, persistence and serialization, seed allocation, test-runner
contracts and statistical-protocol decision logic.

Unit smoke classification is assigned to:

```text
tests/unit/test_api_surface.py
tests/unit/test_compare_aliases.py
tests/unit/test_system.py
```

Other unit files receive `level_basic`, but that marker does not change
the **total-only** unit execution rule. Examples of independent oracles and
other additive protections introduced in the recovery sequence are in
`tests/unit/test_independent_metric_oracles.py`,
`tests/unit/test_stability_separation_contract.py` and
`tests/unit/test_performance_stability_v2.py`.

A small exact geometric calculation or an independently audited numerical
oracle establishes that a metric computes the expected value on *that
input*. It cannot alone prove the stability of an entire stochastic MOEA
trajectory or its distribution of results.

### 2.2 Integration — composition and persistence

Integration tests live under `tests/integration/`. The explicitly
selected smoke files are:

```text
tests/integration/test_topology_gt_protocol.py
tests/integration/test_view_dispatch_pipeline.py
```

All other integration files receive `level_basic`. The basic scope
includes smoke. Tests exercise real combinations of problem/algorithm,
metric/statistical processing, view dispatch, GT, archives and reporting.
For example, the CI's integration workflow runs the GT empirical-search
contract and creates small example experiment archives before the full
integration-basic run; those preparation steps are part of the workflow
rather than hidden behavior of `test.py`.

### 2.3 Numerical stability — reference behavior

The numerical stability scope consists of these explicit files:

| File | Level | What it protects |
|---|---|---|
| `tests/test_stability_smoke.py` | smoke | Python/NumPy RNG signatures, experiment seed scheduling, light GT geometry and fixed-seed solver fronts |
| `tests/test_stability_basic.py` | basic | Historical sampled/analytic GT geometry and canonical seeded MOP×MOEA final-front references |
| `tests/test_stability_metrics.py` | basic | Independently derived 2D/3D GD, IGD, GD+, IGD+, EMD and exact/Monte Carlo HV behavior |
| `tests/test_stability_deep.py` | historical deep | Old statistical experiment lacking `tests/baselines_v0.8.0.csv`; intentionally excluded |

The historical numerical reference inputs live in
`tests/stability_reference_fronts.json`,
`tests/stability_reference_gt.json` and
`tests/calibration_reference_audit_v0.13.2.json`. For the conventional
metrics, `tests/conventional_metric_reference_v1.json` supplies an
**additional**, versioned fixed-input reference. Its analytical derivation
is documented in
[`docs/audits/conventional-metric-reference-v1.md`](audits/conventional-metric-reference-v1.md).
The separate standard-library-only script
`python scripts/audit/verify_conventional_metric_references.py`
checks 21 frozen values without invoking MoeaBench, NumPy, SciPy or pymoo.

The conventional-metric regression file contributes 23 test cases: exact
distance/EMD identities, HV `raw` and `rel`, HV `abs` with a
**calibrated MOP-owned GT**, and seeded approximate-HV repeatability.
For `scale="abs"`, GT comes from `mop.calibrate()` and
`mop.calibrated_gt`; the external `ref`, when supplied, determines
normalization bounds, **not the GT denominator**. This protects the
FAIR-independent calibration API and its separation from HV geometry.

Seeded solver-front regressions compare the observed sorted front to
canonical expected coordinates **and shape**. The historical basic
regression uses `np.allclose(atol=1e-12, rtol=1e-8)`; its tolerances
must not be silently relaxed. Analytical front geometry uses its own
explicit thresholds, and the fixed conventional metric fixture
uses `atol=1e-10, rtol=0`. Monte Carlo HV is also checked against an
analytically computed volume with a documented **sampling** allowance,
rather than declaring a host-specific sampled bit pattern universal.
There is no generic six-decimal tolerance overriding these contracts.

**Numerical replay is scoped.** One execution is enough to compare a
fixed seed, environment and reference front with a frozen deterministic
output. That does not justify a claim that the algorithm's *performance
distribution* is unchanged.

## 3. Test isolation and regression safeguards

The autouse fixture `_deterministic_test_rng` in `tests/conftest.py`
resets the process-global Python `random` and NumPy legacy RNG state
to `20260316` before each official pytest test. This protects test
isolation; production MOEA seed derivation is governed separately by
[ADR 0045](adr/0045-component-seed-allocation.md). It does not force
all possible local generators or low-level numerical kernels to be
identical across machines.

Regression references are **scientific contracts**, not disposable CI
artifacts. Before changing a public metric, a GT calibration route,
a random stream or a solver wrapper:

1. Consult the applicable accepted ADRs, including
   [ADR 0018](adr/0018-consolidated-scientific-report-and-gt-density-refinement.md),
   [ADR 0021](adr/0021-numerical-integrity-and-validation-pipeline-float64-policy.md),
   [ADR 0035](adr/0035-triple-mode-hypervolume-reporting.md),
   [ADR 0045](adr/0045-component-seed-allocation.md),
   [ADR 0047](adr/0047-conventional-hypervolume-relative-scaling.md) and
   [ADR 0050](adr/0050-certified-environment-verification-and-provenance.md),
   as relevant.
2. State which public contracts and historical regressions must remain
   invariant. Prefer adding new independent tests to modifying old ones.
3. Investigate an unexpected failure against both the known mathematics
   and the recorded numerical environment; retain the failing evidence.
4. **Warn the maintainer and obtain explicit authorization before changing
   any existing regression test, expected value, frozen fixture or accepted
   ADR contract.** A change incompatible with an ADR requires a reviewed,
   authorized architectural decision first.
5. Never regenerate a baseline to make a newly failing CI run green.
   Track a genuine numerical discrepancy separately from the architecture
   and coverage of the test suite.

This policy is why the NSGA-III×DPF1 investigation in
[issue #92](https://github.com/monacofj/moeabench/issues/92) was resolved
by identifying a controlled **runtime profile**, without altering its
expected front, test or algorithm.

## 4. The reference numerical environment

Matching Python/package versions is **identity evidence**, not proof of
identical native numerical execution. The NSGA-III×DPF1 historical
reference depends on two choices: NumPy float64 `power` dispatch during
SBX variation and OpenBLAS's CPU kernel during NSGA-III
normalization/survival.

The controlled CI investigation
([run 44340186293](https://github.com/monacofj/moeabench/actions/runs/44340186293))
reproduced 18/18 original basic tests on Ubuntu 22.04 and 24.04 with
the same pinned stack and both runtime settings. After the additional
conventional-metric tests were integrated, both regular workflows
([unit-contract CI](https://github.com/monacofj/moeabench/actions/runs/44341867393),
[stability-separation CI](https://github.com/monacofj/moeabench/actions/runs/44341867318))
passed **41/41 numerical basic tests** and the other scopes. Those
results were obtained **without** modifying historical numerical targets.

The `reference-numeric-stability` jobs in
[`.github/workflows/stability-separation.yml`](../.github/workflows/stability-separation.yml)
and
[`.github/workflows/unit-contract-tests.yml`](../.github/workflows/unit-contract-tests.yml)
use Python 3.14.4 with the version-pinned NumPy/SciPy/pymoo/DEAP/moocore/pandas
stack and the following **Linux/x86_64 reference profile**:

```bash
export NPY_DISABLE_CPU_FEATURES=X86_V4
export OPENBLAS_CORETYPE=HASWELL
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# In a developer checkout with the pinned numerical dependencies:
python -m pip install threadpoolctl
python scripts/check_numerical_reference_runtime.py
python test.py --numeric-stability --basic
```

The preflight verifies environment variables **and the effective**
NumPy `power` kernel (`baseline(X86_V2)`), both loaded NumPy/SciPy
OpenBLAS kernels (`Haswell`) and their single-threaded execution.
If this check fails, the host must not be represented as the tested
reference environment; do not bypass it or silently relabel results.
The general-purpose unit, integration and smoke jobs may run on other
runtimes without inheriting these reference-job settings.

The observed profile establishes reproducibility for the **tested**
environment and fixtures; it does not guarantee bitwise portability
across arbitrary processors, operating systems or native libraries.
See [Environment verification](environment-verification.md) and
[ADR 0050](adr/0050-certified-environment-verification-and-provenance.md).

## 5. Statistical performance stability: explicitly opt-in

Statistical performance stability measures whether *mean minimized solver
quality* has worsened beyond a declared practical margin, over an
explicitly chosen set of stochastic runs. It is **separate** from all
pytest `--smoke`/`--basic` scopes and from environment certification:

```bash
python test.py --performance-stability --performance-reference /path/to/reviewed.json
```

The equivalent programmatic entry point is
`mb.system.verify_performance_stability("/path/to/reviewed.json")`.
It invokes only the scenarios enumerated in the reference: there is no
implicit grid of all MOPs×MOEAs, no universal 30-run requirement and no
automatic invocation following a failed numerical test.

**A real, independently approved historical performance reference is
not currently bundled.** Synthetic observations in unit tests exercise
decision logic but cannot certify the released software's performance.
An input reference is JSON with its own schema, explicit MOP/MOEA
constructor parameters, minimized quality metric (GD, IGD, GD+, IGD+
or EMD), fixed reference front, seed list, baseline score for each
seed and reviewer/provenance metadata. Its existence and validity are
checked before running any optimizer. Declaring a reviewer in the
file is audit metadata, not proof of an external review.

### 5.1 Legacy v1

The `moeabench_performance_reference_v1` schema requires at least
**three distinct seeds** and explicit nonnegative absolute/relative
degradation thresholds. For each seed it computes the difference between
current and baseline scores, applies a one-sided paired t-test for a
mean increase, and requires both statistical significance and the
practical threshold to mark `FAILED`. Otherwise it reports `PASSED`
for that rule. This v1 `PASSED` means **no detected regression under
that particular rule**, not evidence of equivalence or noninferiority.
V1 remains available for compatibility, outside the release gate.

### 5.2 Conservative v2

The `moeabench_performance_reference_v2` schema additionally requires
a preregistered positive practical margin, `family_alpha`,
`target_power` (0.80–0.99), externally determined
`pilot_standard_deviation` of paired differences and a
`sample_size_rationale` for each case, plus detailed environment,
measurement and pilot provenance. There are at least eight distinct
seeds per case; a Student-t power calculation using the external
paired-variance pilot may require more, and the verifier rejects an
insufficient declared sample size. It refuses designs requiring over
5,000 paired runs instead of launching an unbounded study.

For each case, with paired differences
`d_i = current_metric(seed_i) - baseline_metric(seed_i)` and margin
`δ = max(absolute_limit, relative_limit × baseline_mean)`, the
protocol uses a Student-t interval with a conservative Bonferroni
familywise correction across the reference's specified cases:

| Result | Statistical interpretation |
|---|---|
| `PASSED` | Upper simultaneous confidence bound strictly below `δ`: evidence against **material mean degradation** |
| `FAILED` | Lower bound strictly above `δ`: evidence of **material mean degradation** |
| `INCONCLUSIVE` | The interval overlaps or touches `δ`: available data cannot establish either conclusion |

If any case fails, the aggregate status is `FAILED`; otherwise any
inconclusive case makes the aggregate `INCONCLUSIVE`. The CLI exits
with **0** only for `PASSED`, **1** for `FAILED` or
`INCONCLUSIVE`, and **2** for reference/validation errors. These
inferences concern mean *paired* changes under the statistical
assumptions of the chosen protocol, not equality of entire performance
distributions or a guarantee over all algorithms/problems.

The protocol's sample size and case count reflect explicit scientific
design and compute budgets. No performance baseline is created or
renewed by the verifier, and **statistical performance comparison is not
a mandatory 1.0.0 release/certification gate** until an independent,
reviewed reference and its assumptions are approved. See the complete
specification in [Performance stability](performance-stability.md)
and the issue [#98](https://github.com/monacofj/moeabench/issues/98).

## 6. CI architecture and expected evidence

Two regular workflows verify different aspects of the implementation:

**Unit contract regression checks**
([`.github/workflows/unit-contract-tests.yml`](../.github/workflows/unit-contract-tests.yml)):
a general-purpose unit/integration job executes the full unit scope,
empirical GT integration, smoke references, representative saved-experiment
generation and basic integration; a separate reference-numeric job runs
the **entire basic stability** suite in the pinned CPU/BLAS profile.
The workflow preserves numerical-drift diagnostics on failure.

**Stability verification separation**
([`.github/workflows/stability-separation.yml`](../.github/workflows/stability-separation.yml)):
the contracts job checks compilation, the stability-separation interface,
the independent fixed-metric oracle script, conventional numerical values,
the synthetic statistical v2 decision tests, all units, a small real
performance-adapter integration and numerical smoke. Its separate
reference-numeric job runs basic numerical stability and checks the
effective native-kernel configuration before doing so.

The CI tests for the **performance protocol** are synthetic API/logic
tests; they **do not** execute a large statistical optimization matrix.
Neither workflow silently omits the historical front tests to achieve
green status. A passing workflow is evidence for its actual selected
suites and runtime; it does not by itself create a new release-certified
manifest.

## 7. Developer certification and shipped manifest

The version manifest
[`moeabench/resources/certified_environment.json`](../moeabench/resources/certified_environment.json)
is a separately reviewed artifact containing Python, MoeaBench and selected
numerical dependency versions. It answers **version identity**, not native
kernel equivalence, stochastic distribution equality or universal
reproducibility.

To prepare a *new* developer-certified environment, explicitly enter
the designated numerical reference environment, confirm its kernel
preflight as in Section 4, and run:

```bash
python scripts/check_numerical_reference_runtime.py
python test.py --basic
python test.py --certify
```

`--certify` independently repeats the complete `--basic` battery
(unit, basic integration and basic numerical stability). It removes
inherited `PYTEST_ADDOPTS` in the certification subprocess so flags
such as `-k`, `--ignore` or `--deselect` cannot silently shrink the
required battery. A failing batch makes certification fail **without
writing** the version manifest. On success it writes
`moeabench/resources/certified_environment.json` using an intermediate
`.pending` file, and prints a message to review and commit the result.

**Important boundaries:** the preflight and matching CPU/BLAS settings
are currently enforced by the reference CI jobs; the standalone
`--certify` command itself checks the complete **test results**, not
a hardware/kernel fingerprint. A developer must establish the
designated reference profile before invoking it. It never
regenerates golden numerical fixtures and it never starts the optional
statistical performance test. The existing shipped manifest still
identifies a previously certified **0.16.1** environment; the
successful issue #92 CI checks did **not** run `--certify` or
approve a new version manifest for the next release.

From an installed package, the separate APIs answer narrower questions:

```python
import moeabench as mb

mb.system.verify_version()               # manifest version identity
mb.system.verify_stability()             # explicit PRNG/front smoke
mb.system.verify_experiment(exp)         # archived experiment provenance
mb.system.verification_status(exp)       # report of known states
```

The first `exp.run()` performs the inexpensive version check; loading
a persisted experiment compares provenance when available.
`exp.report()` reports cached verification states without initiating
costly numerical or performance checks. See
[Environment verification](environment-verification.md).

## 8. Troubleshooting and completion criteria

For a failing unit test, inspect the isolated contract, independently
derived expected value and relevant ADR first. For a deterministic
numerical-stability failure, check the captured Python/package versions,
RNG signature, seed and component-seed policy, and the *effective*
NumPy/BLAS dispatch before interpreting the result as a solver defect.
For a statistical performance failure or inconclusive result, verify
the reference's independent provenance, paired pilot variability,
preregistered margin, multiplicity correction and study power.

Do not equate any of these distinct results:

- `pytest` passes for a fixed mathematical input;
- a stored objective-space front replays in the controlled numerical
  environment;
- a statistical protocol provides evidence against material mean
  degradation;
- the installed version matches a historical developer manifest;
- a new release has been formally certified through `--certify`.

Completion requires the correct suite and its recorded environment,
reviewable independent expected values where applicable, no suppressed
historical failures, and the applicable accepted ADR contracts. The
original three-stage recovery is recorded by issues
[#95](https://github.com/monacofj/moeabench/issues/95),
[#96](https://github.com/monacofj/moeabench/issues/96),
[#97](https://github.com/monacofj/moeabench/issues/97),
[#98](https://github.com/monacofj/moeabench/issues/98) and
[#92](https://github.com/monacofj/moeabench/issues/92); their resolved
code and CI outcomes are separate from any future release-certification
approval.
