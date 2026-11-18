<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench Test Architecture

MoeaBench organizes testing along a **scope** axis and a scope-specific **level** axis.

The available scopes are:

- `unit`
- `integration`
- `stability`

Execution policy:

- selecting a scope runs only that scope
- omitting the scope runs all scopes

The level axis is intentionally not perfectly symmetric across scopes:

- `unit` has a single effective reach: total
- `integration` has two effective reaches: `smoke` and `basic`
- numerical `stability` has two active reaches: `smoke` and `basic`
- statistical `performance-stability` uses an explicit external reference rather than a level flag

The runner still uses a common marker vocabulary for implementation convenience, but the semantic meaning of the levels depends on the selected scope.

## Scope

### `unit`

Local contract tests.

These protect:

- public API surface
- object contracts
- local mathematical invariants
- persistence helpers
- metadata rules
- narrow analytical helpers

Operational rule:

- `unit` always runs the full unit suite
- level flags are accepted by the CLI but ignored for execution

### `integration`

Canonical workflow tests across namespaces.

These protect:

- composition between `metrics`, `stats`, `view`, and persistence
- headless canonical walkthroughs
- view dispatch and GT inference behavior
- correlated result/view pipelines

Integration has two meaningful reaches:

- `smoke`: central pipelines, typically one representative pipeline per major domain
- `basic`: all integration pipelines and their richer variations; this is the total integration reach

### `stability` / `numeric-stability`

Numerical and deterministic stability tests. `--stability` is retained as an alias of `--numeric-stability`.

Stability protects two distinct scientific contracts:

- **Analytical stability**: given frozen inputs such as a front and a ground truth, the current metrics must still reproduce the certified numerical outputs.
- **Algorithmic stability**: given a canonical `(MOP, MOEA, seed, budget)` configuration, the current solver stack must still reproduce the certified final front.

These two contracts are independent:

- metrics may drift while the produced front stays unchanged
- the produced front may drift while the metrics remain internally correct

In the current FAIR-free recovery, `--numeric-stability` directly exercises frozen PRNG signatures, GT geometry and seeded solver fronts. Small independent conventional-indicator oracles run in the `unit` scope; a full `python test.py --basic` includes both. Broader permanent frozen outputs for conventional metrics remain a separately audited coverage gap, and a passing standalone `--numeric-stability` call must not be described as exhaustive metric certification.

## Levels By Scope

### `smoke`

Fast sanity coverage.

Expected usage:

- default local validation
- quick guard against catastrophic drift
- representative sentinels with breadth across the selected scope

Meaning by scope:

- `unit`: equivalent to total unit coverage
- `integration`: central pipelines, ideally one per major domain
- `stability`: deterministic `N=1` checks with light budgets, covering the canonical `(MOEA, MOP)` matrix and a smoke subset of analytical checks

### `basic`

Normal development coverage.

Expected usage:

- routine validation before merge
- broad stability verification
- fuller integration coverage

Meaning by scope:

- `unit`: equivalent to total unit coverage
- `integration`: all integration pipelines and richer variations beyond the central smoke set
- `stability`: everything in smoke, plus full analytical stability checks and comparisons against high-resolution analytical ground truth when such truth is available in closed or directly sampled analytical form

### Statistical `performance-stability`

This opt-in assessment runs only the MOP/MOEA/seed cases specified by a separately reviewed, versioned JSON reference. It compares paired performance observations against externally recorded results and uses both statistical significance and a predefined practical degradation threshold. The reference supplies the fixed objective-space front used for quality measurement. It never creates or refreshes its own baselines. Its check does not run automatically after a numerical failure.

The historical `deep` statistical test has a missing baseline and is excluded from the active execution path. The runner rejects `--stability --deep` and all-scopes `--deep` rather than implying that a skipped test passed. `--unit --deep` remains a synonym for the full unit scope.

## Commands

Use the central orchestrator:

```bash
python3 test.py
```

Default behavior is:

```bash
python3 test.py --smoke
```

Examples:

```bash
python3 test.py --unit --smoke
python3 test.py --integration --basic
python3 test.py --numeric-stability --basic
python3 test.py --performance-stability --performance-reference /path/to/reviewed-performance.json
```

For `unit`, all of the following are equivalent and run the complete unit suite. In runner output, this batch is shown simply as `UNIT`:

```bash
python3 test.py --unit
python3 test.py --unit --smoke
python3 test.py --unit --basic
python3 test.py --unit --deep
```

For `integration`, only `--smoke` and `--basic` are valid because this scope has only two meaningful reaches:

```bash
python3 test.py --integration --smoke
python3 test.py --integration --basic
```

To inspect the current suite classification:

```bash
python3 test.py --list
python3 test.py --list --integration
python3 test.py --list --numeric-stability --smoke
```

The listing uses pytest node ids such as:

```text
tests/integration/test_view_dispatch_pipeline.py::test_view_dispatch_pipeline
```

and annotates each collected test with its exact level.

## Developer environment certification

The testing scopes above also underpin the environment verification system
([ADR 0050](adr/0050-certified-environment-verification-and-provenance.md)).
For release preparation, run the entire suite and then certify the developer
reference environment:

```bash
python test.py --basic
python test.py --certify
```

`--certify` independently executes the complete basic battery (all unit,
basic integration, and basic numerical stability checks) before updating the shipped
`moeabench/resources/certified_environment.json`. The certification subprocess
ignores inherited `PYTEST_ADDOPTS` to prevent environment-supplied selection flags
(such as `-k`, `--ignore`, or `--deselect`) from silently reducing that battery. If any test
fails, the manifest is **not** updated. Review and commit changes to the
manifest; do not revise numerical fixtures simply to make a failing
environment pass. An explicitly skipped test remains recorded as skipped,
and does not count as a passing observation.

This developer command is different from the installed-package API:

```python
import moeabench as mb

versions = mb.system.verify_version()       # version identity, not numerical proof
stability = mb.system.verify_stability()    # explicit PRNG/front smoke checks
```

The public stability check tests only its packaged signatures and front
fixtures, whereas `--certify` runs the broader development suite. The
certified environment may be more narrowly specified than an arbitrary CI
runner; equal seeds and dependency versions alone cannot guarantee bitwise
identity of NSGA-III trajectories across runtimes. Functional dependencies
such as Matplotlib are covered by integration tests, not entered in the
numerical-version manifest. See
[Environment verification](environment-verification.md) for interpretation.

## Marker Policy

All official tests must carry exactly one scope marker and exactly one level marker.

The active marker vocabulary is:

- `scope_unit`
- `scope_integration`
- `scope_stability`
- `level_smoke`
- `level_basic`
- `level_deep`

These markers are the normative classification mechanism of the suite.

Implementation note:

- `unit` tests are encoded with level markers for tooling uniformity, but semantically the unit scope is total-only
- `integration` uses `level_smoke` and `level_basic`
- numerical stability uses `scope_stability` with `level_smoke` and `level_basic`; the old `level_deep` file is retained for historical investigation but is not an active certification path
- performance stability uses an explicit reference-backed runner, not the numerical pytest marker tiers

## Stability Data

MoeaBench treats frozen calibration artifacts and deterministic front baselines as part of the scientific contract.

In particular:

- analytical stability uses frozen reference inputs and frozen expected outputs
- algorithmic stability uses frozen final fronts produced by canonical `(MOP, MOEA, seed, budget)` configurations
- `basic` stability may additionally compare produced fronts against high-resolution analytical ground truth when the problem admits such a truth
- performance stability uses a reviewed independent reference and paired stochastic measurements from selected MOP/MOEA/seed configurations
- agreement down to 6 decimal places is the default certification floor unless a stricter pathway contract applies

## Notes

- `smoke` should remain fast enough for default local use.
- `basic` should provide broad confidence without becoming prohibitive.
- the historical `deep` statistical case currently lacks its baseline and is non-operative; it is not a release-certification gate. Performance stability requires a separately reviewed reference and explicit invocation (ADR 0050).
- `--performance-stability` is explicitly opt-in; a missing or unreviewed reference is an error, not a pass.
- the key scientific distinction inside `stability` is always twofold: numerical stability of the measurements and numerical stability of the produced fronts.
- when no scope is provided, the runner executes `unit`, `integration`, and numerical `stability` at smoke or basic level; `--deep` is rejected for the numerical/all-scopes path.
