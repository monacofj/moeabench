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
- `stability` has three effective reaches: `smoke`, `basic`, and `deep`

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

- composition between `metrics`, `stats`, `clinic`, `view`, and persistence
- headless canonical walkthroughs
- view dispatch and GT inference behavior
- correlated result/view pipelines

Integration has two meaningful reaches:

- `smoke`: central pipelines, typically one representative pipeline per major domain
- `basic`: all integration pipelines and their richer variations; this is the total integration reach

### `stability`

Numerical and deterministic stability tests.

Stability protects two distinct scientific contracts:

- **Analytical stability**: given frozen inputs such as a front and a ground truth, the current metrics, clinical diagnostics, FAIR metrics, and Q-scores must still reproduce the certified numerical outputs.
- **Algorithmic stability**: given a canonical `(MOP, MOEA, seed, budget)` configuration, the current solver stack must still reproduce the certified final front.

These two contracts are independent:

- metrics may drift while the produced front stays unchanged
- the produced front may drift while the metrics remain internally correct

For that reason, `stability` always concerns both:

- whether the measurements still match
- whether the produced front still matches

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

### `deep`

Expensive, high-rigor coverage.

Expected usage:

- release preparation
- calibration renewal
- statistically rigorous revalidation

Meaning by scope:

- `unit`: equivalent to total unit coverage
- `integration`: not applicable
- `stability`: everything in `basic`, plus empirically determined reference regimes where the reference truth is itself calibration-derived rather than analytically available

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
python3 test.py --stability --deep
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
python3 test.py --list --stability --smoke
```

The listing uses pytest node ids such as:

```text
tests/integration/test_view_dispatch_pipeline.py::test_view_dispatch_pipeline
```

and annotates each collected test with its exact level.

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
- `stability` is the scope where the full `smoke/basic/deep` distinction carries scientific meaning

## Stability Data

MoeaBench treats frozen calibration artifacts and deterministic front baselines as part of the scientific contract.

In particular:

- analytical stability uses frozen reference inputs and frozen expected outputs
- algorithmic stability uses frozen final fronts produced by canonical `(MOP, MOEA, seed, budget)` configurations
- `basic` stability may additionally compare produced fronts against high-resolution analytical ground truth when the problem admits such a truth
- `deep` stability may extend the contract to empirically determined reference regimes built from calibration artifacts or MOEA-derived reference procedures
- agreement down to 6 decimal places is the default certification floor unless a stricter pathway contract applies

## Notes

- `smoke` should remain fast enough for default local use.
- `basic` should provide broad confidence without becoming prohibitive.
- `deep` may require explicit opt-in or extra artifacts and is not intended for routine execution.
- the key scientific distinction inside `stability` is always twofold: numerical stability of the measurements and numerical stability of the produced fronts.
- when no scope is provided, the runner executes `unit`, `integration`, and `stability` at the requested level; in this all-scopes mode, `integration` participates up to `basic`, so `python3 test.py --deep` still runs integration with its full `basic` reach.
