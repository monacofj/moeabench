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
