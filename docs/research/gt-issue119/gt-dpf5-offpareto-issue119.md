<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# DPF5 D=2 nonzero-g equation verification and optional-test isolation — issue #119

**Research-only.** No changes to MOP evaluation, PS/PF/calibrate, historical GT or existing regression fixtures.

## CI collection failure and isolation correction

Following commit `93929a3`, the general unit and numerical stability workflows failed **during test collection**, before scientific tests ran: the newly introduced high-precision research module imported `mpmath` at import time, while the standard dependency environment does not include that optional package. Research-specific CI installed mpmath and passed, which masked this integration defect.

The independent new high-precision script now imports `mpmath` *inside* its precision function; the two newly introduced optional-precision tests call `pytest.importorskip("mpmath")` locally. Other independent DPF tests still execute without mpmath. No pre-existing regression tests, existing workflows, or core dependencies were changed for this fix. All general CI must be rerun on final branch revision before any merge.

## Actual evaluator with nonzero g, fixed D=2

Independent in-memory equation study: [PR #139](https://github.com/monacofj/moeabench/pull/139), [research CI #44631874652](https://github.com/monacofj/moeabench/actions/runs/44631874652) **PASS 31 new tests**, including 10 new off-front DPF5 tests; JSON archived by workflow. The study constructs MOP-valid box decision points with `x_M = x_1 ± 0.2`, other distance coordinates 0.5, so **g=0.04**, and independently predicts `F=(1+g)*F_beta_or_gamma` at the exact same decisions (strict P/D interiors; threshold equality excluded). Both requested n=128/512 were run for M=3/4/5/6/10.

The *scaled spherical identity* is `sum(F_i²)/(1+g)²=1`. Measured maximum absolute scaled squared-norm error in the **P branch**:

| M | Current production P | Independent D=2 P |
|---:|---:|---:|
| 3 | 4.44e-16 | 4.44e-16 |
| 4 | 4.44e-16 | 4.44e-16 |
| 5 | 0.0776441 | 4.44e-16 |
| 6 | 0.1792815 | 4.44e-16 |
| 10 | 0.4933417 | 4.44e-16 |

For M>=5, maximum current-vs-independent component differences at g=0.04 were 0.0941594, 0.1703359 and 0.3547909 for M5/6/10 respectively. **D branch** current evaluation continued to match independently rebuilt gamma objectives to float precision and satisfied the scaled identity at all tested M.

**Interpretation:** dimension-dependent P objective mismatch persists at g>0; it is not merely caused by PS generation, nondominance filtering, or a g=0-specific implementation branch. The test is restricted to *D=2*, the previously independently verified geometry. It does not claim global PF membership of these off-front decision points and does not justify extrapolation to D>2.

## Pending gates

- Independently reconstruct published beta/gamma definitions for D>2 and verify evaluation beyond the D=2 sphere case; do not reuse a formula without proof.
- Perform objective-space membership and coverage analysis for Pareto-optimal support, especially the DPF5 conditional boundary and the DPF3 min/max projections.
- Enumerate affected conventional indicators, archived GTs and immutable baselines before proposing a production repair.
- Warn and obtain explicit approval before changing any accepted ADR or **existing regression tests**. No such change has been made.

This document is not a release certification or acceptance of changed numerical defaults.
