<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Finite-reference Hypervolume exceedance: independent reproducer (#89)

**Status:** research evidence; no metric contract, reference, or policy change.

## Mathematical counterexample

Consider the *minimization* DTLZ1 front with two objectives and `g=0`:
`f_1 + f_2 = 1/2`, with both objectives nonnegative. Take an
intentionally sparse finite GT `G = {(0, 0.5), (0.5, 0)}` and an
evaluated point `P = {(0.25, 0.25)}`. All three points are on the
**same true Pareto front**; none is superior to the continuous
mathematical front.

Under an external normalization context given by `G`, the
ideal/nadir are `(0,0)` and `(0.5,0.5)`. MoeaBench's conventional
exact HV reference point is `(1.1,1.1)` in these normalized
coordinates (ADR 0020). Therefore:

- `HV(G) = (1.1)(0.1) + (0.1)(1.1) - (0.1)(0.1) = 0.21`.
- `HV(P) = (1.1 - 0.5)^2 = 0.36`.
- `H_abs(P; G) = 0.36/0.21 = 12/7 ≈ 1.714285714`.

The result follows from finite sampling alone, with an **exact**
two-objective HV calculation; Monte Carlo sampling error, evaluation
error and erroneous mathematical front membership are not required.

This is a deliberately low-budget counterexample, **not an assertion
that a standard production GT budget would generate this degree of
exceedance**. The quantitative frequency and magnitude of exceedance
at normal MOP-specific densities remain an empirical research question.

The identical raw denominator with `scale="rel"` and an external
competitive reference also gives `12/7`: by ADR 0047 this is valid,
because a competitive reference is not a theoretical ceiling. The
shared numerical value does **not** give the two scales the same
interpretation. With `scale="abs"`, the denominator is the MOP-owned
calibrated GT; the optional `ref` controls geometry only.

## Why projection alone is not a remedy

Projecting the evaluated midpoint in this example onto the **continuous**
true Pareto front is the identity map: `P` is already on that front.
The resulting `H_abs > 1` persists because the denominator `G`
under-samples the front. Projecting only onto the *finite* sampled GT
instead merely replaces the evaluated solution with a comparison
against a selected discrete sample and changes the metric. Neither
projection operation establishes a certified continuous-front HV upper
bound.

An improvement must distinguish a verified mathematical upper bound,
an estimator with a quantified numerical error, and a contextual
finite-reference ratio. For disconnected, mixed-dimensional,
degenerate and unavailable analytical fronts, an arbitrary Euclidean
projection also needs independent feasibility and Pareto-membership
justification.

## Evidence and reproducibility

A new, independent test file
`tests/unit/test_hv_finite_reference_issue89.py` checks the existing
public exact-HV behavior and verifies `H_abs > 1` and `H_rel > 1`
without editing any old regression test, fixture or baseline. Run:

```bash
python -m pytest tests/unit/test_hv_finite_reference_issue89.py -q
python test.py --unit
```

These are proposed verification commands; a passing CI run is required
before claiming source-bound execution success.

Governing ADRs: 0019, 0020, 0035 and **0047** (relative-HV semantics);
0042–0043 (diagnostics and engines); 0018 (reference densities).
The existing `H_abs = H(P) / H(GT)` implementation and its regression
guards remain unchanged. Candidate adaptive GT densification, certified
bounds, uncertainty reporting, and user-facing API policies **require
independent evidence and an approved architectural decision**, not
silent clipping, altered historical fixture values or automatic
recalibration.

## Further empirical work (still open)

Evaluate the frequency and magnitude of exceedances across
DTLZ/DPF fronts, accepted calibration budgets and multiple fixed seeds;
exact versus Monte Carlo backends; shared versus external normalization
contexts; sensitivity to population and GT size, distribution and
potentially disconnected front geometry. Analyze whether candidate
correction approaches change rankings before adopting a policy.
