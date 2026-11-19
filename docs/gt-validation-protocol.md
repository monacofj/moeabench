<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Analytical GT validation protocol — issue #119

**Status:** experimental, read-only diagnostic pilot. No GT replacement is
approved by this document. Original MOP evaluation equations are immutable
unless separately reviewed. Issues: [#119](https://github.com/monacofj/moeabench/issues/119)
and [#65](https://github.com/monacofj/moeabench/issues/65).

## Approved GT interface and budget direction (ADR 0057)

ADR 0057 now records the accepted architectural direction (not yet a
runtime implementation):

- `mop.ps(n_points)` and `mop.pf(n_points)` remain generators of
  mathematical Pareto-set/front samples, with the existing public roles.
  Do not add calibration side effects or a `mode="calibrate"` switch.
- `mop.calibrate()` retains the MOP-owned reference, external/search
  source, load/reload, persisted sidecar and `calibrated_gt` contracts.
  If no budget is explicitly requested **and new generation is needed**,
  a future geometry-/quality-/cost-aware policy will select it.
  Calibrate should not regenerate when an existing source is loaded.
- A visualization budget is independent of scientific GT resolution.
  Persisting the *actual GT used* in `exp.save()` is separate integration
  work subject to existing selective persistence contracts.
- No general implicit on-disk cache for every `pf()` call; no fixed
  precomputed catalog for arbitrary M/K/D.
- The existing 2,000 DTLZ / 10,000 DPF runtime calibration defaults
  remain **unchanged** until the numerical replacement and migration
  policy are independently verified and authorized.
- Do not change archived GTs, frozen numerical references or regression
  tests without prior maintainer warning/approval.

**Research next:** derive a defensible default budget rule from
intrinsic geometry, coverage-error convergence and bounded runtime;
test MOP-specific deterministic samplers and distinguish exact
geometric checks from finite dense-reference diagnostics. Report
actual returned cardinality, duplicates, extremes and relative
errors as `M`/intrinsic dimension and budgets vary. Changes to
`calibrate()` occur only after approval of the concrete algorithm.

## Distinct validity questions

1. **Pareto membership and completeness.** Establish the true feasible
   nondominated locus from mathematics or independent references. A set
   may have no internal dominance and nevertheless omit entire Pareto
   components (DTLZ5/6 M>=4 is a known example). Norm or objective-sum
   identities demonstrate *surface membership*, not Pareto completeness.
2. **Coverage in objective space.** Measure distances from a separate,
   high-resolution geometrical reference to the candidate finite GT.
   Report mean, 95th percentile, maximum; inspect missing components,
   boundary extremes and disconnected regions independently.
3. **Spatial regularity.** Report nearest-neighbor distance distribution
   and coefficient of variation (CV), along with duplicate fraction.
   Low CV does **not** imply good global coverage (a dense cluster can
   have very regular spacing); always interpret with covering radius.
4. **Sampling law and representativeness.** Define the objective-space
   geometrical measure, not just a uniform distribution over decision
   parameters. A probability-uniform *random* sample is not necessarily
   regular; regularity requires a separate finite-sample check. For
   mixed intrinsic dimensions, report each component separately and
   seek explicit scientific authorization for a cross-component weight.
5. **Reproducibility and convergence.** Verify same seed/budget yields the
   same ordered points; compare independent references/seeds and fixed
   budgets; assess indicator sensitivity with *unchanged* MOEA outputs.

## Pilot (DTLZ1–4)

Run from project root after installing the project and SciPy:

```bash
python scripts/audit/gt_geometry_pilot.py --budget 256 --budget 1024 --reference-size 4096 --objectives 3 5
python -m pytest tests/unit/test_gt_geometry_pilot_issue119.py -q
```

The pilot deliberately draws independent objective-space references:

- **DTLZ1:** normalize independent exponential variables, then scale by
  0.5. This is uniform in simplex `sum(f_i)=0.5` with respect to its
  intrinsic area measure, in any `M>=2`.
- **DTLZ2/3/4:** normalize absolute standard Gaussian vectors. This is
  uniform in surface area of the positive unit hypersphere.

Reference samples use a *different* RNG (`default_rng`) and seed from
MOP `ps()` (historically `RandomState(42)`). Their finite size means
measured distances are estimators, **not** exact continuous-surface
Hausdorff bounds. Different reference seeds should be compared. For
DTLZ1, objective-space surface-uniform sampling is **not** equivalent
to the current uniform decision-parameter sampling. This is a
hypothesis to measure, not permission to rewrite the canonical sampler.

The output is deterministic JSON with explicit sample budgets,
independent reference size and seed, objective-space bidirectional
nearest distances, finite-sample neighbor spacing, geometric residuals,
uniqueness and extrema coverage. No diagnostic threshold is
misrepresented as a mathematical theorem or CI acceptance gate.

## Controlled farthest-point candidate experiment

A second **read-only** diagnostic compares the current actual MOP
`pf(n_points)` with an independent objective-space greedy
farthest-point selection (FPS) candidate. It uses the same finite-budget
quality diagnostics and a third, independently seeded diagnostic
reference; no candidate is chosen from that diagnostic reference.

```bash
python scripts/audit/gt_fps_comparison.py --objectives 3 5 --budget 256 1024
python -m pytest tests/unit/test_gt_fps_comparison_issue119.py -q
```

FPS initializes at all analytical objective-axis extrema, then
selects the candidate with the largest current nearest-sample
distance. Its `--candidate-pool`, `--reference-size` and seed
are explicit and its output is reproducible. The candidate is
constructed in objective space *only*: this study does not propose a
replacement for MOP `ps()`, which must return decision vectors
mapping to the chosen front. Constructing valid decision-space
preimages remains a separate correctness obligation.

Importantly, the classical 2-approximation guarantee for finite
metric k-center applies relative to **the supplied finite candidate
pool**, with the algorithm's seed-center constraints accounted for;
it is **not** a certified bound on covering an entire continuous
Pareto front. Runtime measurements are exploratory and system
dependent. The independent finite reference's worst nearest distance
is only a lower bound on the continuous covering radius; no pass/fail
claim may be derived from it alone. The experiment must be repeated
with candidate pool/reference sizes and dimensions varied before
selecting a production generator.

## Extension and acceptance criteria

For DTLZ7/DPF2 use the *independently derived exact admissible Pareto
regions*, then assess per-region coverage/spacing. For DPF3/4/5, do not
treat filtering nondominated points **within the sample** as proof of
true global Pareto optimality. For DTLZ8/9, compare against analytical
feasibility and geometry including disconnected or constrained
boundaries. For original DTLZ5/6 M>=4, **#65 must first characterize
the complete PF including `g>0`** before a purported uniform sample
can be certified. A provisional subset is not a GT oracle.

Before promoting a new generator: present per-MOP independent
membership/completeness evidence; cover all known components/extremes;
report finite-budget spacing/covering and convergence on multiple
references; compare IGD/IGD+/GT-based HV using fixed fronts; verify
determinism and operational cost; document regression and certification
impact. A single favorable summary statistic is insufficient.

**ADRs and test safety:** ADR 0006 governs native analytical PS/PF and
ADR 0018 currently fixes default calibration densities at 2,000 DTLZ /
10,000 DPF. This pilot changes neither. Any incompatible modification
to an accepted ADR requires explicit maintainer authorization. Warn
and obtain authorization before editing existing regression tests,
canonical GT files, numerical baselines or certification records.
Historical data remain traceable; mathematically incorrect GTs should
ultimately be replaced as current reference data after review.
