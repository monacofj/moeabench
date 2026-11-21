> **Historical research record (2026-11-21).** This analysis evaluates the pre-correction DTLZ8 implementation and experimental samplers. The production correction was subsequently merged in [PR #146](https://github.com/monacofj/moeabench/pull/146). Claims about the *incumbent* below refer to the former implementation, not current `main`. The original executable research scripts and 115-test workflow remain inspectable in [PR #144](https://github.com/monacofj/moeabench/pull/144).\n\n# DTLZ8 production migration decision — issue #119

**Status: historical implementation proposal; authorization granted and production correction completed in PR #146.**

## Decision requested

ADR 0057 (accepted, 2026-11-19, with the 2026-11-20 addendum)
authorizes DPF3/DPF5-specific implementation. Its broader on-demand
geometry-aware MOP contract is accepted in principle, but the dated
addendum does **not** authorize production DTLZ8 PS/PF replacement,
regression edits, new automatic calibration budgets, or historical
reference migrations. Before modifying `moeabench/mops/DTLZ8.py`,
the maintainer must approve a DTLZ8-specific implementation addendum
and the reference migration boundaries.

The research-only branch #144 contains no production changes. Its
analytical Pareto membership and independent feasibility checks reveal:
- M=2 legacy dynamic PS contains feasible off-front objectives.
- M=3/5/10 bundled static PS CSVs contain feasible off-front objectives
  (all examined rows); their 1,000-row cap also violates requests for
  2,000-point samples.
- M=4 dynamic PS objectives lie on the published front, but the
  square-grid truncation and duplicate seams violate requested
  cardinality and uniqueness.

## Narrow implementation proposal

1. Preserve `DTLZ8.evaluation()`, `_calc_constraints()`,
   `get_n_ieq_constr()`, constructor rules, `BaseMop.ps()`,
   `BaseMop.pf()`, and `mop.calibrate()` behavior; modify only
   DTLZ8's internal analytical PS generation once authorized.
2. Use geometry-specific deterministic finite samples:
   M=2 is the exact single-constraint line;
   M>=3 is the central line and M-1 analytical patches.
   The hybrid is the current **empirical first choice** for M>=3
   except M=4, pending the relative line/patch allocation decision.
   For M=4 a separate legacy-preserving, farthest-first addition
   approach keeps every analytically valid unique native point.
   In an eventual production design, avoid calling production
   `ps()` recursively: isolate the incumbent M=4 sampler as a
   deterministic helper if preservation is approved.
3. Keep the explicit requested point count exact when there are
   enough distinct analytical components. Define a clear error for
   genuinely underspecified budgets; do not silently return fewer.
   Do **not** modify the automatic n=2000 DTLZ budget or invent a
   universal density/error threshold. Keep public defaults initially
   at `ps(100)` and `pf(100)`.
4. Maintain the currently accepted DTLZ8 constructor behavior for
   `N >= M` and `N % M != 0`: warn and ignore trailing decision
   variables on evaluation. Constant-block objective inversions
   reconstruct the active `floor(N/M)*M` positions and fill ignored
   trailing positions with a deterministic in-bounds value (0.5).
   Independent new tests in #144 check round-trip objectives at
   representative M/N, including nonmultiples.
5. Preserve all external source contracts of `mop.calibrate()`
   (`source_gt`, `source_search`, `source_baseline`, force/load
   and sidecar persistence). No implicit cache, automatic historical
   sidecar replacement or export-side GT generation.
6. The M=3/5/10 legacy CSVs must be marked scientifically invalid
   as Pareto references in the migration plan. Do not silently delete
   these files or mutate frozen GT sidecars: identify their consumers,
   then replace or retire them by a separately approved versioned
   migration. Ordinary Git history preserves old content, but
   archived certificates keep their original immutable references.

## Independent acceptance gates

- Objective and decision mapping for M=2,3,4,5,10, plus arbitrary
  supported M and both divisible/nondivisible N.
- Published front membership, constraint feasibility, sample-wise
  nondominance, boundaries/extremes, line and patch presence.
- Exact returned number and unique objectives at 100, 128, 512,
  2,000 (plus error/edge budgets) without nondeterministic RNG.
- Component-wise independent finite-probe max and mean gaps,
  nearest-neighbor regularity, explicit line/patch budget allocation,
  runtime and peak memory. No choice from a pooled CV or a
  speculative canonical mixed-dimensional measure.
- Candidate indicator sensitivity: fixed experimental populations
  against old and new finite GTs, separately reporting IGD/IGD+,
  numerical reference differences, and any MOEA rank reversals.
  Reference changes alter IGD+ by construction. These cannot justify
  weakening historical tests.
- Full unit/integration and reference numerical-stability suite.
  Any expected frozen-reference failure is escalated as a migration
  decision, not bypassed or masked.
- ADR 0056 certification lifecycle and ADR 0057 implementation
  sign-off. Explicit maintainer approval **before** existing
  regression tests, frozen numerical references, baselines or ADRs
  are changed.

## Open scientific policy, not an implicit default

The hybrid (Halton-size line with lattice patches) was componentwise
nonworse than Halton in the observed three-probe comparison across
M=2,3,5,10 and budgets 128/512/2000. It does **not** dominate
the pure lattice in every patch (M=3 at 2,000 points is a concrete
counterexample). A final rule must state whether line resolution,
worst component covering radius, within-component regularity, or
some explicit multi-criterion policy governs source selection.
No comparison against the defective static CSVs may be used to
certify the continuous Pareto truth.

## Related independent ADR issue: DPF4 PR #143

ADR 0057's November 20 addendum states that ordered chaotic
thresholds were authorized for DPF3 specifically, with no silent
change to DPF4 chaotic projection behavior. Draft PR #143 sorts
DPF4 projection thresholds, albeit without mutating the shared
pool. Treat this as a separate ADR-clarification / explicit
authorization gate; passing CI is not an architectural override.

**Decision required:** authorize a DTLZ8-specific ADR 0057 addendum
covering production PS/PF correction under these constraints,
without authorizing changes to existing tests, baselines, calibration
defaults or certificate reference artifacts. A separate, explicit
migration decision is needed for any of those protected assets.
