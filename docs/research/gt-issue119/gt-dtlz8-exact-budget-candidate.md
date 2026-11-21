# DTLZ8 geometric reference candidate — issue #119

Status: **research only; no production, ADR, regression fixture or baseline changes**.

## Source and objective-space characterization

Deb, Thiele, Laumanns, Zitzler, *Scalable Test Problems for Evolutionary Multiobjective Optimization*, Section 8.8, equation (28), gives for minimization (M >= 3):

- `f_M + 4 f_i >= 1` for each `i < M`;
- `2 f_M + min_{i<j<M}(f_i+f_j) >= 1`;
- `0 <= f_i <= 1`.

For **M=2**, the published pairwise constraint is absent. The sole
constraint gives a complete one-dimensional front
`f_1=(1-t)/4`, `f_2=t`, `0<=t<=1`, sampled independently with
both endpoints. For **M>=3** the geometry below applies.

Set `t=f_M`, `a(t)=(1-t)/4`, `b(t)=1-2t`.
For `1/3 <= t <= 1`, the nondominated central line has
`f_1=...=f_{M-1}=a(t)`.
For `0 <= t < 1/3`, the second inequality is active. The first
`M-1` coordinates, sorted increasingly, must have two smallest
coordinates totaling `b(t)`. An efficient point with a unique
smallest index `k` has `f_k=low`, all `f_j=high=b(t)-low`
for `j!=k`, and `a(t)<=low<=b(t)/2`. These are M-1 patches,
which meet along their common equal-coordinate edge. For M=3
they form two triangular halves; the original publication calls
the objective-space component a line plus a hyper-plane.

The existing `DTLZ8.ps()` has a 1,000-row static-data ceiling for
M=3/5/10, and its dynamic square-grid truncation can return too
few and duplicated objective vectors. Those are **finite-sampling
defects**, independent of the original objective evaluator.

## Candidate policy

The isolated module `scripts/audit/gt_dtlz8_candidate_issue119.py`
constructs the exact intended objective-space components and
inverts the block means into constant-block decision vectors.

For M=2 and `n>=2`, the candidate samples the single line directly.
For M>=3 and a requested budget `n >= M+1`, it allocates
`max(2, floor(n/M))` rows to the central line and distributes the
remainder as evenly as possible across the M-1 patches.
Line points include both endpoints. Every patch includes its t=0
low/high extremum, and the common t=0 equal-coordinate seam is
included once when budget permits. Remaining patch points use
deterministic base-2 and base-3 radical inverses strictly inside
the parameter rectangle, avoiding interface duplicates. It yields exactly n
distinct objective vectors and deterministic replay for supported M.
As a second research candidate, `patch_strategy="lattice"` uses a
deterministic, approximately square, parameter-space lattice for the
remaining patch interiors, retaining the same anchors and budgets.
This may have better small-budget covering properties, but is not
assumed to be uniform in objective geometry. Both variants are
compared independently with the incumbent.

This allocation is **only an explicit component-presence policy**.
It is *not* a canonical measure, uniform objective-surface density,
or proof of optimal finite-sample spacing.

The accompanying new independent test module checks the requested
cardinality, objective-to-decision inversion, published constraints,
presence of every patch, central-line endpoints and deterministic
replay for M=2,3,4,5,10 and budgets up to 2,000.

## Independent analytical Pareto membership

The module `scripts/audit/gt_dtlz8_membership_issue119.py`
checks each incumbent static/dynamic GT objective vector against
the published piecewise Pareto equations, with a documented 1e-10
float64 diagnostic tolerance. This is stronger than checking only
nondominance within a finite sample. The audit independently checks
published constraint feasibility, distinguishing infeasible rows from
feasible rows that lie above the actual Pareto front. It separately reports counts,
unique points, residual magnitudes and violations for M=2,3,4,5,10
at n=128,512,2000, alongside the candidate variants. It does not
modify the meaning of the public `feasible` flag.

An initial reproducible run at n=128,512,2000 found **all** examined
static reference objectives (M=3,5,10) outside the published analytical
Pareto geometry by more than 1e-10. The dynamic M=4 objectives were
on-geometry within float64 rounding, but lacked the requested unique
count. For M=2 the legacy fallback also returned non-Pareto rows.
This evidence **disqualifies blindly augmenting or preserving the static
files** as canonical GT; individual feasibility checks and the source
provenance must be examined before a production replacement. These
negative results are retained in the research record.

A concrete first static M=3 example evaluates to approximately
`(0.09595176, 0.19891921, 0.62134608)`. It is feasible, but at
the same final objective its analytical line counterpart is
`(0.09466348, 0.09466348, 0.62134608)`, which strictly dominates it.

## Independent finite-probe comparison

The standalone script
`scripts/audit/gt_dtlz8_component_comparison_issue119.py`
compares the incumbent against this candidate at identical requested
budgets. It generates **separate** objective-space probes for the line
and each patch using radical-inverse bases 5 and 7 (independent of
the candidate's bases 2 and 3), with separate deterministic
irrational rotations for each probe seed and records per-component maximum and
mean nearest-neighbor distances plus positive-part IGD+ distances
for the incumbent, Halton candidate and structured lattice candidate.
Suggested run:

```sh
python -m scripts.audit.gt_dtlz8_component_comparison_issue119 --objectives 2 3 4 5 10 --budgets 128 512 --seeds 11941 11942 11943 --probes 256 > dtlz8-component-comparison.json
```

The first trial (before decorrelating probe seeds) revealed a **negative
result** that must not be suppressed: at M=4, n=512, the incumbent
largest observed gap on each patch was about 0.0324 whereas the
candidate was about 0.0473. At M=4, n=128, corresponding figures
were about 0.0692 and 0.0938. Therefore the current proposed
sampler cannot be promoted to production as a general replacement.
The revised three-rotation experiment must be reviewed separately,
and candidates should retain or improve the incumbent's within-patch
coverage before scientific acceptance.

These are explicitly **finite-probe** diagnostics, not continuous
Hausdorff certificates. No line/patch aggregate or canonical
mixed-dimensional probability weighting is imposed; the report
must be reviewed component by component. New smoke tests check
probe construction and diagnostic invariants.

## M=4 completion control: guaranteed nonworsening finite-set covering

Because the dynamic M=4 sampler is independently verified to produce
valid analytical Pareto objectives (but truncates and duplicates them),
`m4_preserving_completion()` provides a **research-only** comparison:
it retains every exact-unique objective in original order and greedily
selects the next **objective-space farthest** point from a deterministic
lattice candidate pool until the requested size is reached. The original
row-order append was rejected because it predominantly added line points
instead of improving the sparser two-dimensional patches. Any off-front incumbent point causes refusal rather than reuse.

Provided the incumbent contains at most `n` distinct valid points,
the completed point set contains the incumbent unique set as a subset.
Consequently **for every independently chosen query point** the
nearest Euclidean distance to the completed set cannot exceed the
distance to the incumbent set. The same monotonic property applies
to the positive-part nearest distance used in IGD+ when the reference
points and normalization are fixed. New tests directly assert this
superset property and numerical nonworsening at n=128,512.

This is *not* a proof of improved spatial regularity, and retains some
incumbent clustering. Farthest-first addition targets the largest gaps
of the finite candidate pool but does not certify continuous covering
or guarantee a lower component-specific nearest-neighbor CV. It also **cannot** be used for the faulty
M=2,3,5,10 sources: off-front points are scientifically ineligible
for preservation. Treat this as a controlled feasibility and coverage
baseline, not the finally chosen sampler.

## Spatial regularity and reference-sensitive indicator audit

A further independent script,
`scripts/audit/gt_dtlz8_spacing_sensitivity_issue119.py`,
now reports component-specific nearest-neighbor mean, maximum and
coefficient of variation for **distinct on-front points**. Off-front
points are counted separately, rather than distorting geometric
regularity on a false Pareto reference. Components with fewer than
two unique points are explicitly marked as undefined.

For each finite reference (native, Halton, lattice and the M=4
incumbent-preserving completion), it measures **conventional
positive-part IGD+** for three fixed, independently generated
analytical probe populations: line-only, one-patch-only, and a
population with equal point counts per component. The latter is
an explicitly declared *finite-record sampling policy*, not a
canonical measure of a mixed-dimensional Pareto front. Every
number is indexed by reference, population, objective dimension,
budget and independent probe seed. No intercomponent summary
regularity or probability measure is inferred.

Use the deterministic research workflow artifact
`dtlz8-spacing-igd-sensitivity.json` to inspect whether a better
finite-reference covering gap also preserves local spacing and
whether changing the reference alters numerical scores or relative
rankings of the fixed analytical populations. Improvements in
reference sampling need not preserve numerical IGD+ values; that
is a scientifically meaningful migration impact, not grounds
to silently rewrite a frozen baseline.

## Remaining scientific checks before production

1. Independently prove the exact nondominated characterization,
   including the equality edges and M=3 symmetry, against the
   published constraint geometry.
2. Run and review independent component-wise covering gaps,
   nearest-neighbor spacing and discrepancies at higher resolutions;
   examine the three predetermined probe offsets for M=2,3,4,5,10
   and ensure any claimed improvement holds for each relevant region.
3. Study how line-versus-patch allocation affects IGD and IGD+,
   recognizing their finite-reference weights; do not silently
   equate a one-dimensional line with an area measure.
4. Record actual 2,000-point calibration behavior, computational
   cost and any downstream numerical-reference differences.
5. Review ADR 0006, 0018, 0057 and 0056 before changing production
   or replacing historical reference artifacts. Explicit maintainer
   authorization is required for an ADR-incompatible change or
   edits to existing regression tests, fixtures and baselines.

The M=2 boundary is deliberately unsupported by this candidate:
the production special-case constraint implementation does not have
the published M>=3 pairwise expression. It needs a separate audit.

**Source:** Deb et al., original report:
https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
