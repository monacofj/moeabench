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
