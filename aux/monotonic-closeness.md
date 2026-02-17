## Change request: make CLOSENESS monotone w.r.t. “worse than blur” (no score increase beyond AnchorBad)

### Problem
Current `q_closeness` is computed as:
- `q = d_bad / (d_bad + d_ideal)` with `d_bad = W1(U, B)` and `d_ideal = W1(U, I)`

This **can increase** when the solution becomes *worse than the blur baseline* (moves further away from `B`), which violates the intended semantics:
> If the result is worse than `GT_blur` (AnchorBad), CLOSENESS must **not** be higher than when it equals `GT_blur`.

Concrete 1D counterexample:
- Ideal `I=0`, Blur `B=5`.
- If `U=5` (equals blur): `q=0`.
- If `U=10` (worse than blur): `q=0.33` (increases).

### Required property
Let `I` be the ideal distribution (degenerate at 0 in `u=d/s` units) and `B` be the blur baseline distribution.
Define a CLOSENESS score such that:
- `CLOSENESS(U=B) = 0` (or minimal),
- if `W1(U, I)` increases beyond `W1(B, I)` (worse than blur), CLOSENESS must **not increase** (should saturate at 0).

### New definition (W1-consistent, bounded, monotone)
Keep using W1 on 1D distributions of `u = d(p,GT)/s_fit`.

Compute:
- `d_ideal = W1(U, I)`
- `d_bad_ideal = W1(B, I)`  (constant per (MOP,K); precompute/store)
Then define:

`closeness = 1 - min(1, d_ideal / d_bad_ideal)`

Equivalently:
- `closeness = max(0, 1 - d_ideal / d_bad_ideal)`

### Notes / Implementation details
- `d_bad_ideal` must be > 0; add guard:
  - if `d_bad_ideal < eps`, set `closeness = 0` and log a warning (should not happen if blur is non-degenerate).
- This definition guarantees:
  - `closeness ∈ [0,1]`
  - `closeness(U=B)=0`
  - if `U` is worse than `B` w.r.t. distance to ideal, closeness stays at 0 (never increases).
- Keep NEAR@1 / NEAR@2 as additional interpretability metrics (optional but recommended).

### Where to change
- Replace the current `q_closeness` computation in the diagnostics module (where W1 is currently computed) with the bounded formula above.
- Update report label “Q_CLOSENESS” to reflect this bounded definition (still a [0,1] score).