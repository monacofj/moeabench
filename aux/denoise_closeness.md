# Spec (v2): Rename FIT→HEADWAY and add Q_CLOSENESS using GT-normal-blur (Half-Normal) + W1 Q-score

## Design goals
1) **User clarity**: “HEADWAY” means *better than noise baseline* (progress), not “close to GT”.
2) **No semantic overlap**: “CLOSENESS” measures only *perpendicular proximity to GT* (not coverage/regularity).
3) **Consistency**: Q_CLOSENESS uses the same Q-score “dialect” as other Q metrics (ECDF + Wasserstein-1).
4) **Scientific plausibility**: GT degradation uses a conventional error model (Gaussian along estimated normal ⇒ Half-Normal magnitudes).

---

## Part A — Rename FIT to HEADWAY (no math changes)
### A1. Public labels
- Replace all user-facing mentions:
  - `FIT` → `HEADWAY`
  - `q_fit` → `q_headway`
  - “Fit / Proximity” wording → “Headway / Better-than-noise progress”
- Keep the underlying computation unchanged.

### A2. Documentation snippet (glossary)
- **HEADWAY**: “How far the algorithm’s outcome is from a noise baseline (AnchorBad), relative to the ideal (AnchorGood). High HEADWAY does **not** imply closeness to the GT manifold.”

---

## Part B — Add CLOSENESS as a new Q metric (Q_CLOSENESS)

### B0. Core quantity (fair distance)
Let GT points be `G ⊂ R^M`. For an observed front `F = {p_j}` (after standard K-selection):
- For each `p_j`, compute `d_j = min_{g∈G} ||p_j - g||`.
- Normalize by GT resolution `s_fit` (already available/used): `u_j = d_j / s_fit`.

Interpretation: `u_j` is “distance-to-GT in units of GT resolution”.

CLOSENESS must depend on the distribution of `u_j` only (perpendicular proximity surrogate).

---

## Part B1 — Estimate local normals on the GT (generic for all MOPs)
Goal: approximate a unit normal `n(g)` at each GT point, using only the GT cloud.

### B1.1 Connected components (important for discontinuous GT)
1) Build a kNN graph on GT points using `k_graph` (default same as k below).
2) Compute connected components on this graph.
3) Store component id for each GT point.

### B1.2 Per-point normal via local PCA
For each GT point `g_i`:
1) Choose neighbor count:
   - `k = clamp(round(sqrt(|G|)), 10, 30)` (reduce if component smaller).
2) Find `k` nearest neighbors **within the same component** as `g_i`.
3) Run PCA on the neighbor coordinates (centered).
4) Let `n_i` be the eigenvector of the smallest eigenvalue.
5) Normalize: `n_i ← n_i / ||n_i||`.

Fallbacks:
- If component size < (k+1), set `k = max(3, component_size-1)`.
- If PCA is ill-conditioned, expand k modestly or use isotropic blur (see B2.4) as last resort.

Rationale: GT approximates an (M−1)-dimensional manifold; PCA of local neighborhood estimates tangent space; smallest-variance direction approximates normal.

---

## Part B2 — Construct a plausible “bad” baseline by normal blurring the GT
We define AnchorBad for CLOSENESS as `G_blur`, obtained by perturbing each GT point along its estimated normal.

### B2.1 Blur model (conventional)
For each GT point `g_i`, sample magnitude `|δ_i|` from a **Half-Normal** distribution:
- `|δ_i| ~ HalfNormal(σ)`.

Half-Normal is the magnitude of a 1D Gaussian error, i.e., a conventional additive error model.

### B2.2 Direction/sign choice
We do not require a globally consistent “outward” notion. Use:
- Draw sign `s_i ∈ {+1, -1}` uniformly.
- Proposed point: `g_i' = g_i + s_i * |δ_i| * n_i`.

Domain safety (recommended):
- If any coordinate of `g_i'` becomes negative, flip sign once.
- If still negative, resample `s_i` or reduce `|δ_i|` (e.g., multiply by 0.5) until valid.

### B2.3 Calibrate σ in units of s_fit (to avoid saturation)
Define normalized blur distance:
- `u_blur_i = min_{g∈G} ||g_i' - g|| / s_fit` (distance back to original GT, in fair units).

We choose σ so that the blur baseline is “clearly off GT but not astronomically bad”.
Calibration targets (initial convention):
- `median(u_blur) ≈ 2`
- `p95(u_blur) ≈ 5`

Implementation approach:
1) Initialize `σ = 2 * s_fit` (reasonable starting point).
2) Generate a provisional `G_blur` and compute `u_blur`.
3) Update σ by multiplicative scaling to match `median(u_blur)` (primary target), then check p95:
   - `σ ← σ * (2 / median(u_blur))`
4) Repeat 3–6 iterations or until median within tolerance (e.g., ±5%).
5) Verify `p95(u_blur)` is in [4.5, 5.5]; if not, adjust with a small corrective factor (optional), or accept and document.

Note: For Half-Normal, p95/median is fixed (~2.58). So if you exactly match median=2, p95 will land near 5.16 automatically. This is a *feature*: it reduces degrees of freedom and makes the baseline less arbitrary.

### B2.4 Fallback when normals are unreliable
If normals cannot be robustly estimated (rare, but possible for very sparse GT):
- Use isotropic blur with the same Half-Normal magnitude:
  - sample a random unit vector `v` uniformly on S^(M−1),
  - set `g_i' = g_i + s_i * |δ_i| * v`.
This is inferior semantically but preserves reproducibility.

### B2.5 Store baseline
Store for each `(MOP, K)`:
- `G_blur` (or equivalently its ECDF of u_blur),
- `σ`,
- seed used,
so that audits are deterministic.

---

## Part B3 — Define Q_CLOSENESS using the same Q-score form as other metrics
We define distributions (1D samples):
- Observed: `U = {u_j}` from the algorithm run.
- Ideal: `I = {0,0,...,0}` (degenerate at 0; same length as U or a fixed large length).
- Bad: `B = {u_blur_i}` from the GT-blur baseline (precomputed).

Compute Wasserstein-1 distances:
- `d_bad = W1(U, B)`
- `d_ideal = W1(U, I)`

Define:
- `q_closeness = d_bad / (d_bad + d_ideal)`

Properties:
- If U is close to Ideal (near 0) ⇒ `d_ideal` small ⇒ q close to 1.
- If U resembles blurred baseline ⇒ `d_bad` small ⇒ q close to 0.
- This matches existing Q-score semantics and avoids the FIT/BBox-random saturation issue.

---

## Part B4 — Reporting and interpretation
### B4.1 New metric name
- Add metric **CLOSENESS** to the clinical matrix.
- Report `q_closeness` as the main CLOSENESS score.

Optional but highly recommended (for interpretability):
- `near@1 = 100 * mean(u_j <= 1)`
- `near@2 = 100 * mean(u_j <= 2)`
These are not part of Q but help users instantly see “how much is actually on GT”.

### B4.2 Verdict logic (initial suggestion)
To prevent “headway-only” false positives, require CLOSENESS:
- A run/problem cannot be “good” overall if CLOSENESS is below threshold.
- Reuse existing thresholds (0.34/0.67) for CLOSENESS to stay consistent, then tune later.

---

## Part B5 — Acceptance tests (must pass)
1) **DTLZ2** (NSGA2/NSGA3): high CLOSENESS and high near@1.
2) **DTLZ6** with visible parallel arcs: low CLOSENESS even if HEADWAY is high.
3) **Disconnected GT** (e.g., DTLZ7-like): CLOSENESS stable (no cross-component neighbor contamination).
4) Reproducibility: `G_blur`, σ, and q_closeness identical across runs given same seeds.

---