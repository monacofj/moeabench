# Peer Review Response: Calibration Audit (v0.7.6)

## Topic A: Hypervolume Integrity & Normalization Standards

### The Issue
The peer review identified a "conceptual/accounting" inconsistency in the calibration report. Specifically, while the documentation claimed "Strict Theoretical Normalization" into the unit interval $[0, 1]$, the reported Hypervolume (HV) metrics frequently exceeded the theoretical maximum of $1.0$.

For example:
- **DPF4:** Mean HV $\approx 1.15$
- **DTLZ1:** Mean HV $\approx 1.13$

This discrepancy raised concerns about whether the normalization was merely cosmetic (clipping) or if the reference point was inconsistent with the unit hypercube.

### Technical Diagnosis
An audit of the codebase (`compute_baselines.py` and `GEN_hypervolume.py`) confirmed that the mathematical core is sound but operated with a non-standard configuration:

1.  **Normalization Process:** The system correctly identifies the `Ideal` and `Nadir` points from the union of the Ground Truth and all observed fronts, mapping the objective space to $[0, 1]$.
2.  **Reference Point Configuration:** The historical default for the Hypervolume calculation was hardcoded to `1.1` in the normalized space. This was intended to ensure that extreme points on the Pareto front contribute to the volume, preventing boundary effects where optimal points yield zero volume improvement.
3.  **Geometric Consequence:** A reference point of `1.1` in 3D define a bounding box of volume $1.1^3 \approx 1.331$.
    - Thus, a result of $HV = 1.15$ is numerically valid and within the bounds ($1.15 < 1.331$).
    - It is **not** an error of normalization, but a misunderstanding caused by the implicit assumption that "normalized HV" must be $\le 1.0$ (which assumes a reference point of exactly $1.0$).

### Corrective Actions
To resolve the ambiguity without invalidating previous experimental data, the following changes were implemented in the `moeabench` codebase:

1.  **Explicit Configuration:** The `GEN_hypervolume` class was refactored to accept a configurable `ref_point` parameter, removing the hardcoded value.
2.  **Transparency in Pipelines:** The baseline computation script (`compute_baselines.py`) now explicitly defines `ref_point=1.1`, documenting the intent.
3.  **Sanity Enforcement:** A runtime check was added to compare the calculated HV against the theoretical maximum volume of the reference box ($Ref^M$).
4.  **Standardized Nomenclature:** To eliminate future confusion, we are adopting a tripartite definition for Hypervolume reporting:
    *   **HV Raw:** The absolute value calculated with Reference Point 1.1 (e.g., 1.15).
    *   **HV Ratio:** $HV_{raw} / Ref^M$. Measures volume coverage of the reference cube. Always mathematically $\le 1.0$ (e.g., $1.15 / 1.331 \approx 0.86$).
    *   **HV Rel (Relative):** $HV_{raw} / HV_{GT}$. Measures convergence to the known ground truth (e.g., $1.15 / 1.156 \approx 0.99$).
    All three metrics will be presented in the final certification report.


### Conclusion for Reviewers
The "HV > 100%" phenomenon is a documented feature of using a conservative reference point ($1.1$) to preserve boundary contributions. It does not indicate a failure in the $[0,1]$ mapping. The codebase reflects this explicitly now.

## Topic B: The Divergence of Metrics (EMD vs IGD)

### The Paradox
In the calibration audit, a second anomaly was observed: for specific problems such as DPF3, the **Inverted Generational Distance (IGD)** reported excellent performance (values $\approx 0.002$), implying the solution was virtually identical to the Pareto Front. However, the **Earth Mover's Distance (EMD)** for the same solution reported a catastrophic failure (values $> 0.40$), implying a massive discrepancy in distribution.

This divergence challenges the intuitive assumption that "converging to the front" (IGD) implies "solving the problem" (EMD). To investigate whether this was a defect in the EMD metric calculation—specifically, a sensitivity to population size differences—we conducted a controlled isolation experiment.

### Experimental Isolation
The audit isolated the DPF3 problem, comparing a solution set of size $N=200$ against a ground truth of size $N=1573$. Three scenarios were evaluated to identify the root cause of the error signal:

1.  **Baseline Measurement:** The raw EMD calculation confirmed the massive error ($0.425$), despite the IGD being negligible ($0.002$).
2.  **Resampling Hypothesis:** We tested if the disparity in set limits (200 vs 1573) was inflating the transport cost. By bootstrapping the solution set to match the ground truth size ($N=1573$), the EMD remained stubbornly high ($0.409$), disproving the hypothesis that this was a mere cardinality artifact.
3.  **Control Group Validation:** A synthetic "perfect" solution was created by taking a random uniform subsample of the Ground Truth. This control group yielded an EMD of just $0.009$.

### Interpretation
The findings reveal that the **divergence is physically meaningful**, not an artifact.

IGD is a measure of **proximity**: it asks, "Are the points close to the optimal manifold?" The answer for DPF3 is yes. The algorithm successfully collapsed the population onto the curve.

EMD is a measure of **topology and distribution**: it asks, "Does the population cover the manifold with the same density as the ground truth?" The answer is a definitive no. The high EMD value indicates that while the points are *on* the curve, they are topologically clustered—likely collapsing into a few small regions or "clumps"—leaving vast sections of the Pareto Front unexplored.

Therefore, no corrective action is required for the metric codebase. The EMD is correctly serving its purpose as a discriminator for topological diversity, penalizing algorithms that achieve convergence (low IGD) at the expense of diversity (high EMD).

## Topic C: Algorithm Limits on Degenerate Manifolds

### Investigation of Failures
Completing the audit, we investigated why the reference algorithms (NSGA-II and MOEA/D) exhibited "High EMD / Low IGD" behavior on the DPF3 problem.

### Diagnosis: The $x^{100}$ Transformation
The DPF3 benchmark defines its manifold using a highly non-linear mapping:
$$ \theta = x^{100} \cdot \frac{\pi}{2} $$
This power-law transformation causes an extreme "compression" of the objective space. For uniformly sampled decision variables $x \in [0, 1]$, the vast majority of points map to $\theta \approx 0$, collapsing the Pareto Front into a small, dense cluster. Only values of $x$ extremely close to $1.0$ (e.g., $>0.99$) can generate points in the "tail" of the front that reach the boundaries of the objective space.

### The Algorithm Trap
1.  **Boundary Recession:** Our mechanics audit reveals that NSGA-II (Pop=200) fails to maintain individuals in this "sparse tail" region. The final population consistently exhibits a **gap of ~65%** in the objective bounds (covering only $[0, 0.34]$ instead of $[0, 1]$).
2.  **Metric Deception:**
    *   **IGD (Low):** Since the Ground Truth itself is generated via sampling (which mimics the density bias), 99% of the reference points lie in the "dense cluster". The algorithm covers this cluster well, resulting in a low average distance error ($0.002$).
    *   **EMD (High):** The EMD metric, being sensitive to distribution, correctly identifies that the algorithm has completely missed the "long tail" of the manifold (the rare, extreme points), rendering a high error score ($0.42$).

### Conclusion
The "failure" is not a software bug but a demonstration of the metrics working as intended. DPF3 is designed to break algorithms that rely on uniform initialization. The high EMD score accurately reflects the **Loss of Extents** caused by the degenerate geometry.
