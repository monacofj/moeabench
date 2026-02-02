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
