# ADR 0030: Clinical Instrument Architecture

## Status
Proposed (2026-02-18)

## Context
Following the formalization of v0.9 "Scientific Certification" (ADR 0026 and 0028), there was a need for a visualization layer that matches this metrological rigor. Standard "Performance Plots" (like HV-over-time) are insufficient for diagnosing *why* an algorithm fails or which morphological pathologies are present.

The goal is to provide a "Clinical Layer" in `mb.view.clinic_*` that treats metrics as raw physical facts (Layer 1) and certifications as engineering grades (Layer 2).

## Decision
We implement a **Quadriga of Clinical Instruments** based on a polymorphic, semantically-aligned architecture.

### 1. The Instrument Abstraction
Unlike traditional plotting functions that are tied to a specific metric (e.g., `plot_hv`), clinical instruments are **agnostic to the physical dimension**. An instrument is a specialized way of looking at a distribution:

*   **`clinic_ecdf` (The Judge)**: Analyzes goal-attainment probability.
*   **`clinic_distribution` (The Pathologist)**: Analyzes error morphology (shape, bias, outliers).
*   **`clinic_radar` (The Certification)**: A holistic biopsy of the 6 Q-Scores.
*   **`clinic_history` (The Monitor)**: Analyzes the temporal stability of the physical facts.

### 2. Semantic Distribution Layer (SDL)
To enable this universal support, we refactor the diagnostic engine to return **Structured Results** instead of raw scalars or arrays. The base `DiagnosticValue` (and its child `FairResult`) now carries a `raw_data` payload:

```python
class DiagnosticValue:
    value: float             # Representative scalar (The "Fact")
    raw_data: np.ndarray     # Underlying distribution (The "Evidence")
    name: str                # Metric identifier
    description: str         # Narrative insight
```

This allows instruments to automatically extract the correct distribution for any given metric:
- For `closeness`: `raw_data` = distances from population to GT.
- For `coverage`: `raw_data` = distances from GT to population (uncovered regions).
- For `regularity`: `raw_data` = point-to-point spacing distribution.

### 3. Visual Standard & Scale Invariance
Clinical plots must follow a strict aesthetic and metrological standard:
- **Less is Better**: All clinical physical metrics are normalized such that $0.0$ is the ideal state. Visualizations emphasize the "Distance from Zero".
- **Scale Invariance**: Vertical and horizontal "drop-lines" at the **Median (50%)** and **Robust Max (95%)** are required in ECDF plots to provide scale-invariant anchors according to the MoeaBench resolution factor ($s_k$).
- **Color Consistency**: Teal is used for the current solution; Red is used for critical thresholds (Headway/95th percentile).

## Consequences
- **Technically Sound Visuals**: Graphs are no longer just "pictures"; they are statistically accurate representations of the underlying FAIR distributions.
- **Universal API**: Users can explore any performance dimension using any instrument (`mb.view.clinic_ecdf(exp, metric="balance")`).
- **Internal Complexity**: The diagnostic pipeline (fair.py, auditor.py, qscore.py) must strictly adhere to the `FairResult` interface to avoid breaking the visual layer.
- **Performance**: Carrying `raw_data` in every diagnostic result increases memory usage slightly during large-scale audits, but is necessary for deep diagnostics.
