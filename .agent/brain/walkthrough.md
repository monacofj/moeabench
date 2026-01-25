# Walkthrough: Scientific Audit & DTLZ8 Ground Truth

I have successfully completed the scientific audit and the rectification of the DPF family. The analytical Ground Truth (v0.7.5) now provides a perfect, high-density reference that overlays the legacy data.

## Changes Made

### 1. Scientific Rectification (DPF Projections)
- **[DPF2](file:///home/monaco/Work/moeabench/MoeaBench/mops/DPF2.py)**: Restored squared projections (`square=True`). This was the "smoking gun" identified in the 3D audit, resolving the scale discrepancy between v0.7.5 (linear) and the legacy (squared) data.
- **[DPF4](file:///home/monaco/Work/moeabench/MoeaBench/mops/DPF4.py)**: Restored squared mapping for all objectives to match the original DPF definition.

### 2. Structured Sampling (`ps()`)
- **[BaseDPF](file:///home/monaco/Work/moeabench/MoeaBench/mops/base_dpf.py)**: Replaced stochastic variables with structured `linspace` sampling.
- **[DPF5](file:///home/monaco/Work/moeabench/MoeaBench/mops/DPF5.py)**: Implemented 2D manifold sampling, transforming "clouds" into dense, sharp surfaces.

### 3. Audit Branding & Branching
- **Branding**: Updated all notebooks and diagnostic scripts to use the term **"Legado (MoeaBench/legacy)"** instead of "v0.6.x".
- **Branching**: Sanitized the **[benchmark_gallery.ipynb](file:///home/monaco/Work/moeabench/misc/benchmark_gallery.ipynb)** setup to pull strictly from the `add-test` branch via a clean `git clone` and local editable install.

### 4. DTLZ8 Ground Truth (Subtask X)
- **Generation**: Executed super-convergent NSGA-III (Pop=1000, Gen=2000) for $M \in \{3, 5, 10\}$.
- **Integration**: Updated [DTLZ8.py](file:///home/monaco/Work/moeabench/MoeaBench/mops/DTLZ8.py) to automatically load these high-fidelity CSVs from the `MoeaBench/mops/data/` directory.
- **Dynamic Resilience**: Implemented a fallback error for non-standard $M$ while planning the Phase X2 analytical solver.

## Verification Results

The 3D audit in `misc/figs/` confirms that the **v0.7.5 Ground Truth (Blue)** now perfectly overlays the **Legado (Red)** rastro, but with superior density and precision.
For DTLZ8, the new ground truth represents the intersection of the "cliff-like" constraints, providing a strictly feasible reference set that legacy data often failed to achieve.

````carousel
![DPF2 Fixed](/home/monaco/Work/moeabench/.agent/brain/DPF2_audit.png)
<!-- slide -->
![DPF5 Dense](/home/monaco/Work/moeabench/.agent/brain/DPF5_audit.png)
<!-- slide -->
![DTLZ9 Exact](/home/monaco/Work/moeabench/.agent/brain/DTLZ9_audit.png)
<!-- slide -->
![DTLZ8 GT](/home/monaco/Work/moeabench/.agent/brain/DTLZ8_audit.png)
````

### Status
- **Commited & Ready**: All changes including DTLZ8 high-fidelity datasets (M=3, 5, 10) are ready on the `add-test` branch.
- **Ready for Review**: The `benchmark_gallery.ipynb` is updated and fully operational with the new DTLZ8 logic.
