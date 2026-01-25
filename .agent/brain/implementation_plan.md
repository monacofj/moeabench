# Plan: DTLZ8 Ground Truth Generation (Subtask X)

The DPF family has been rectified. We now move to **Subtask X**: establishing the definitive Ground Truth for **DTLZ8**, which is a constrained problem that lacks a simple analytical $ps()$ formula.

## Proposed Changes

### [Mops Component]

#### [MODIFY] [DTLZ8.py](file:///home/monaco/Work/moeabench/MoeaBench/mops/DTLZ8.py)
- **Phase X1**: Implement a lookup mechanism in `ps()` and `pf()` to load static high-fidelity CSV files for $M \in \{3, 5, 10\}$.
- **Phase X2**: Implement a guided analytical solver for dynamic sampling of `ps()` when parameters fall outside the standard set.

### [Data Component]

#### [NEW] [DTLZ8 High-Fidelity CSVs](file:///home/monaco/Work/moeabench/MoeaBench/mops/data/)
- Generate `DTLZ8_{M}_optimal.csv` files using a heavy-duty NSGA-III (Pop=1000, Gen=2000) for $M=3, 5, 10$.
- Store both `F` (objectives) and `X` (decision variables) to allow full `ps()` reconstruction.

## Verification Plan

### Automated Tests
- Run `certify_mops.py` to ensure generated DTLZ8 points satisfy all $M$ constraints ($G \le 0$) with high precision.
- Perform a 3D visual audit for $M=3$ comparing the new X1 files against the legacy "nuvem".

### Manual Verification
- Review the density and coverage of the generated M=30 front to ensure no "holes" exist in the manifold.
