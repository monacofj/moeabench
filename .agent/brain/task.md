# Scientific Audit & Testing Framework

## Phase 1: Scientific Audit
- [x] Reorganize legacy data to `tests/audit_data/` <!-- id: 20 -->
- [x] **Phase 1A: Analytical Certification (The MOP Brain)** <!-- id: 21 -->
    - [x] Correct `DTLZ9` manifold sampling to curve-based ($f_1 = \dots = f_{M-1}$) <!-- id: 23 -->
    - [x] Implement `certify_mops.py` for Invariant Check (SOS/Sum) <!-- id: 24 -->
    - [x] Certify all MOPs at $M=3$ and $M=31$ <!-- id: 25 -->
    - [x] Generate Scientific Audit Report (Phase 1A) <!-- id: 100 -->
- [x] **Phase 1C: Ground Truth Generation (The Gold Standard)** <!-- id: 50 -->
    - [x] Document Scientific Integrity Policy in ADR 0015 <!-- id: 51 -->
    - [x] Generate Analytical CSVs for DTLZ1-7/9 and DPF1-5 <!-- id: 52 -->
    - [x] Execute High-Fidelity NSGA-III for DTLZ8 <!-- id: 53 -->
    - [x] Freeze `tests/ground_truth/` as definitive v0.7.5 reference <!-- id: 54 -->
- [ ] **Phase 1B: Empirical Calibration (The MOEA Rastro)** <!-- id: 26 -->
    - [x] **Phase 1B-A: Legacy Audit (The Confrontation)** <!-- id: 27 -->
    - [x] Analyze `conf.txt` for exact legacy parameters (N, M, K, D)
    - [x] Update `audit_legacy.py` with parity parameters
    - [x] Implement persistence for Shadow Data (F & X) in `tests/shadow_data/`
    - [x] Update `audit_legacy.py` for target-relative reporting (Sum/SOS)
    - [x] Create idiomatic diagnostic script `misc/check_dtlz2.py` (using `topo_shape`)
    - [x] Create idiomatic diagnostic script `misc/check_dtlz2.py` (using `topo_shape`)
    - [x] Create idiomatic diagnostic script `misc/check_dtlz2.py` (using `topo_shape`)
    - [x] Transform and Fix `misc/benchmark_gallery.ipynb` (Paths, Overlays, DTLZ8 Fix)
    - [x] Rectify Branding: "Legado (MoeaBench/legacy)" term across `misc/` <!-- id: 70 -->
    - [x] Sanitize Branching: Limit `add-test` usage to `misc/` notebooks <!-- id: 71 -->
    - [x] Generate definitive Scientific Audit Report (`misc/opt_report.md`) <!-- id: 101 -->
    - [x] Clean up setup logic in `benchmark_gallery.ipynb` (Clone + Local Install)
    - [x] Clean up root-level diagnostic scripts
    - [ ] Re-run shadow tests (v0.7.5) and generate definitive `docs/legacy_report.md`
- [x] Scientific Audit of Geometric Discrepancies <!-- id: 58 -->
    - [x] SOS Audit: Confirmed DPF2 regression (Code > File)
    - [x] Invariant Audit: DTLZ1 files in $\sum < 0.5$ state (Incompatible)
    - [x] DTLZ9 Conflict: Isolated UI Label/User report mismatch
- [/] **Phase 1B-B: Baseline Calibration (The Reference)** <!-- id: 31 -->
    - [x] Scientific Audit of Geometric Discrepancies (SOS/Sum Invariants)
    - [x] Rectify DPF Family Sampling and Projection Logic <!-- id: 62 -->
        - [x] Create Implementation Plan for Structured Sampling
        - [x] Implement Structured `ps()` in `base_dpf.py`
        - [x] Align `g`-factor minimization in DPF1-5
        - [x] Restore Squared Projections in DPF2 and DPF4 (Scientific Rectification) <!-- id: 75 -->
        - [x] Verify clean geometric fronts vs Clouds
    - [x] Subtask X: Scientific Audit & DTLZ8 Ground Truth Generation <!-- id: 80 -->
        - [x] Phase X1: Static High-Fidelity Files (M=3, 5, 10) <!-- id: 81 -->
        - [ ] Phase X2: Guided Analytical Solver <!-- id: 82 -->
    - [ ] Execute NSGA-II, NSGA-III, and MOEAD against Ground Truth <!-- id: 32 -->
        - [ ] Determine convergence thresholds (Pop/Gen/Seed) <!-- id: 33 -->
        - [ ] Establish definitive performance baselines (IGD/HV) <!-- id: 34 -->

## Phase 2: Testing Framework
- [ ] Implement Multi-Tiered Pytest Framework (Fast/Smoke/Heavy) <!-- id: 35 -->
- [ ] Integrate CI validation in GitHub Actions <!-- id: 36 -->

## Accomplishments (v0.6.3+)
- [x] Implement robust input validation in `BaseMop` and `BaseDPF`
- [x] Fix `DTLZ6` numerical regression
- [x] Standardize `**kwargs` across all MOPs
- [x] Add population size guards for DEAP
- [x] Create regression unit tests
