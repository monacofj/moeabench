MoeaBench Testing Infrastructure (v0.7.7)
========================================

The project utilizes a testing pyramid structure to ensure the functional,
mathematical, and statistical integrity of the framework.

HOW TO RUN THE TESTS:
--------------------
Use the central orchestrator `test.py` located in the project root.
Note: Unit Tests run automatically as a FOUNDATION for all tiers.

1. Core Logic:
   $ python3 test.py --unit   # Functional only (~3s)
  
2. Daily Validation (Default):
   $ python3 test.py          # Unit + Light Tier (~6s)
   $ python3 test.py --light  # Same as above (explicit)
   
3. Convergence Regression Verification:
   $ python3 test.py --smoke  # Unit + Smoke Tier (~10 min)

4. Complete Scientific Audit:
   $ python3 test.py --heavy  # Unit + Heavy Tier (Hours)

5. Total Verification (Unit + Light + Smoke):
   $ python3 test.py --all


Calibration and Baselines
-------------------------
The scientific integrity of MoeaBench is anchored in the following artifacts:

- baselines_v0.8.0.csv: Consolidated performance metrics (IGD, HV, GD, SP) 
  derived from N=30 runs. This is the oracle for regression testing.
- CALIBRATION_v0.8.0.html: Comprehensive interactive report featuring 3D 
  visualizations and convergence analysis.
- calibration_data/: Raw CSV files per run/generation used to generate the 
  baselines and the report.
- ground_truth/: Mathematical optimal manifolds used as the reference "Truth".


COMPONENT DESCRIPTION:
----------------------

1. UNIT TESTS (tests/unit/):
   Pure and fast unit tests. These verify persistence logic (ZIP), data export 
   (CSV), metric evaluators, and the state of the Experiment object.

2. LIGHT TIER (tests/test_light_tier.py):
   Validates geometric and mathematical invariants of benchmarks (DTLZ/DPF) 
   without relying on stochasticity.

3. SMOKE TIER (tests/test_smoke_tier.py):
   Runs light iterations of algorithms with deterministic seeds and compares the 
   observed IGD against the official release baseline (baselines_v0.7.7.csv).

4. HEAVY TIER (tests/test_heavy_tier.py):
   Executes the full statistical calibration suite (N=30) and performs hypothesis 
   tests to validate new implementations.


NOTES:
------
- Pytest is the underlying execution engine. You can run specific directories 
  using `pytest tests/unit/`.
- Scripts in `examples/` are not included in the test orchestrator to avoid 
  graphical and CPU overhead, being treated strictly as documentation.
