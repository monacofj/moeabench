MoeaBench Testing Infrastructure (v0.9+)
========================================

The project utilizes a testing pyramid structure to ensure the functional,
mathematical, and statistical integrity of the framework.

HOW TO RUN THE TESTS:
--------------------
Use the central orchestrator `test.py` located in the project root.
Note: Unit Tests run automatically as a FOUNDATION for all tiers.

1. Core Logic:
   $ python3 test.py --unit         # Functional only (~3s)

2. Regression Tests:
   $ python3 test.py --regression   # Numerical Certification (~2s)
  
3. Daily Validation (Default):
   $ python3 test.py                # Unit + Light + Regression (~6-8s)
   $ python3 test.py --light        # Same as above (explicit)
   
4. Convergence Regression Verification:
   $ python3 test.py --smoke        # Unit + Smoke Tier (~10 min)

5. Complete Scientific Audit:
   $ python3 test.py --heavy        # Unit + Heavy Tier (Hours)

6. Total Verification (Unit + Light + Smoke + Regression):
   $ python3 test.py --all


Calibration and Reference Data
------------------------------
The scientific integrity of MoeaBench is anchored in the following artifacts:

- calibration_reference_audit_v0.9.json: Raw coordinates for 42 reference 
  scenarios. Used as input for the Regression Tier.
- calibration_reference_targets.json: Validated Q-Scores and FAIR metrics 
  (v0.9+). The oracle for numerical reproducibility.
- baselines_v0.8.0.csv: Consolidated performance metrics (IGD, HV) derived 
  from N=30 runs. Used by the Smoke Tier for convergence checks.
- ground_truth/: Mathematical optimal manifolds used as the reference "Truth".


COMPONENT DESCRIPTION:
----------------------

1. UNIT TESTS (tests/unit/):
   Pure and fast unit tests. These verify persistence logic (ZIP), data export 
   (CSV), metric evaluators, and the state of the Experiment object.

2. LIGHT TIER (tests/test_light_tier.py):
   Validates geometric and mathematical invariants of benchmarks (DTLZ/DPF) 
   without relying on stochasticity.

3. REGRESSION TIER (tests/test_regression_tier.py):
   Certified numerical reproducibility. Verifies if Q-Scores and FAIR metrics 
   match the Calibration Reference targets (v0.9+) down to 6 decimal places.

4. SMOKE TIER (tests/test_smoke_tier.py):
   Runs light iterations of algorithms with deterministic seeds and compares the 
   observed IGD against the release baseline (baselines_v0.8.0.csv).

5. HEAVY TIER (tests/test_heavy_tier.py):
   Executes the full statistical calibration suite (N=30) and performs hypothesis 
   tests to validate new implementations.


NOTES:
------
- Pytest is the underlying execution engine. You can run specific directories 
  using `pytest tests/unit/`.
- Scripts in `examples/` are not included in the test orchestrator to avoid 
  graphical and CPU overhead, being treated strictly as documentation.
