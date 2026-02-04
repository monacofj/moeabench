import os
import sys

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from tests.calibration.generate_baselines import run_generation
from tests.calibration.compute_baselines import compute_baselines
from tests.calibration.generate_visual_report import generate_visual_report

# The 4 critical optimized problems identified in the walkthrough
# DTLZ3 (Multimodal), DTLZ5 (Degenerate), DTLZ6 (Biased), DPF3 (Curved)
mops = ["DTLZ3", "DTLZ5", "DTLZ6", "DPF3"]
algs = ["MOEAD"]
repeat = 30 # High rigor for distribution analysis

print(f"=== Starting Statistical Calibration (N=30) ===")
print(f"Problems: {mops}")
print(f"Algorithm: {algs}")
print(f"Estimated Time: ~4 hours (Standard Intensity)")

# Run generation
run_generation(mops=mops, algs=algs, repeat=repeat)

print("\nRecomputing consolidated metrics (N=30)...")
compute_baselines()

print("\nRegenerating Visual Calibration Report...")
generate_visual_report()

print("\nStatistical Run Complete.")
