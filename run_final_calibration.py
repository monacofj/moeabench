import os
import sys

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from tests.calibration.generate_baselines import run_generation
from tests.calibration.compute_baselines import compute_baselines
from tests.calibration.generate_visual_report import generate_visual_report

# Comprehensive Optimization set
mops = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DPF1", "DPF2", "DPF3", "DPF4"]
algs = ["MOEAD"]
repeat = 5

print(f"Starting Scenario Final (v2): Regenerating baselines for {mops}...")
run_generation(mops=mops, algs=algs, repeat=repeat)

print("\nRecomputing consolidated metrics...")
compute_baselines()

print("\nRegenerating Visual Calibration Report...")
generate_visual_report()

print("\nScenario Final Complete.")
