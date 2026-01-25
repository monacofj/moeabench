import os
import sys
import pandas as pd
import numpy as np

def discover_root():
    curr = os.getcwd()
    for _ in range(3):
        if os.path.exists(os.path.join(curr, "MoeaBench")):
            return curr
        curr = os.path.dirname(curr)
    return os.getcwd()

PROJECT_ROOT = discover_root()
print(f"Discovered Root: {PROJECT_ROOT}")

# Simulating DTLZ1 legacy resolution
mop_name = "DTLZ1"
M = 3
prefix = "lg__" if "DTLZ" in mop_name else "lg_"
leg_filename = f"{prefix}{mop_name}_{M}_opt_front.csv"
leg_path = os.path.join(PROJECT_ROOT, "tests/audit_data", f"legacy_{mop_name}", leg_filename)

if os.path.exists(leg_path):
    print(f"SUCCESS: {leg_path} exists.")
else:
    print(f"FAILURE: {leg_path} NOT FOUND.")

