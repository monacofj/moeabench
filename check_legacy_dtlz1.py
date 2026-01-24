
import pandas as pd
import numpy as np

legacy = pd.read_csv("tests/audit_data/legacy_DTLZ1/lg__DTLZ1_3_opt_front.csv", header=None).values
print(f"Legacy DTLZ1 (M=3) Sum of Objectives (Should be 0.5):")
print(np.mean(np.sum(legacy, axis=1)))
