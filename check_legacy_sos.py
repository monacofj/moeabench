
import pandas as pd
import numpy as np

legacy = pd.read_csv("tests/audit_data/legacy_DTLZ2/lg__DTLZ2_3_opt_front.csv", header=None).values
sos = np.sum(legacy**2, axis=1)
print(f"Legacy DTLZ2 (M=3) Sum of Squares mean: {np.mean(sos)}")
print(f"Legacy DTLZ2 (M=3) Sum of Squares std: {np.std(sos)}")
