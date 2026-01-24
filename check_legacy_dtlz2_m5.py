
import pandas as pd
import numpy as np

legacy = pd.read_csv("tests/audit_data/legacy_DTLZ2/lg__DTLZ2_5_opt_front.csv", header=None).values
sos = np.sum(legacy**2, axis=1)
print(f"Legacy DTLZ2 (M=5) SOS mean: {np.mean(sos)}")
