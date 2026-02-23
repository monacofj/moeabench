import MoeaBench as mb
import numpy as np

# 1. Exp2: Small population (Narrow front, unstable)
exp2 = mb.experiment("NSGA3_Narrow")
exp2.mop = mb.mops.DTLZ2()
exp2.moea = mb.moeas.NSGA3()
exp2.population = 10
exp2.run(5)

# 2. Exp1: Large population (Broad front, superior)
exp1 = mb.experiment("NSGA3_Broad")
exp1.mop = mb.mops.DTLZ2()
exp1.moea = mb.moeas.NSGA3()
exp1.population = 100 # Broadens the front coverage
exp1.run(5)

print("\n--- INDIVIDUAL EVALUATION (Local coordinate systems) ---")
hv2_alone = mb.metrics.hv(exp2)
print(f"Exp2 Mean (Alone): {hv2_alone.mean():.4f}")

hv1_alone = mb.metrics.hv(exp1)
print(f"Exp1 Mean (Alone): {hv1_alone.mean():.4f}")

print("\n--- SHARED EVALUATION (Shared coordinate system/Nadir) ---")
# This simulates what perf_history(exp1, exp2) does
hv2_shared = mb.metrics.hv(exp2, ref=[exp1, exp2])
hv1_shared = mb.metrics.hv(exp1, ref=[exp1, exp2])

print(f"Exp2 Mean (Shared): {hv2_shared.mean():.4f}")
print(f"Exp1 Mean (Shared): {hv1_shared.mean():.4f}")

if hv2_shared.mean() > hv2_alone.mean():
    diff = hv2_shared.mean() - hv2_alone.mean()
    print(f"\nUPLIFT CONFIRMED: Exp2 'subiu' {diff*100:.1f}%.")
    print("Razão: O Exp1 expandiu o Nadir, diminuindo o peso relativo da variância do Exp2.")

# Diagnostic of Nadir range
def get_range(exp):
    f = np.vstack([r.front() for r in exp])
    return np.max(f, axis=0) - np.min(f, axis=0)

print(f"\nObjective Range Exp2: {get_range(exp2)}")
print(f"Objective Range Exp1: {get_range(exp1)}")
