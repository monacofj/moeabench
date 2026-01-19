import MoeaBench as mb
import numpy as np

# 1. Setup experiment
exp = mb.experiment()
exp.mop = mb.mops.DTLZ2(M=3)
exp.moea = mb.moeas.NSGA3()

# 2. Run 3 times
exp.run(repeat=3)

# 3. Verify counts
cloud_pop = exp.pop()
last_run_pop = exp.last_run.pop()

print(f"Total Cloud Population Size: {len(cloud_pop)}")
print(f"Last Run Population Size: {len(last_run_pop)}")

# 4. Verify Filters
cloud_front = exp.front()
last_front = exp.last_run.front()

print(f"Super-Elite (Cloud Front) Size: {len(cloud_front)}")
print(f"Local Elite (Last Run Front) Size: {len(last_front)}")

# Global check
assert len(cloud_front) >= len(last_front), "Cloud front should be at least as large as a single front"
assert len(cloud_pop) == 3 * len(last_run_pop), "Cloud should be 3x last run"

print("\nSUCCESS: Manager Mode is semantically consistent!")
