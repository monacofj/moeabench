import numpy as np
from MoeaBench import mb
import matplotlib.pyplot as plt

# 1. Test Topography
print("Testing Topography...")
data1 = np.random.rand(20, 2)
data2 = np.random.rand(20, 2) + 0.2

# topo_shape
mb.view.topo_shape(data1, labels=["Set A"])
# topo_density
mb.view.topo_density(data1, data2, show=False)

# 2. Test Performance
print("Testing Performance...")
# perf_history needs a MetricMatrix or Experiment, but let's test imports
try:
    print(f"perf_history: {mb.view.perf_history}")
    print(f"perf_spread: {mb.view.perf_spread}")
    print(f"perf_density: {mb.view.perf_density}")
except AttributeError as e:
    print(f"Error in Performance imports: {e}")

# 3. Test Stratification
print("Testing Stratification...")
try:
    print(f"strat_ranks: {mb.view.strat_ranks}")
    print(f"strat_hierarchy: {mb.view.strat_hierarchy}")
    print(f"strat_tiers: {mb.view.strat_tiers}")
except AttributeError as e:
    print(f"Error in Stratification imports: {e}")

# 4. Test Stats Renaming
print("Testing Stats Renaming...")
try:
    print(f"perf_probability: {mb.stats.perf_probability}")
    print(f"perf_distribution: {mb.stats.perf_distribution}")
    print(f"topo_distribution: {mb.stats.topo_distribution}")
    print(f"topo_attainment: {mb.stats.topo_attainment}")
except AttributeError as e:
    print(f"Error in Stats imports: {e}")

# 5. Test Legacy Aliases
print("Testing Legacy Aliases...")
try:
    print(f"spaceplot: {mb.spaceplot}")
    print(f"timeplot: {mb.timeplot}")
except AttributeError as e:
    print(f"Error in Legacy Aliases: {e}")

print("\nVerification script finished.")
