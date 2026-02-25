import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(1000, 1)
y = np.random.rand(1000, 1)
z = np.random.rand(1000, 1)

s1 = 15

try:
    ax.scatter(x, y, z, s=s1)
    print("s1 with (1000, 1) works")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("s1 with (1000, 1) fails")

