import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

s1 = 15.0

try:
    ax.scatter(x, y, z, s=s1)
    print("s1 works")
except Exception as e:
    print("s1 fails:", e)
