import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(1000)
y = np.random.rand(1000)
z = np.random.rand(1000)

s1 = 15

try:
    ax.scatter(x, y, z, color='lightgray', marker='o', s=s1)
    print("s1 works")
except Exception as e:
    print("s1 fails:", e)
