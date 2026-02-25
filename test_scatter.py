import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

s1 = 15
s2 = [15]
s3 = np.array([15])

try:
    ax.scatter(x, y, z, s=s3)
    print("s3 works")
except Exception as e:
    print("s3 fails:", e)
