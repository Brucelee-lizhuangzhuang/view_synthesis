from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import time


fig = plt.figure()
plt.ion()
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

for root, folder, files in os.walk("./results/features"):
    for file in sorted(files):
        if ".npy" not in file:
            continue

        p = np.load(os.path.join("./results/features",file))
        p = np.reshape(p, (13,3))
        ax.clear()
        ax.grid(False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.scatter(p[:,0],p[:,1],p[:,2])
        plt.draw()
        plt.pause(1)
        plt.show()

