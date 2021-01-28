import numpy as np
from matplotlib import pyplot as plt
import os

img_name = os.path.join("../", "img", "oscillations.pdf")
dt = 0.1
T = 500
w = 2 * np.pi * (1.0 / T)
t = np.arange(20000) * dt

y = np.sin(w * t)
p1 = (688.3, 0.7)
p2 = (1061.6, 0.7)
fig = plt.figure(figsize=(10,4))
plt.plot(t, y, linewidth = 2, color = 'k')
plt.plot(t, 0.7 * np.ones_like(t), linewidth = 1, linestyle = '--', color = 'r')
plt.scatter(*p1, s=50, color = 'r')
plt.scatter(*p2, s=50, color = 'r')
plt.axis("off")

plt.savefig(img_name)
plt.show()
plt.close()
