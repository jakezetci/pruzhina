import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/amps.txt')
left = data.T[0]
right = data.T[1]
freq = data.T[2]
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(freq, left)
ax2.plot(freq, right)
ax3.plot(freq, left*right)
plt.show()