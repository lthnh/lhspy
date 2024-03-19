import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

fs = 100
ts = 1 / fs
n = np.array(range(0, 65))
sample = np.sin(2 * np.pi * n * ts)
plt.plot(n, sample, 'ro')
plt.show()
print(sample)