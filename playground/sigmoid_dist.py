import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

bins = np.arange(-10, 10, 0.1)
plt.plot(bins, sigmoid(bins), linewidth=2, color="r")
plt.show()
