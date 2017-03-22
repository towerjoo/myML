import matplotlib.pyplot as plt
import numpy as np

def softplus(x):
    return np.log(1 + np.exp(x))

bins = np.arange(-10, 10, 0.1)
plt.plot(bins, softplus(bins), linewidth=2, color="r")
plt.show()
