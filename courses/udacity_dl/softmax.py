"""Softmax."""
import sys
import numpy as np
import math

scores = np.array([3.0, 1.0, 0.2])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print softmax(scores / 10)
sys.exit(0)
# Plot softmax curves
import matplotlib.pyplot as plt
import ipdb;ipdb.set_trace()
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
