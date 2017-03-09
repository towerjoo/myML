import matplotlib.pyplot as plt
import numpy as np
import random
from functools import partial

def target(point1, point2, x, y):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    return 1 if slope * x - y > 0 else -1

    
def plot_training_set(f, X1, X2):
    positiveX1 = []
    positiveX2 = []
    negativeX1 = []
    negativeX2 = []
    for index, x in enumerate(X1):
        if f(x=x, y=X2[index]) == 1:
            positiveX1.append(x)
            positiveX2.append(X2[index])
        else:
            negativeX1.append(x)
            negativeX2.append(X2[index])
    plt.plot(positiveX1, positiveX2, "ro")
    plt.plot(negativeX1, negativeX2, "bx")

def plot_target_function(point1, point2):
    # make the line longer
    X = [-1, 1]
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    Y = [slope * x for x in X]
    plt.plot(X, Y, "g")
    
def sign(x):
    return 1 if x > 0 else -1

def pla(N):
    point1 = np.random.uniform(-1, 1, 2)
    point2 = np.random.uniform(-1, 1, 2)
    f = partial(target, point1=point1, point2=point2) 
    X1 = np.random.uniform(-1, 1, N)
    X2 = np.random.uniform(-1, 1, N)
    plot_target_function(point1, point2)
    plot_training_set(f, X1, X2)

    d = 2
    w = [0] * d
    M = 1000 # iterations
    misclassified_index = list(range(N)) # all are misclassified
    for i in range(M):
        # pick a random misclassified point
        #print "#{}: misclassified num {}".format(i, len(misclassified_index))
        index = random.choice(misclassified_index)
        x1 = X1[index]
        x2 = X2[index]
        y = f(x=x1, y=x2)
        w = (w[0] + x1 * y, w[1] + x2 * y)
        # update the misclassified
        misclassified_index = [index for index, value in enumerate(X1) if f(x=value, y=X2[index]) != sign(value * w[0] + X2[index] * w[1])]
        plot_hypothes(w, i, is_last=len(misclassified_index) == 0)
        if len(misclassified_index) == 0:
            break
    iterations = i + 1
    plt.show()

    disagrees_num = 0
    for i in range(M):
        x, y = np.random.uniform(-1, 1, 2)
        if f(x=x, y=y) != sign(w[0] * x + w[1] * y):
            disagrees_num += 1
    return iterations, float(disagrees_num) / M

def plot_hypothes(w, i, is_last=False):
    slope = -w[0] / w[1]
    X = [-1, 1]
    Y = [x * slope for x in X]
    plt.plot(X, Y, "y" if is_last else "k")
    plt.annotate(str(i+1), xy=(X[0], Y[0]), xytext=(X[0]-0.1, Y[0]), arrowprops=dict(arrowstyle='->', connectionstyle="arc3"))

if __name__ == "__main__":
    N = 100
    M = 1 # runs
    iterations = 0
    prob = 0
    for i in range(M):
        it, p = pla(N)
        iterations += it
        prob += p
    print iterations / float(M), prob / M
