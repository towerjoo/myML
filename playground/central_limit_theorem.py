import random
import matplotlib.pyplot as plt


# bernoulli distribution

def one_sample(N=50):
    return sum([random.choice([0, 1, 2, 3]) for i in range(N)]) 

def plot(N=50, M=1000, style="b_"):
    Y = []
    Z = []
    for i in range(M):
        one = one_sample(N)
        try:
            index = Z.index(one)
            Y[index] += 1
        except:
            Y.append(0)
            Z.append(one)
            Z.sort()
    plt.plot(Z, Y)
plot(90, 2000, "b_")
plot(90, 2000, "y_")
plot(90, 2000, "r_")
plt.show()
