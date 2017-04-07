import numpy as np
import sys

def RR(X, y, _lambda):
    name = "wRR_{}.csv".format(_lambda)
    _lambda = float(_lambda)
    rows, cols = X.shape
    inv =np.linalg.inv(_lambda * np.identity(cols) + np.matmul(X.T, X)) 
    w = np.matmul(np.matmul(inv,X.T), y)
    np.savetxt(name, w)
    return w

def active_learning(X, y, _lambda, _sigma, x_test):
    N = 10
    locations = []
    name = "active_{}_{}.csv".format(_lambda, _sigma)
    _sigma = float(_sigma)
    for i in range(N):
        w = RR(X, y, _lambda)
        sigma_0 = _sigma + np.matmul(x_test, np.sum(x_test, 0, keepdims=True).T)
        index = np.argmax(sigma_0)
        y_0 = np.matmul(x_test[index], w)
        locations.append(str(index+1))
        X = np.append(X, [x_test[index]], 0)
        y = np.append(y, [y_0], 0)
        x_test = np.delete(x_test, index, axis=0)
        _sigma = sigma_0[index]
    fh = open(name, "w")
    fh.write(",".join(locations))
    fh.close()


if __name__ == "__main__":
    _, p_lambda, p_sigma2, x_train, y_train, x_test = sys.argv
    x_train = np.loadtxt(x_train, delimiter=',')
    y_train = np.loadtxt(y_train, delimiter=',')
    x_test = np.loadtxt(x_test, delimiter=',')
    RR(x_train, y_train, p_lambda)
    active_learning(x_train, y_train, p_lambda, p_sigma2, x_test)


