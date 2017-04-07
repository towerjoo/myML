from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
