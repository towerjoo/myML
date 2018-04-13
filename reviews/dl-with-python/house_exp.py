import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import boston_housing
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# encoding the data to make them have the same length
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

def build_model():
    network = models.Sequential()
    network.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1], )))
    network.add(layers.Dense(64, activation="relu"))
    network.add(layers.Dense(1)) # this is a linear transformation

    # mse: mean squared error
    # mae: mean absolute error
    network.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return network

# since the training dataset is too small, ~500 samples
# we can use k-fold validation

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print "processing fold #", i
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_labels[i*num_val_samples:(i+1)*num_val_samples]

    # the remaining samples are training data
    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i+1)*num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
print all_scores, np.mean(all_scores)
# the above validation process is to allow you to find the best parameters, e.g epochs, network archietecture, etc.
# after that, you can use the best parameters to train with the whole training set
# and evaluate against the test data
