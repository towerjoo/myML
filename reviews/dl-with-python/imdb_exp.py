import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import imdb
from keras.utils import to_categorical

def vectorize_sequences(sequences, dimensions=10000):
    # one-hot encoding
    results = np.zeros((len(sequences), dimensions))
    for i, c in enumerate(sequences):
        results[i, c] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# encoding the data to make them have the same length
train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)

#vectorize labels
train_labels = np.asarray(train_labels).astype("float32")
test_labels = np.asarray(test_labels).astype("float32")


network = models.Sequential()
network.add(layers.Dense(16, activation="relu", input_shape=(10000, )))
network.add(layers.Dense(16, activation="relu"))
network.add(layers.Dense(1, activation="sigmoid"))

network.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# split training data to training and validation
x_val = train_data[:10000]
y_val = train_labels[:10000]
partial_x_train = train_data[10000:]
partial_y_train = train_labels[10000:]

history = network.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
