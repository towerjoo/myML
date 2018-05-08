from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras import layers
from keras.layers import Flatten, Dense, Embedding
from keras.optimizers import RMSprop

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation="relu"))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation="relu"))
model.add(layers.GlobalMaxPooling1D())
model.add(Dense(1))
model.compile(optimizer=RMSprop(lr=1e-4), loss="binary_crossentropy", metrics=["acc"])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=.2)
