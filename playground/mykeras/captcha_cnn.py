from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import uuid
import os
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D

def gen_captcha():
    image = ImageCaptcha(width=60, height=100)
    for i in range(1000):
        d = random.randint(0, 4)
        img = image.generate_image(str(d))
        img = img.convert('L')
        guid = uuid.uuid4()
        img.save("train/{}-{}.png".format(d, guid))

    for i in range(100):
        d = random.randint(0, 4)
        img = image.generate_image(str(d))
        img = img.convert('L')
        guid = uuid.uuid4()
        img.save("test/{}-{}.png".format(d, guid))

def load_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in os.listdir("train/"):
        im = Image.open(os.path.join("train", f))
        x_train.append(np.asarray(im, dtype=np.float32))
        y_train.append(f.split('-')[0])
    for f in os.listdir("test/"):
        im = Image.open(os.path.join("test", f))
        x_test.append(np.asarray(im, dtype=np.float32))
        y_test.append(f.split('-')[0])
    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

def build_model(number_classes, input_shape):
    layers = [
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(.5),
        Dense(number_classes, activation='softmax')
    ]
    model = Sequential(layers)
    return model
    

def train():
    batch_size = 128
    epochs = 12
    width = 60
    height = 100
    number_classes = 5
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], width, height, 1)
    x_test = x_test.reshape(x_test.shape[0], width, height, 1)
    input_shape = (width, height, 1)

    print x_train.shape, x_test.shape

    y_train = keras.utils.to_categorical(y_train, number_classes)
    y_test = keras.utils.to_categorical(y_test, number_classes)
    print y_train.shape, y_test.shape

    model = build_model(number_classes, input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    

if __name__ == "__main__":
    train()
