# %tensorflow_version 1.x
import keras

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint


def get_model(input_shape):
    model = Sequential([
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=1),
        Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        BatchNormalization(axis=1),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(9, activation='softmax')
    ])

    return model

