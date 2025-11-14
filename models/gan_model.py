import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten


def build_generator():
  model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(784, activation='sigmoid'),
    Reshape((28,28,1))
  ])
  return model


def build_discriminator():
  model = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model