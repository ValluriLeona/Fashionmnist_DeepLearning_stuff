from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


# Basic Autoencoder


def build_autoencoder():
  input_img = Input(shape=(784,))
  encoded = Dense(128, activation='relu')(input_img)
  decoded = Dense(784, activation='sigmoid')(encoded)


  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adam', loss='mse')
  return autoencoder