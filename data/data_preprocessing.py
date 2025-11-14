import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def load_data():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


  x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
  x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


  return x_train, y_train, x_test, y_test


if __name__ == "__main__":
  x_train, y_train, x_test, y_test = load_data()
  print("Training set:", x_train.shape, y_train.shape)