from tensorflow.keras.optimizers import SGD, Adam, RMSprop


def get_optimizers():
  return {
    "sgd": SGD(learning_rate=0.01, momentum=0.9),
    "adam": Adam(learning_rate=0.001),
    "rmsprop": RMSprop(learning_rate=0.0005)
  }