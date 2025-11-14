import numpy as np
from data_preprocessing import load_data


# Random sampling mini-batches


def get_random_minibatch(x, y, batch_size=64):
  idx = np.random.randint(0, x.shape[0], batch_size)
  return x[idx], y[idx]


if __name__ == "__main__":
  x_train, y_train, _, _ = load_data()
  xb, yb = get_random_minibatch(x_train, y_train)
  print("Mini-batch:", xb.shape, yb.shape)