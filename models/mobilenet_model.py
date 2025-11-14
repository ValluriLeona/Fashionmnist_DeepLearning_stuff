from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


def build_mobilenet():
  base = MobileNetV2(input_shape=(96,96,3), include_top=False)
  base.trainable = False


  model = Sequential([
    base,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])


  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model