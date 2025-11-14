from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
import os

def build_rnn():
    model = Sequential([
        SimpleRNN(128, input_shape=(28,28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------------------------
# LOAD DATA
# ------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0


# ------------------------------
# BUILD + TRAIN
# ------------------------------
model = build_rnn()

print("Training RNN model...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
print("Training complete!")


# ------------------------------
# SAVE MODEL
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "rnn_model.keras")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

model.save(SAVE_PATH)

print("\nRNN model saved to:", SAVE_PATH)
