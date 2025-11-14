from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import os

def build_vgg_like():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),

        Flatten(),
        Dense(128, activation='relu'),
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

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0


# ------------------------------
# BUILD MODEL
# ------------------------------
model = build_vgg_like()

print("Training VGG-like model...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
print("Training complete!")


# ------------------------------
# SAVE MODEL
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "vgg_model.keras")

# Create directory if not exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

model.save(SAVE_PATH)

print("\nVGG-like model saved to:", SAVE_PATH)
