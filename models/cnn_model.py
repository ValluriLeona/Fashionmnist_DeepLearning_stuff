from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def save_cnn_model(model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "cnn_model.keras")

    print("BASE_DIR =", BASE_DIR)
    print("SAVE_PATH =", SAVE_PATH)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    model.save(SAVE_PATH)
    print("CNN model saved to:", SAVE_PATH)


# ---- RUN THIS ----
if __name__ == "__main__":
    model = build_cnn_model()
    save_cnn_model(model)
