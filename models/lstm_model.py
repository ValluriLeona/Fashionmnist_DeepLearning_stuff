from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os


def build_lstm():
    model = Sequential([
        LSTM(128, input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ----------------------------------------------------
# SAVE MODEL (same as other model files)
# ----------------------------------------------------

def save_lstm_model(model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "lstm_model.keras")

    # Create "saved_models" folder if needed
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    model.save(SAVE_PATH)
    print("LSTM model saved to:", SAVE_PATH)


# ----------------------------------------------------
# RUN WHEN EXECUTED DIRECTLY
# ----------------------------------------------------
if __name__ == "__main__":
    lstm = build_lstm()
    save_lstm_model(lstm)
