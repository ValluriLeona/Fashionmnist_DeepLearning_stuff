from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import os


# ------------------------------
#  BUILD REGULARIZED MODEL
# ------------------------------

def build_regularized_model():
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(784,)),
        Dropout(0.3),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),

        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------
#  SAVE MODEL (same style as others)
# ------------------------------

def save_regularized_model(model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "regularized_model.keras")

    # Create directory if not exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    model.save(SAVE_PATH)
    print("Regularized model saved to:", SAVE_PATH)
