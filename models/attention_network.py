import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Multiply
import os


def build_attention_network():
    inputs = tf.keras.Input(shape=(28, 28))

    # Compute attention weights
    score = Dense(28)(inputs)
    attention_weights = Softmax(axis=1)(score)

    # Weighted feature map
    context_vector = Multiply()([inputs, attention_weights])

    x = tf.keras.layers.Flatten()(context_vector)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ----------------------------------------------------
# SAVE MODEL (same pattern as all other models)
# ----------------------------------------------------

def save_attention_network(model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "attention_network.keras")

    # Create directory if needed
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    model.save(SAVE_PATH)
    print("Attention Network saved to:", SAVE_PATH)


# ----------------------------------------------------
# RUN WHEN FILE IS EXECUTED DIRECTLY
# ----------------------------------------------------
if __name__ == "__main__":
    model = build_attention_network()
    save_attention_network(model)
