import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, ReLU, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
import os


# ---------------------------------
# RESIDUAL BLOCK
# ---------------------------------
def res_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x


# ---------------------------------
# BUILD MODEL
# ---------------------------------
def build_resnet():
    inp = Input(shape=(28,28,1))
    
    x = Conv2D(32, (3,3), padding='same')(inp)
    
    x = res_block(x, 32)
    x = res_block(x, 32)
    
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------
# LOAD DATA
# ---------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)


# ---------------------------------
# TRAIN MODEL
# ---------------------------------
model = build_resnet()
print("Training ResNet...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
print("Training complete!")


# ---------------------------------
# SAVE MODEL
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "resnet_model.keras")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

model.save(SAVE_PATH)
print("\nResNet model saved to:", SAVE_PATH)
