from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ----------------------------------------------------
# LOAD ALL SAVED MODELS
# ----------------------------------------------------

MODEL_DIR = "saved_models"

models = {
    "Sequential Model": load_model(os.path.join(MODEL_DIR, "sequential_model.keras")),
    "CNN Model": load_model(os.path.join(MODEL_DIR, "cnn_model.keras")),
    "RNN Model": load_model(os.path.join(MODEL_DIR, "rnn_model.keras")),
    "LSTM Model": load_model(os.path.join(MODEL_DIR, "lstm_model.keras")),
    "VGG-like Model": load_model(os.path.join(MODEL_DIR, "vgg_model.keras")),
    "ResNet Model": load_model(os.path.join(MODEL_DIR, "resnet_model.keras")),
    "Attention Network": load_model(os.path.join(MODEL_DIR, "attention_network.keras"))
}

# ----------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# ----------------------------------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1) / 255.0
    return img


@app.route('/')
def index():
    return render_template("index.html")


# ----------------------------------------------------
# PREDICT WITH ALL MODELS
# ----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No image selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    processed_img = preprocess_image(filepath)

    # Store results as list of dicts
    predictions_output = []

    for model_name, model in models.items():
        preds = model.predict(processed_img)
        class_index = np.argmax(preds)
        class_name = class_names[class_index]
        confidence = float(np.max(preds)) * 100

        predictions_output.append({
            "model": model_name,
            "prediction": class_name,
            "confidence": round(confidence, 2)
        })

    return render_template(
        "index.html",
        uploaded_image=filepath,
        results=predictions_output
    )


if __name__ == "__main__":
    app.run(debug=True)
