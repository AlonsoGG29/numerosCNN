"""
API Server para el modelo CNN de reconocimiento de dígitos.
Carga el modelo entrenado (modelo_digitos_postal.keras) y expone
un endpoint POST /predict que recibe una imagen y devuelve el dígito predicho.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde el frontend React

# Cargar el modelo al iniciar el servidor
MODEL_PATH = "modelo_digitos_postal.keras"
print(f"Cargando modelo desde {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")


def preprocess_image(image_bytes):
    """
    Preprocesa una imagen recibida para que sea compatible con el modelo CNN.
    - Convierte a escala de grises
    - Redimensiona a 28x28
    - Normaliza píxeles a [0, 1]
    - Invierte colores si el fondo es claro (el modelo fue entrenado con MNIST:
      fondo negro, dígito blanco)
    - Reshape a (1, 28, 28, 1) para la predicción
    """
    # Abrir imagen desde bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convertir a escala de grises
    img = img.convert("L")

    # Redimensionar a 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convertir a array numpy
    img_array = np.array(img, dtype="float32")

    # Normalizar a [0, 1]
    img_array = img_array / 255.0

    # MNIST tiene fondo negro y dígito blanco.
    # Si la imagen subida tiene fondo claro, invertir.
    mean_val = np.mean(img_array)
    if mean_val > 0.5:
        img_array = 1.0 - img_array

    # Reshape para el modelo: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint para predecir el dígito de una imagen."""
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    try:
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)

        # Realizar la predicción
        predictions = model.predict(processed, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        # Devolver todas las probabilidades por clase
        probabilities = {str(i): float(predictions[0][i]) for i in range(10)}

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para verificar que el servidor está activo."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
