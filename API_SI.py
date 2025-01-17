# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json, Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import json
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app)

# Base directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "models/selectConfig.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models/selectModel.weights.h5")
PEPPER_CONFIG_PATH = os.path.join(BASE_DIR, "models/pepperConfig.json")
PEPPER_WEIGHTS_PATH = os.path.join(BASE_DIR, "models/pepperModel.weights.h5")
POTATO_CONFIG_PATH = os.path.join(BASE_DIR, "models/potatoConfig.json")
POTATO_WEIGHTS_PATH = os.path.join(BASE_DIR, "models/potatoModel.weights.h5")
TOMATO_CONFIG_PATH = os.path.join(BASE_DIR, "models/tomatoConfig.json")
TOMATO_WEIGHTS_PATH = os.path.join(BASE_DIR, "models/tomatoModel.weights.h5")
CLASS_NAMES = ["Pepper", "Potato", "Tomato"]

def load_model(config_path=CONFIG_PATH, weights_path=WEIGHTS_PATH):
    """
    Load and configure the model from JSON config and weights files.
    Returns the configured model or raises an exception if there's an error.
    """
    try:
        # Load model architecture
        with open(config_path, "r") as config_file:
            model_config = json.load(config_file)

        # Extract the model architecture configuration
        if isinstance(model_config, dict) and "config" in model_config:
            model_architecture = model_config["config"]
        else:
            raise ValueError("Invalid model configuration format")

        # Create model from JSON string
        try:
            from tensorflow.keras.models import model_from_config
            model = model_from_config(model_architecture)
        except:
            # Alternative method if the first fails
            json_config = json.dumps(model_architecture)
            model = model_from_json(json_config)

        # Load weights
        try:
            model = Sequential.from_config(model_architecture)
            model.load_weights(weights_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def checkFiles():
    # Check if files exist
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
    if not os.path.exists(PEPPER_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {PEPPER_CONFIG_PATH}")
    if not os.path.exists(PEPPER_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {PEPPER_WEIGHTS_PATH}")
    if not os.path.exists(POTATO_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {POTATO_CONFIG_PATH}")
    if not os.path.exists(POTATO_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {POTATO_WEIGHTS_PATH}")
    if not os.path.exists(TOMATO_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {TOMATO_CONFIG_PATH}")
    if not os.path.exists(TOMATO_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {TOMATO_WEIGHTS_PATH}")

def decode_base64_image(base64_string):
    """
    Decode a base64 string into a PIL Image.
    """
    try:
        # Remove header if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        # Decode base64 string
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise ValueError("Invalid base64 image format")


def preprocess_image(image):
    """
    Preprocess a PIL Image for model prediction.
    """
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize and preprocess
        image = image.resize((256, 256))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError("Error preprocessing image")


# Load model at startup
try:
    # Check if files exist
    checkFiles()
    model = load_model()
    pepper_model = load_model(PEPPER_CONFIG_PATH, PEPPER_WEIGHTS_PATH)
    potato_model = load_model(POTATO_CONFIG_PATH, POTATO_WEIGHTS_PATH)
    tomato_model = load_model(TOMATO_CONFIG_PATH, TOMATO_WEIGHTS_PATH)
except Exception as e:
    logger.critical(f"Failed to load model at startup: {str(e)}")
    raise


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests with either file upload or base64 image data.
    """
    try: 
        # Check if the request has JSON data
        if request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "No image data found in JSON"}), 400

            # Process base64 image
            image = decode_base64_image(data["image"])

        # Check if the request has file data
        elif "file" in request.files:
            file = request.files["file"]
            if not file:
                return jsonify({"error": "Empty file"}), 400
            image = Image.open(file.stream)

        else:
            return jsonify({"error": "No image data found in request"}), 400

        # Preprocess and predict
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        print(predicted_class)
        healthy = None
        if predicted_class == "Pepper":
            healthy = pepper_model.predict(preprocessed_image)
            healthy = predict_pepper(healthy)
        elif predicted_class == "Potato":
            healthy = potato_model.predict(preprocessed_image)
            healthy = predict_potato(healthy)
        elif predicted_class == "Tomato":
            healthy = tomato_model.predict(preprocessed_image)
            healthy = predict_tomato(healthy)

        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "healthy": healthy,
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(CLASS_NAMES, prediction[0])
            },
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def predict_pepper(probs):
    predicted_class = np.argmax(probs)
    if predicted_class == 0:
        return "The pepper is not healthy"
    else:
        return "The pepper is healthy"

def predict_potato(probs):
    predicted_class = np.argmax(probs)
    if predicted_class == 0:
        return "Late blight of potato"
    elif predicted_class == 1:
        return "The potato is healthy"
    else: 
        return "Early blight of potato"
    
def predict_tomato(probs):
    predicted_class = np.argmax(probs)
    if predicted_class == 0:
        return "Late blight of tomato"
    elif predicted_class == 1:
        return "The tomato is healthy"
    else: 
        return "Early blight of tomato"

@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint with API information.
    """
    return jsonify({
        "message": "Plant Disease Classification API",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "accepts": ["multipart/form-data (file)", "application/json (base64 image)"],
                "returns": "JSON with prediction results"
            }
        },
        "supported_classes": CLASS_NAMES
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000) # Cambiar puerto 5000

