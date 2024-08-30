import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Load the model
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    try:
        # Run prediction
        results = model(image_path)
        detected_classes = [r.names for r in results]  # Extract class names
        return jsonify({"detected_classes": detected_classes, "image_path": image_path})
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": "An error occurred during detection"}), 500


