from flask import Flask, request, jsonify,render_template
import os
import re
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
app = Flask(__name__)

# Set a directory to temporarily save the uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to perform object detection and process results
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Get the image file from the request
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400

    # Secure the filename and save the image
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    # # Set up paths and configuration
    # model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'best.pt')
    
    # # Run the YOLO detection model on the saved image
    # command = f"yolo task=detect mode=predict model={model_path} conf=0.25 source={image_path} save=True"
    # output = os.popen(command).read().splitlines()
    
    model = YOLO("best.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(image_path)  # return a list of Results objects
    
    for result in results:
        class_names = [model.names[int(cls)] for cls in result.boxes.cls]
        result.save(filename="output.jpg")
        print("Detected class names:", class_names)
    
    # result = output[5].split(":")[1].split(",")[:-1]
    # output_list = [extract_words(item) for item in result]
    # output_list[0] = ''.join(output_list[0][1:])
    # Encode the output image to base64 string
    with open("output.jpg", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    os.remove(image_path)
    # Return class names and base64-encoded image
    return jsonify({
        "class_names": class_names,
        "image": encoded_image
    })
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
