from flask import Flask, request, jsonify, render_template
import os
import base64
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set a directory to temporarily save the uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary for class descriptions
class_descriptions = {
    'Kamalam': 'Kamalam, or the lotus, is a symbol of purity and beauty in Hindu iconography...',
    'Karanda Magudam': 'Karanda Magudam, or the Karanda crown, is a traditional headgear...',
    'Karudaasanam': 'Karudaasanam, or the Garuda seat, is a pedestal or throne in the shape of Garuda...',
    'Padmasanam': 'Padmasanam, or the lotus seat, is a meditative posture...',
    'Soolam': 'Soolam, or the trident, is a prominent weapon often held by deities like Shiva...',
    'Sugasanam': 'Sugasanam, or the comfortable seat, refers to a throne or seat...',
    'Udukkai': 'Udukkai, or the drum, is an instrument often depicted in the hands of deities...',
    'Varadham': 'Varadham, or the boon-giving gesture, is depicted as a hand raised...',
    'Yogasanam': 'Yogasanam refers to a posture of meditation or yoga...',
    'abaya': 'Abhaya, or the gesture of fearlessness, is represented by a raised hand...'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Try saving the image
    try:
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    try:
        model = YOLO("best.pt")  # Load the model
        results = model(image_path)  # Perform detection
    except Exception as e:
        return jsonify({"error": f"Failed to perform detection: {str(e)}"}), 500

    class_names = [model.names[int(cls)] for cls in results[0].boxes.cls]
    descriptions = [class_descriptions.get(name, "No description available") for name in class_names]
    
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
    
    try:
        results[0].save(filename=output_image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save output image: {str(e)}"}), 500

    try:
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"Failed to encode output image: {str(e)}"}), 500

    os.remove(image_path)
    os.remove(output_image_path)

    return jsonify({
        "class_names": class_names,
        "descriptions": descriptions,
        "image": encoded_image
    })

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
    app.run(host='0.0.0.0', port=5000)
