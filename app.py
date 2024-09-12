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

    'Kamalam': 'Kamalam, or the lotus, is a symbol of purity and beauty in Hindu iconography. It is often depicted as a seat or a base on which deities are seated. The lotus represents divine beauty and purity, emerging from the mud but remaining unstained. By including Kamalam in statues, it conveys the idea of spiritual purity and the transcendence of material impurities.',
    'Karanda Magudam': 'Karanda Magudam, or the Karanda crown, is a traditional headgear often seen in Hindu statues, particularly those of deities like Shiva and Vishnu. It is characterized by its unique shape resembling a bird’s beak. This crown symbolizes the divine authority and majesty of the deity. It emphasizes the deity\'s role as a supreme ruler who governs with wisdom and power.',
    'Karudaasanam': 'Karudaasanam, or the Garuda seat, is a pedestal or throne in the shape of Garuda, the mythical eagle. In Hindu statues, it is associated with deities like Vishnu, who rides Garuda as his mount. The inclusion of Karudaasanam signifies the deity’s connection with cosmic power and the ability to transcend earthly limitations.',
    'Padmasanam': 'Padmasanam, or the lotus seat, is a meditative posture where the deity is seated in a cross-legged position on a lotus flower. This pose represents spiritual enlightenment and serenity. By depicting deities in Padmasanam, the statues convey a sense of deep meditation and the attainment of higher spiritual knowledge.',
    'Soolam': 'Soolam, or the trident, is a prominent weapon often held by deities like Shiva. It symbolizes the three aspects of divine energy: creation, preservation, and destruction. The presence of Soolam in a statue represents the deity’s power to balance these cosmic forces and uphold dharma.',
    'Sugasanam': 'Sugasanam, or the comfortable seat, refers to a throne or seat that is often depicted in Hindu statues. It represents the deity’s royal and divine status, indicating their supreme authority and comfort in their divine realm. This feature underscores the deity’s grace and majesty.',
    'Udukkai': 'Udukkai, or the drum, is an instrument often depicted in the hands of deities like Shiva. It symbolizes the rhythm and cosmic sound of creation. The presence of Udukkai in statues conveys the idea of divine music and the continuous beat of cosmic rhythm that sustains the universe.',
    'Varadham': 'Varadham, or the boon-giving gesture, is depicted as a hand raised with the palm open. It symbolizes the deity’s ability to bestow blessings and fulfill the desires of devotees. In statues, Varadham conveys the deity’s benevolence and their role as a protector and benefactor.',
    'Yogasanam': 'Yogasanam refers to a posture of meditation or yoga. It is often depicted in Hindu statues to show the deity in a state of profound meditation and spiritual practice. This posture conveys the deity’s mastery over yoga and meditation, emphasizing their spiritual insight and self-realization.',
    'abaya': 'Abhaya, or the gesture of fearlessness, is represented by a raised hand with the palm facing outward. It signifies protection and reassurance. In Hindu statues, the Abhaya gesture conveys the deity’s role in safeguarding devotees from fear and danger, providing them with a sense of security and divine protection.'

    # Your descriptions
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
    image.save(image_path)

    model = YOLO("best.pt")  # Ensure 'best.pt' is accessible
    results = model(image_path)  # return a list of Results objects
    
    class_names = [model.names[int(cls)] for cls in results[0].boxes.cls]
    descriptions = [class_descriptions.get(name, "No description available") for name in class_names]
    
    # Save the output image
    output_image_path = "output.jpg"
    results[0].save(filename=output_image_path)
    
    with open(output_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    os.remove(image_path)
    os.remove(output_image_path)
    
    return jsonify({
        "class_names": class_names,
        "descriptions": descriptions,
        "image": encoded_image
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
