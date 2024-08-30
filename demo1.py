import streamlit as st
from ultralytics import YOLO
import os
import base64
from PIL import Image

# Set a directory to temporarily save the uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO("best.pt")

def save_image(uploaded_file):
    filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filename

def perform_detection(image_path):
    # Run YOLO detection model on the saved image
    results = model(image_path)

    for result in results:
        class_names = [model.names[int(cls)] for cls in result.boxes.cls]
        result.save(filename="output.jpg")
    
    return class_names

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

# Streamlit UI
st.title("Object Detection with YOLO")
st.write("Upload an image to perform object detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = save_image(uploaded_file)
    st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        class_names = perform_detection(image_path)
        encoded_image = get_base64_image("output.jpg")

        st.write(f"Detected class names: {', '.join(class_names)}")
        st.image("output.jpg", caption="Detected Objects", use_column_width=True)

        # Optionally display the base64 string
        st.text_area("Base64 Encoded Image", encoded_image)

    # Clean up the uploaded and output files
    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists("output.jpg"):
        os.remove("output.jpg")
