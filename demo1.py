# app.py

import streamlit as st
import os
import base64
from ultralytics import YOLO
from PIL import Image

# Ensure the model file is in the correct directory or provide the correct path
MODEL_PATH = 'best.pt'

# Load the YOLO model
model = YOLO(MODEL_PATH)  # Make sure 'best.pt' is available in the deployment environment

def detect_objects(image):
    # Save the uploaded image
    filename = "uploaded_image.jpg"
    image_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    image.save(image_path)

    # Run the YOLO detection model on the saved image
    results = model(image_path)  # return a list of Results objects

    # Extract detected class names
    class_names = []
    for result in results:
        class_names = [model.names[int(cls)] for cls in result.boxes.cls]
        result.save(filename="output.jpg")  # Save the output image

    # Encode the output image to base64 string
    with open("output.jpg", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(image_path)  # Clean up uploaded image
    return class_names, encoded_image

# Streamlit app
st.title("Object Detection with YOLO")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Perform object detection
    class_names, encoded_image = detect_objects(uploaded_file)

    # Display results
    st.write("Detected class names:", class_names)

    # Convert base64-encoded image to displayable format
    st.image(base64.b64decode(encoded_image), caption="Detected Objects Image.", use_column_width=True)
