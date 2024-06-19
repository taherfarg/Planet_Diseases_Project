from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import numpy as np
import cv2  # OpenCV for green color detection
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('model.h5')
labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
          'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
          'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
          'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
          'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
          'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
          'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
          'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
          'Tomato___healthy']

# Load disease information from a JSON file
with open('disease_info.json') as f:
    disease_info = json.load(f)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return predicted_class_index, confidence

def contains_green(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.05  # Threshold ratio of green pixels

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg"))
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg")

    if contains_green(image_path):
        processed_image = prepare_image(image, target=(224, 224))
        prediction = model.predict(processed_image).tolist()
        predicted_label = labels[np.argmax(prediction)]
        confidence = max(prediction[0])

        response = {
            'prediction': predicted_label,
            'confidence': confidence,
            'information': disease_info.get(predicted_label, "No information available.")
        }
    else:
        response = {
            'prediction': "No plant leaf detected",
            'confidence': 0,
            'information': "No plant leaf detected in the image."
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
