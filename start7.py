

from flask import Flask, request, render_template
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.nasnet import preprocess_input
import base64

app = Flask(__name__)

# Load TensorFlow model
model = tf.keras.models.load_model('last_face_model.h5', compile=False)

@app.route("/")
def read_root():
    return render_template("index.html")

def predict_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = prediction[0][0]
    # prediction = tf.nn.softmax(prediction)
    print(prediction)
    # prediction = np.argmax(prediction)
    return "Human" if prediction < 0.05 else "Non-Human"
"""
@app.route("/recognize", methods=["POST"])
def recognize():
    image_data = request.get_data()
    image_data = base64.b64decode(image_data.split(",")[1])
    image = io.BytesIO(image_data)
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = prediction[0][0]
    # prediction = tf.nn.softmax(prediction)
    print(prediction)
    # prediction = np.argmax(prediction)
    return prediction
"""
@app.route("/recognize", methods=["POST"])
def recognize():
    image_data = request.get_data()
    image_data = base64.b64decode(image_data.split(",")[1])
    image = io.BytesIO(image_data)
    result = predict_image(image)
    if result == "Human":
        with open("saved_image.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))
    else:
        pass
    return {"prediction": result}
