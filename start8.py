import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import uvicorn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cloudinary
from cloudinary.uploader import upload
import base64

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('last_face_model.h5', compile=False)


# Configure Cloudinary with your credentials
cloudinary.config( 
	  cloud_name = "dcysfieol", 
	  api_key = "851589193853581", 
	  api_secret = "J2FZWZLTigmfpt9VEozTm7tbzFE" 
	)

def upload_to_cloudinary(image_path):
    response = upload(image_path)
    return response['secure_url']  # Return the URL of the uploaded image


@app.route('/predict_face', methods=['POST'])
def predict_emotion():
    try:
        file = request.files['file']
        contents = file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        img = cv2.resize(gray, (224, 224))
        img_to_save = cv2.resize(frame, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        prediction = prediction[0][0]
        # prediction = tf.nn.softmax(prediction)
        print(prediction)
        # prediction = np.argmax(prediction)
        pred = "Human" if prediction < 0.05 else "Non-Human"
        cloudinary_url = ""
        if pred == "Human":
        	# f = open('captured_image.jpg', 'wb')
        	# f.write(frame)
        	cv2.imwrite('captured_image.jpg', img_to_save)
        	cloudinary_url = upload_to_cloudinary("captured_image.jpg")
        	
        if cloudinary_url is not None:
        	response = {"image_url":cloudinary_url}
        else:
        	response = {"pred": pred}
        return jsonify(response), 200
    except Exception as e:
    	print(e)
    	return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

