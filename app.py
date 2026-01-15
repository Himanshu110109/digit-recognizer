from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("digit_model.h5")

@app.route("/")
def home():
    return render_template("digit_canvas_app.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.json
    img_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(img_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(image, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 784)

    pred = model.predict(img)
    digit = int(np.argmax(pred))

    return jsonify({"digit": digit})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
