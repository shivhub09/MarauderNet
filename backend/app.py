from flask import Flask, request
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('golden_trio.h5')

@app.route('/predictimage', methods=['POST'])
def predict_image():
    image_file = request.files['image']
    
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (256, 256))
    resized_image = np.expand_dims(resized_image, axis=0)
    resized_image = resized_image.astype(np.float32) / 255.0
    
    yhat = model.predict(resized_image)
    max_index = np.argmax(yhat[0])
    
    if max_index == 0:
        return "Ron Weasley"
    elif max_index == 1:
        return "Harry Potter"
    elif max_index == 2:
        return "Hermoine Granger"
    else:
        return "Error"

if __name__ == '__main__':
    app.run()
