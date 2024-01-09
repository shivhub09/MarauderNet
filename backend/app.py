from flask import Flask
import cv2
import tensorflow as tf
import numpy as np


app = Flask(__name__)

model = tf.keras.models.load_model('golden_trio.h5')

@app.route('/predictimage/<path:image_path>')
def predictImage(image_path):
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)
    max_index = np.argmax(yhat[0])
    if max_index == 0:
        print("RON")
    elif max_index == 1:
        print("Harry Potter")
    elif max_index == 2:
        print("Hermoine")
    else:
        print("Error")

if __name__  == '__main__':
    app.run()