import os
import tensorflow as tf
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image: Image.Image):
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        return "Abnormal (Fracture)", float(pred)
    else:
        return "Normal", float(1 - pred)
