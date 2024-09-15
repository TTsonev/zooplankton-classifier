import tensorflow as tf

import numpy as np
from numpy import asarray
from PIL import Image
from io import BytesIO

model = tf.keras.models.load_model('../data/model.zip')

CLASSES = sorted(["Lucicutiidae", "Mecynocera", "Mysida", "Ostracoda", 
            "Pleuromamma", "Pontellidae", "Rhincalanidae", "Sapphirina", 
            "Scolecitrichidae", "Sergestidae", "Subeucalanidae", "Temoridae", 
            "Acartiidae", "Aetideidae", "Calocalanus", "Calyptopsis", 
            "Candaciidae", "Centropagidae", "Cladocera", "Copilia", 
            "Eucalanidae", "Euchaetidae", "Haloptilus", "Harpacticoida"])

IMG_HEIGHT, IMG_WIDTH = 90, 90

COLOR_MODE = 'RGB'

def classify(content):
    img = convert_img_to_tf_array(content)
    prediction = model.predict(img)
    return {
        "prediction": CLASSES[np.argmax(prediction)], 
        "confidence": (max(prediction[0])/1)
    }

def convert_img_to_tf_array(content):
    img = Image.open(BytesIO(content))
    img = img.convert(COLOR_MODE).resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
    return tf.expand_dims(asarray(img), 0) 
