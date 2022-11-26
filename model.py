import tensorflow as tf

import numpy as np
from numpy import asarray
from PIL import Image
from io import BytesIO

model = tf.keras.models.load_model('model')

classes = ["Lucicutiidae", "Mecynocera", "Mysida", "Ostracoda", 
            "Pleuromamma", "Pontellidae", "Rhincalanidae", "Sapphirina", 
            "Scolecitrichidae", "Sergestidae", "Subeucalanidae", "Temoridae", 
            "Acartiidae", "Aetideidae", "Calocalanus", "Calyptopsis", 
            "Candaciidae", "Centropagidae", "Cladocera", "Copilia", 
            "Eucalanidae", "Euchaetidae", "Haloptilus", "Harpacticoida"]

classes = sorted(classes)

color_mode = 'RGB'
img_height, img_width = 90, 90

def classify(content):
    img = Image.open(BytesIO(content))
    img = img.convert(color_mode)
    img = img.resize((img_height, img_width), Image.NEAREST)
    img_arr = asarray(img)
    img_arr = tf.expand_dims(img_arr, 0)     
    prediction = model.predict(img_arr)
    return {
        "prediction": classes[np.argmax(prediction)], 
        "confidence": (max(prediction[0])/1)
    }
