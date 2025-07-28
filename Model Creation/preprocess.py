from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

def preprocess_image(image):
    image = image.resize((64, 64))
    image = img_to_array(image).astype("float32") / 255.0
    return image

def preprocess_xray(path, target_size=(224, 224)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = img_to_array(image).astype("float32") / 255.0
    return image

