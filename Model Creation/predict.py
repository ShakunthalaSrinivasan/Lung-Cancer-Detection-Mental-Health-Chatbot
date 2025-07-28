from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from detector.preprocess import preprocess_image

# Load the model once
model = load_model("detector/model.h5")

def predict_lung_cancer(image_file):
    try:
        # Convert uploaded file to PIL Image
        img = Image.open(image_file).convert("RGB")

        # Preprocess the image
        img = preprocess_image(img)

        # Predict
        prediction = model.predict(np.expand_dims(img, axis=0))[0][0]

        return "Cancerous" if prediction > 0.5 else "Non-Cancerous"

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction Failed"
