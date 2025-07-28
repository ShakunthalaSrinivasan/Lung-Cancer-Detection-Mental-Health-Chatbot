from flask import Flask, request, render_template, jsonify, send_file
from openai import OpenAI
from gtts import gTTS
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from detector.predict import predict_lung_cancer  # For histopathology
# === Load X-ray model ===
xray_model = load_model("detector/xray_model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === GPT Chatbot Response ===
def get_chat_response(user_input):
    messages = [
        {"role": "system", "content":  "You are a compassionate and concise AI assistant specializing in both mental health "
            "and lung cancer support. Respond only in ENGLISH using friendly and supportive tone. "
            "You must:\n"
            "- Answer questions related to emotional well-being, depression, anxiety, etc.\n"
            "- Also answer questions about lung cancer, symptoms, diagnosis, treatment, lifestyle tips.\n"
            "- Keep responses short (2–4 lines), clear, and positive.\n"
            "If unsure about any medical condition, advise the user to consult a healthcare professional."},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# === Voice Output ===
def generate_tts(text):
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(TEMP_DIR, filename)
    tts = gTTS(text=text)
    tts.save(filepath)
    return filepath

# === Lung Cancer Prediction from X-ray ===
def predict_xray(image_file):
    try:
        print("Received file:", image_file)
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = xray_model.predict(image)[0][0]
        print("Prediction score:", prediction)
        return "Cancerous" if prediction > 0.5 else "Non-Cancerous"
    except Exception as e:
        print("X-ray prediction error:", e)
        return "Prediction Failed"

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    response = get_chat_response(user_input)
    audio_path = generate_tts(response)
    audio_file = os.path.basename(audio_path)
    return jsonify({"response": response, "audio": f"/audio/{audio_file}"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    result = predict_lung_cancer(file)
    return jsonify({"prediction": result})

@app.route("/predict_xray", methods=["POST"])
def predict_xray_route():
    if 'xray' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['xray']  # ✅ use 'xray' instead of 'image'
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    result = predict_xray(file)
    explanation = "This result is based on chest X-ray image classification. For accurate diagnosis, please consult a doctor."
    return jsonify({"prediction": result, "explanation": explanation})


@app.route("/audio/<filename>")
def audio(filename):
    filepath = os.path.join(TEMP_DIR, filename)
    return send_file(filepath, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)
