from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pickle
import tempfile

# MediaPipe imports
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands

app = Flask(__name__)

# -------------------------------
# Load XGBoost model and scaler
# -------------------------------
with open("hand_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("class_to_idx.pkl", "rb") as f:
    class_to_idx = pickle.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# -------------------------------
# MediaPipe Hands Model
# -------------------------------
hands_model = Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_hand_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_model.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_lm = results.multi_hand_landmarks[0]
    feature = []
    for lm in hand_lm.landmark:
        feature.extend([lm.x, lm.y, lm.z])

    tensor = np.array(feature, dtype=np.float32).reshape(1, -1)
    tensor = scaler.transform(tensor)
    return tensor

# -------------------------------
# API Route
# -------------------------------
@app.route("/")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_path = tmp.name
        file.save(img_path)

    try:
        features = extract_hand_features(img_path)
        if features is None:
            return jsonify({"error": "No hand detected"}), 400

        pred_idx = model.predict(features)[0]
        pred_class = idx_to_class[pred_idx]

        return jsonify({"predicted_class": pred_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
