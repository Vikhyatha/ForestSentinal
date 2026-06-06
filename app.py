from flask import Flask, render_template, request, jsonify, url_for, session
from werkzeug.utils import secure_filename
import os
import time
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
from datetime import datetime
from groq import Groq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle
try:
    import joblib
except Exception:
    joblib = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "forestsentinel-dev-secret")

AUDIO_DIR = "audio"
STATIC_DIR = "static"
MODEL_PATH_PICKLE = "sound_model.pkl"
MODEL_PATH_JOBLIB = "sound_model.joblib"
LAST_RESULT_FILE = "last_result.json"

MAX_CHAT_MESSAGE_LEN = 1000

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_model():
    if os.path.exists(MODEL_PATH_PICKLE):
        try:
            with open(MODEL_PATH_PICKLE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("Could not load sound_model.pkl:", e)

    if os.path.exists(MODEL_PATH_JOBLIB) and joblib is not None:
        try:
            return joblib.load(MODEL_PATH_JOBLIB)
        except Exception as e:
            print("Could not load sound_model.joblib:", e)

    return None


def load_last_result():
    if os.path.exists(LAST_RESULT_FILE):
        try:
            with open(LAST_RESULT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "file_name": "-",
        "result": "No analysis yet",
        "confidence": 0,
        "direction": "Unknown",
        "spectrogram_url": "",
        "class_image": "",
        "duration_sec": 0,
        "sample_rate": 0,
        "rms_energy": 0,
        "zero_crossing_rate": 0,
        "spectral_centroid": 0,
        "top3_predictions": [],
        "confidence_level": "Unknown",
        "alert_level": "Normal",
        "analysis_time": "-",
        "recommendation": "Upload an audio file to start analysis.",
        "warning": ""
    }


def save_last_result(data):
    with open(LAST_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1), y, sr


def generate_spectrogram(y, sr, out_path):
    plt.figure(figsize=(8, 4))
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(d, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def estimate_direction(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    if len(centroid) < 2:
        return "Unknown"
    slope = np.polyfit(np.arange(len(centroid)), centroid, 1)[0]
    if slope > 2:
        return "Toward sensor"
    if slope < -2:
        return "Away from sensor"
    return "Stable / unclear"


def get_confidence(m, features):
    if hasattr(m, "predict_proba"):
        probs = m.predict_proba(features)[0]
        return float(np.max(probs) * 100)

    if hasattr(m, "decision_function"):
        score = np.array(m.decision_function(features)).ravel()
        conf = 1.0 / (1.0 + np.exp(-np.max(score)))
        return float(conf * 100)

    return 75.0


def get_top3_predictions(m, features, predicted_label, fallback_conf):
    if hasattr(m, "predict_proba") and hasattr(m, "classes_"):
        probs = m.predict_proba(features)[0]
        classes = m.classes_
        idxs = np.argsort(probs)[::-1][:3]
        top3 = []
        for idx in idxs:
            top3.append({
                "label": str(classes[idx]),
                "confidence": round(float(probs[idx] * 100), 2)
            })
        return top3

    return [{
        "label": str(predicted_label),
        "confidence": round(float(fallback_conf), 2)
    }]


def get_class_image_url(pred_label):
    label = str(pred_label).strip().lower()
    image_map = {
        "bird": "bird.png",
        "animal": "animal.png",
        "chainsaw": "chainsaw.png"
    }
    img_name = image_map.get(label, "")
    if not img_name:
        return ""
    img_path = os.path.join(STATIC_DIR, img_name)
    if not os.path.exists(img_path):
        return ""
    return url_for("static", filename=img_name)


def confidence_level(conf):
    if conf >= 85:
        return "High"
    if conf >= 65:
        return "Medium"
    return "Low"


def alert_level(pred, conf):
    p = str(pred).lower()
    if p == "chainsaw" and conf >= 70:
        return "Critical"
    if p == "animal" and conf >= 70:
        return "Warning"
    return "Normal"


def recommendation(pred, conf, direction):
    p = str(pred).lower()
    if p == "chainsaw" and conf >= 70:
        return "Possible illegal logging activity detected. Verify location immediately."
    if p == "bird" and conf >= 70:
        return "Bird activity detected. Mark as biodiversity presence zone."
    if p == "animal" and conf >= 70:
        return "Wildlife movement detected. Monitor this area and reduce disturbance."
    return f"Low-confidence signal. Collect another sample. Direction: {direction}."


def extract_audio_stats(y, sr):
    duration = len(y) / sr if sr else 0
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return {
        "duration_sec": round(duration, 2),
        "sample_rate": int(sr),
        "rms_energy": round(rms, 6),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid": round(centroid, 2)
    }


model = load_model()
LAST_RESULT = load_last_result()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("main.html")


@app.route("/images-page")
def images_page():
    return render_template("images_page.html", data=LAST_RESULT)


@app.route("/analysis-page")
def analysis_page():
    return render_template("analysis_page.html", data=LAST_RESULT)


@app.route("/last-result")
def last_result():
    has_data = bool(LAST_RESULT.get("spectrogram_url") or LAST_RESULT.get("class_image"))
    return jsonify({"has_data": has_data, "data": LAST_RESULT})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "ok": False,
            "error": "Message is required."
        }), 400

    if len(message) > MAX_CHAT_MESSAGE_LEN:
        return jsonify({
            "ok": False,
            "error": "Message is too long."
        }), 400

    try:
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are Forest AI Assistant for ForestSentinel."
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        reply = completion.choices[0].message.content

        return jsonify({
            "ok": True,
            "reply": reply
        })

    except Exception as e:
        print("CHAT ERROR:", e)

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    global LAST_RESULT
    try:
        if model is None:
            return jsonify({"result": "Model not loaded", "confidence": 0, "direction": "Unknown"}), 500

        if "audio" not in request.files:
            return jsonify({"result": "No file uploaded", "confidence": 0, "direction": "Unknown"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"result": "No file selected", "confidence": 0, "direction": "Unknown"}), 400

        original_name = secure_filename(file.filename)
        unique_name = f"{int(time.time() * 1000)}_{original_name}"
        audio_path = os.path.join(AUDIO_DIR, unique_name)
        file.save(audio_path)

        features, y, sr = extract_features(audio_path)
        pred = model.predict(features)[0]
        conf = get_confidence(model, features)
        direction = estimate_direction(y, sr)

        stats = extract_audio_stats(y, sr)
        top3 = get_top3_predictions(model, features, pred, conf)
        conf_level = confidence_level(conf)
        alert = alert_level(pred, conf)

        warning_msg = ""
        if conf < 70:
            warning_msg = "Low confidence - please recheck with another audio sample."

        spec_name = f"spectrogram_{int(time.time() * 1000)}.png"
        spec_path = os.path.join(STATIC_DIR, spec_name)
        generate_spectrogram(y, sr, spec_path)

        LAST_RESULT = {
            "file_name": original_name,
            "result": str(pred),
            "confidence": round(conf, 2),
            "direction": direction,
            "spectrogram_url": url_for("static", filename=spec_name),
            "class_image": get_class_image_url(pred),
            "duration_sec": stats["duration_sec"],
            "sample_rate": stats["sample_rate"],
            "rms_energy": stats["rms_energy"],
            "zero_crossing_rate": stats["zero_crossing_rate"],
            "spectral_centroid": stats["spectral_centroid"],
            "top3_predictions": top3,
            "confidence_level": conf_level,
            "alert_level": alert,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendation": recommendation(pred, conf, direction),
            "warning": warning_msg
        }

        save_last_result(LAST_RESULT)
        return jsonify(LAST_RESULT)

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({
            "result": f"Error processing audio: {str(e)}",
            "confidence": 0,
            "direction": "Unknown",
            "spectrogram_url": "",
            "class_image": ""
        }), 500


if __name__ == "__main__":
    app.run(debug=True)