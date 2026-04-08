"""
app.py — Flask Backend API
============================
Serves the /predict endpoint and hosts the frontend static files.
Loads the trained Logistic Regression model on startup.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import preprocess_single_input, MANDATORY_FIELDS, DEFAULTS
from utils.explainer import explain_prediction, get_feature_importance
from utils.recommendations import get_risk_level, get_recommendations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "heart_disease_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "model", "feature_names.pkl")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# ---------------------------------------------------------------------------
# Load model artefacts
# ---------------------------------------------------------------------------
print("Loading model artefacts...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)
print(f"[OK] Model loaded — {len(feature_names)} features")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serve static frontend files (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict heart disease risk.

    Expects JSON body with patient parameters.
    Mandatory: age, sex, cp
    Optional: trestbps, chol, fbs, restecg, thalch, exang, oldpeak
    """
    try:
        data = request.get_json(force=True)

        # Validate mandatory fields
        missing = [f for f in MANDATORY_FIELDS if f not in data or data[f] in (None, "")]
        if missing:
            return jsonify({
                "error": f"Missing mandatory fields: {', '.join(missing)}",
                "mandatory_fields": MANDATORY_FIELDS,
            }), 400

        # Preprocess input
        X_input = preprocess_single_input(data, scaler, feature_names)

        # Predict probability
        probability = float(model.predict_proba(X_input)[0][1])
        risk_score = round(probability, 4)
        risk_level = get_risk_level(probability)

        # Build raw_input dict for explanations (with numeric values)
        raw_input = {}
        for key in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
            raw_input[key] = data.get(key, DEFAULTS.get(key, ""))
        for key in ["sex", "fbs", "exang", "cp", "restecg"]:
            raw_input[key] = data.get(key, DEFAULTS.get(key, ""))

        # Generate explanations
        explanations, contributions = explain_prediction(
            model, X_input, feature_names, raw_input
        )

        # Get recommendations
        recommendations = get_recommendations(risk_level, raw_input)

        # Feature importance (global)
        importance = get_feature_importance(model, feature_names)

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": explanations,
            "recommendations": recommendations,
            "feature_contributions": contributions,
            "feature_importance": importance[:8],  # Top 8
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return global feature importance from the trained model."""
    importance = get_feature_importance(model, feature_names)
    return jsonify({"feature_importance": importance})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_features": len(feature_names)})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
