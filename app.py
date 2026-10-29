
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()]
)

app.logger.setLevel(logging.INFO)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


# Log each incoming request
@app.before_request
def log_request_start():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.perf_counter()

    app.logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


# Log response status and execution time
@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    app.logger.info(
        "Request completed | id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


@app.get("/health")
def health_check():
    try:
        load_model()
        return jsonify({
            "status": "healthy",
            "service": "student-result-prediction",
        }), 200
    except (FileNotFoundError, OSError, ValueError):
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "unhealthy",
        }), 503


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        app.logger.warning("Prediction rejected: JSON body is missing or invalid")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        app.logger.warning(
            "Prediction rejected: missing fields=%s",
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        app.logger.info(
            "Prediction completed | result=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except (TypeError, ValueError):
        app.logger.warning("Prediction rejected: invalid numeric feature values")
        return jsonify({
            "error": "Feature values must be valid numeric inputs"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
