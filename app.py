
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request, g
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = Flask(__name__)

# --------------------------------------------------
# 1. Logging configuration (Practical 11)
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app.logger.setLevel(logging.INFO)

# --------------------------------------------------
# 2. Model configuration
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# --------------------------------------------------
# 3. Prometheus metrics (Practical 12)
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total student result prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

# Make the metric available before the first model load.
MODEL_LOADED.set(0)


# --------------------------------------------------
# 4. Model loading
# --------------------------------------------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        return model

    except Exception:
        MODEL_LOADED.set(0)
        app.logger.exception("event=model_load_failed")
        raise


# --------------------------------------------------
# 5. Request logging
# --------------------------------------------------

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


@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())
    duration = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id

    # Exclude Prometheus scrapes from normal API request metrics.
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(duration)

    app.logger.info(
        "Request completed | id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration * 1000,
    )

    return response


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# 7. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    try:
        load_model()

        return jsonify({
            "status": "healthy",
            "service": "student-result-prediction",
        }), 200

    except Exception:
        app.logger.exception("event=health_check_failed")

        return jsonify({
            "status": "unhealthy",
        }), 503


# --------------------------------------------------
# 8. Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------
# 9. Student result prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    # Validate JSON body.
    if not isinstance(data, dict):
        PREDICTIONS.labels(result="error").inc()

        app.logger.warning(
            "event=prediction_rejected reason=invalid_json"
        )

        return jsonify({
            "error": "JSON request body is required",
        }), 400

    # Validate required input fields.
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        PREDICTIONS.labels(result="error").inc()

        app.logger.warning(
            "event=prediction_rejected missing_fields=%s",
            missing_fields,
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    # Convert and validate feature values.
    try:
        values = {}

        for feature in FEATURES:
            value = data[feature]

            # Reject missing values, booleans and non-numeric values.
            if value is None or isinstance(value, bool):
                raise ValueError(
                    f"Invalid numeric value for {feature}"
                )

            values[feature] = float(value)

        sample = pd.DataFrame([values], columns=FEATURES)

        # Reject NaN and infinite values.
        if not all(
            pd.notna(value) and abs(value) != float("inf")
            for value in values.values()
        ):
            raise ValueError("Feature values must be finite numbers")

    except (TypeError, ValueError, OverflowError):
        PREDICTIONS.labels(result="error").inc()

        app.logger.warning(
            "event=prediction_rejected reason=invalid_features"
        )

        return jsonify({
            "error": "Feature values must be valid finite numbers",
        }), 400

    # Run model inference.
    try:
        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()

        app.logger.info(
            "event=prediction_succeeded result=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        app.logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed",
        }), 500


# --------------------------------------------------
# 10. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
    )
