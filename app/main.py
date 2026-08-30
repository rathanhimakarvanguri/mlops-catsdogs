"""
FastAPI inference service for the Cats-vs-Dogs classifier.

Endpoints:
    GET  /health   -> liveness/readiness probe
    POST /predict  -> multipart image upload -> {label, probabilities}
    GET  /metrics  -> Prometheus-format counters (request count, latency)

Logging: every request is logged as a single structured JSON line to
stdout (path, status, latency_ms, predicted_label). Raw image bytes and
any client-identifying data are never logged.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from model_utils import load_model, predict, preprocess_image  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).resolve().parent / "model.pt"))

logger = logging.getLogger("catsdogs_api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.2")

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total inference requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model weights not found at {MODEL_PATH}. Run src/train.py first."
            )
        _model = load_model(MODEL_PATH)
    return _model


@app.on_event("startup")
def _startup():
    try:
        get_model()
        logger.info(json.dumps({"event": "startup", "status": "model_loaded"}))
    except Exception as e:
        logger.warning(json.dumps({"event": "startup", "status": "model_missing", "error": str(e)}))


@app.get("/health")
def health():
    model_ready = Path(MODEL_PATH).exists()
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    start = time.time()
    endpoint = "/predict"
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
        model = get_model()
        label, probs = predict(model, tensor)
        latency = time.time() - start

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        logger.info(json.dumps({
            "event": "prediction",
            "endpoint": endpoint,
            "predicted_label": label,
            "latency_ms": round(latency * 1000, 2),
        }))

        return {
            "label": label,
            "probabilities": {cls: float(p) for cls, p in zip(["cat", "dog"], probs)},
        }
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        logger.error(json.dumps({"event": "prediction_error", "error": str(e)}))
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
