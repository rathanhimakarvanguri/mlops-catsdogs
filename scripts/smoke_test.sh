#!/usr/bin/env bash
# Post-deploy smoke test: checks /health, then sends one real prediction
# request. Exits non-zero (failing the CD pipeline) on any failure.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
TEST_IMAGE="${2:-data/raw/cats/cats_000.jpg}"

echo "Smoke test target: $BASE_URL"

echo "-> checking /health"
HEALTH_STATUS=$(curl -s -o /tmp/health.json -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH_STATUS" != "200" ]; then
  echo "FAIL: /health returned $HEALTH_STATUS"
  cat /tmp/health.json || true
  exit 1
fi
echo "health OK: $(cat /tmp/health.json)"

if [ ! -f "$TEST_IMAGE" ]; then
  echo "WARN: sample image $TEST_IMAGE not found, skipping /predict check"
  exit 0
fi

echo "-> checking /predict"
PREDICT_STATUS=$(curl -s -o /tmp/predict.json -w "%{http_code}" \
  -X POST "$BASE_URL/predict" -F "file=@${TEST_IMAGE}")
if [ "$PREDICT_STATUS" != "200" ]; then
  echo "FAIL: /predict returned $PREDICT_STATUS"
  cat /tmp/predict.json || true
  exit 1
fi
echo "predict OK: $(cat /tmp/predict.json)"

echo "Smoke test PASSED"
