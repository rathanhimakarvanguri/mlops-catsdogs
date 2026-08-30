"""
M5 - Model Performance Tracking (Post-Deployment)

Simulates collecting a small batch of live requests + true labels (in a
real deployment these would come from the request log / a feedback table)
and recomputes accuracy/precision/recall so degradation over time is
visible.

Usage:
    python scripts/eval_drift.py --base-url http://localhost:8000 \
        --labeled-dir data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score

LABEL_MAP = {"cats": "cat", "dogs": "dog"}


def collect_labeled_sample(labeled_dir: Path, n_per_class: int = 10):
    samples = []
    for folder, true_label in LABEL_MAP.items():
        files = sorted((labeled_dir / folder).glob("*"))[:n_per_class]
        samples.extend((f, true_label) for f in files)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--labeled-dir", default="data/raw")
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--out", default="artifacts/drift_report.json")
    args = parser.parse_args()

    samples = collect_labeled_sample(Path(args.labeled_dir), args.n_per_class)
    if not samples:
        raise SystemExit("No labeled samples found — check --labeled-dir")

    y_true, y_pred = [], []
    for path, true_label in samples:
        with open(path, "rb") as f:
            resp = requests.post(f"{args.base_url}/predict", files={"file": f}, timeout=10)
        resp.raise_for_status()
        pred_label = resp.json()["label"]
        y_true.append(true_label)
        y_pred.append(pred_label)

    report = {
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dog": precision_score(y_true, y_pred, pos_label="dog", zero_division=0),
        "recall_dog": recall_score(y_true, y_pred, pos_label="dog", zero_division=0),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
