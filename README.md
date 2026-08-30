# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification (cats vs. dogs) for a pet adoption platform, built as
an end-to-end MLOps pipeline: data versioning, model training with experiment
tracking, packaging, containerization, and CI/CD-based deployment.

**Course:** MLOps (S1-25_AIMLCZG523) — Assignment 2

## Project structure

```
mlops-catsdogs/
├── data/raw/{cats,dogs}/       # raw images (DVC-tracked, not committed to git)
├── data/processed/             # train/val/test .npz tensors (generated)
├── src/
│   ├── preprocess.py           # resize, augment, 80/10/10 split
│   ├── model_utils.py          # SimpleCNN + shared pre/post-processing
│   └── train.py                # training loop + MLflow experiment tracking
├── app/
│   └── main.py                 # FastAPI inference service
├── tests/                      # pytest unit tests
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── smoke_test.sh           # post-deploy health + prediction check
│   └── eval_drift.py           # post-deployment performance tracking
├── k8s/                        # Deployment + Service manifests (optional target)
├── monitoring/prometheus.yml
├── .github/workflows/{ci.yml, cd.yml}
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── dvc.yaml, .dvc/config
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# get a dataset into data/raw/{cats,dogs}/ — either the real Kaggle dataset
# (https://www.kaggle.com/datasets/tongpython/cat-and-dog) or a quick synthetic
# stand-in for a dry run:
python scripts/generate_synthetic_data.py

python src/preprocess.py
cd src && python train.py --epochs 5 && cd ..
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/raw/cats/<some_image>.jpg"
```

## Run in Docker

```bash
docker build -t catsdogs-api:local .
docker run -p 8000:8000 catsdogs-api:local
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): on every push/PR — installs deps, trains a
  model, runs pytest, builds the Docker image, and pushes it to GHCR.
- **CD** (`.github/workflows/cd.yml`): triggered after CI succeeds on `main` —
  pulls the image, deploys via Docker Compose, and runs `scripts/smoke_test.sh`
  (fails the pipeline if the health check or a live prediction fails).

## Monitoring

- `/metrics` exposes Prometheus-format request count and latency.
- `scripts/eval_drift.py` hits the deployed `/predict` endpoint with labeled
  samples and reports accuracy/precision/recall for post-deployment tracking.

## Experiment tracking

Training runs (params, metrics, loss curve, confusion matrix, model artifact)
are logged with MLflow:

```bash
mlflow ui --port 5001
```
