# fraud-detection-mlops

> Real-time fraud detection API with a production-grade MLOps pipeline.
> XGBoost + LightGBM ensemble trained on the IEEE-CIS Kaggle dataset, served via FastAPI,
> tracked with MLflow, versioned with DVC, and monitored end-to-end with Prometheus + Grafana.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Pipeline](#ml-pipeline)
- [Project Structure](#project-structure)
- [Stack](#stack)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [DVC Data & Model Versioning](#dvc-data--model-versioning)
- [Monitoring](#monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Running Tests](#running-tests)
- [Model Performance](#model-performance)
- [Environment Variables](#environment-variables)

---

## Overview

This project tackles real-world fraud detection challenges:

- **Class imbalance** — the IEEE-CIS dataset is ~3.5% fraud. We use SMOTE oversampling + ensemble stacking to handle this without throwing away legitimate transactions.
- **Low-latency inference** — sub-50ms prediction via a lean FastAPI + joblib model loading setup.
- **Full reproducibility** — every dataset version, feature set, and model artifact is tracked in DVC backed by S3. You can roll back to any prior model in minutes.
- **Drift-aware retraining** — Evidently monitors incoming transaction distributions. When drift exceeds threshold, GitHub Actions fires a full retraining run automatically.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │           GitHub Actions             │
                         │  (CI on push + weekly cron + drift)  │
                         └────────────────┬────────────────────┘
                                          │ dvc repro
                    ┌─────────────────────▼──────────────────────┐
                    │              DVC Pipeline                   │
                    │  preprocess → features → train → evaluate  │
                    └──────┬────────────────────────┬────────────┘
                           │                        │
                    ┌──────▼──────┐         ┌───────▼──────┐
                    │  AWS S3     │         │   MLflow     │
                    │ (data +     │         │  (metrics +  │
                    │  models)    │         │   artifacts) │
                    └─────────────┘         └──────────────┘

  Client ──POST /predict──► FastAPI ──► XGB + LGB ensemble ──► fraud score
                               │
                        ┌──────▼──────┐
                        │ Prometheus  │◄── /metrics scrape
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Grafana   │  (dashboards + alerts)
                        └─────────────┘
```

---

## ML Pipeline

```
IEEE-CIS Raw Data
      │
      ▼
 preprocess.py          merge txn + identity, encode categoricals,
                        impute nulls, drop high-null columns
      │
      ▼
 feature_engineering.py time features, log-amount, email freq encoding,
                        velocity features
      │
      ▼
 hyperparameter_tuning  Optuna (50 trials) optimizing average_precision
      │
      ▼
 train.py               SMOTE on train split only → XGBoost + LightGBM
                        → ensemble average → MLflow tracking
      │
      ▼
 models/                xgb_model.pkl + lgb_model.pkl (DVC-tracked)
```

**Why ensemble?** XGBoost handles dense tabular patterns well; LightGBM is faster on high-cardinality categoricals. Averaging their probabilities consistently beats either alone on PR-AUC for this dataset.

**Why SMOTE on train split only?** Applying SMOTE before splitting leaks synthetic minority samples into validation, inflating metrics. We fit SMOTE only on `X_train` after the split.

---

## Project Structure

```
fraud-detection-mlops/
├── src/
│   ├── data/
│   │   ├── preprocess.py           raw → cleaned DataFrame
│   │   └── feature_engineering.py  cleaned → feature matrix
│   ├── training/
│   │   ├── train.py                SMOTE + ensemble + MLflow logging
│   │   └── hyperparameter_tuning.py Optuna study for XGB + LGB
│   ├── api/
│   │   ├── main.py                 FastAPI app, Prometheus instrumentation
│   │   ├── predict.py              inference logic
│   │   └── schemas.py              Pydantic request/response models
│   └── monitoring/
│       └── drift_detector.py       Evidently drift reports
├── data/
│   ├── raw/                        DVC-tracked (IEEE-CIS CSVs)
│   ├── processed/                  DVC-tracked
│   └── features/                   DVC-tracked
├── models/                         DVC-tracked (.pkl artifacts)
├── tests/
│   └── test_api.py                 pytest: responses, schema, latency
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml          API + MLflow + Prometheus + Grafana
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json
├── .github/
│   └── workflows/
│       └── retrain.yml             CI/CD + auto-retrain on drift
├── dvc.yaml                        pipeline stage definitions
├── params.yaml                     XGB/LGB hyperparameter defaults
├── .env.example
└── requirements.txt
```

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| ML models | XGBoost 2.0, LightGBM 4.3 | Best-in-class on tabular fraud data |
| Imbalance handling | imbalanced-learn (SMOTE) | Oversamples minority without data loss |
| Hyperparameter tuning | Optuna 3.6 | TPE sampler, pruning, MLflow integration |
| Experiment tracking | MLflow 2.12 | Params, metrics, artifact registry |
| Data versioning | DVC 3.49 + AWS S3 | Full pipeline reproducibility + rollback |
| API | FastAPI 0.111 + Uvicorn | Async, fast, auto OpenAPI docs |
| Drift detection | Evidently 0.4 | Statistical drift reports on live traffic |
| Metrics | Prometheus + Grafana | Real-time latency and prediction dashboards |
| CI/CD | GitHub Actions | Retrains on drift or weekly schedule |
| Infra | Docker + docker-compose | One-command local + prod parity |

---

## Prerequisites

- Python 3.10+
- Docker + docker-compose
- AWS account with an S3 bucket (for DVC remote)
- Kaggle account (to download IEEE-CIS dataset)
- Git

---

## Local Setup

**1. Clone and create virtualenv**

```bash
git clone https://github.com/Codex610/Real-Time-Fraud-Detection-API-with-Full-MLOps-Pipeline.git
cd Real-Time-Fraud-Detection-API-with-Full-MLOps-Pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# edit .env — add your AWS keys, S3 bucket, MLflow URI
```

**3. Download IEEE-CIS data**

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

**4. Initialize DVC and push data**

```bash
dvc init
dvc remote add -d myremote s3://your-bucket/fraud-detection
dvc add data/raw/train_transaction.csv data/raw/train_identity.csv
dvc push
```

**5. Run the full pipeline**

```bash
dvc repro          # preprocess → features → tune → train
```

**6. Start the API**

```bash
uvicorn src.api.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

---

## Docker Setup

Spins up 4 containers: API, MLflow server, Prometheus, Grafana.

```bash
cp .env.example .env   # fill in values first

docker-compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| Fraud Detection API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

To retrain and hot-reload the model without downtime:

```bash
dvc repro
docker-compose restart api
```

---

## API Reference

### `POST /predict`

Run fraud inference on a single transaction.

**Request**

```json
{
  "transaction_amt": 117.50,
  "product_cd": "W",
  "card_type": "credit",
  "p_emaildomain": "gmail.com",
  "features": [0.12, 1.0, 23.5, 0.0, 150.0]
}
```

**Response**

```json
{
  "fraud_probability": 0.0431,
  "prediction": "legit",
  "model_version": "v1.3.0",
  "latency_ms": 11.4
}
```

**Status codes**

| Code | Meaning |
|---|---|
| 200 | Prediction returned |
| 422 | Validation error — check request schema |
| 503 | Model not loaded |

---

### `GET /health`

Liveness check. Returns model load status.

```json
{
  "status": "ok",
  "models_loaded": true,
  "uptime_seconds": 3842
}
```

---

### `GET /metrics`

Prometheus scrape endpoint. Exposes:

- `predictions_total{result="fraud|legit"}` — counter
- `prediction_latency_seconds` — histogram (p50, p95, p99)
- `model_load_status` — gauge (1 = loaded, 0 = failed)

---

## MLflow Experiment Tracking

Every training run logs:

- **Params** — all XGB and LGB hyperparameters, SMOTE `k_neighbors`, threshold
- **Metrics** — precision, recall, F1, ROC-AUC, PR-AUC on validation set
- **Artifacts** — both model `.pkl` files, feature importance plots, confusion matrix

```bash
mlflow ui --port 5000
# open http://localhost:5000
```

To load the best registered model:

```python
import mlflow
model = mlflow.sklearn.load_model("models:/fraud-detector/Production")
```

---

## DVC Data & Model Versioning

The DVC pipeline has 3 stages defined in `dvc.yaml`:

```
preprocess → feature_engineering → train
```

**Useful commands**

```bash
dvc repro                  # run only changed stages
dvc repro --force          # force full rerun
dvc push                   # push data + models to S3
dvc pull                   # pull latest from S3

# roll back to a prior model version
git checkout v1.2.0
dvc checkout               # swaps in the models from that commit
```

---

## Monitoring

### Prometheus + Grafana

Grafana dashboard (imported from `monitoring/grafana/dashboard.json`) shows:

- Predictions per second (fraud vs legit split)
- p95 / p99 inference latency
- Fraud rate over time
- Model load status

### Drift Detection

`src/monitoring/drift_detector.py` uses Evidently to compare a reference window
(training distribution) against recent live traffic. It outputs a drift score per feature
and a boolean `dataset_drift` flag.

The GitHub Actions workflow checks drift on a schedule. If `dataset_drift=True`,
it triggers a full `dvc repro` retraining run.

---

## CI/CD Pipeline

`.github/workflows/retrain.yml` runs on:

- Every push to `main` (test + lint only)
- Weekly cron (`0 2 * * 1`) — drift check + conditional retrain
- Manual `workflow_dispatch` — force retrain

**Pipeline steps:**

```
lint + test → dvc pull → drift check → [retrain if drift] → dvc push → docker build → deploy
```

Secrets required in GitHub repo settings:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
DOCKER_REGISTRY_TOKEN
SERVER_IP
SERVER_USER
SERVER_SSH_KEY
```

---

## Running Tests

```bash
pytest tests/ -v

# specific test
pytest tests/test_api.py::test_latency -v

# with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Test suite covers:

- `/health` and `/predict` response codes and schema
- Fraud probability in `[0, 1]` range
- Prediction label is `fraud` or `legit`
- End-to-end inference latency under 50ms

---

## Model Performance

Evaluated on the held-out IEEE-CIS test set (20% stratified split):

| Metric | Score |
|---|---|
| Precision | **0.87** |
| Recall | 0.74 |
| F1 | 0.80 |
| ROC-AUC | 0.93 |
| PR-AUC | 0.81 |

> Precision is prioritized over recall here — false positives (blocking legit transactions)
> are more damaging to user trust than false negatives in this deployment context.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Yes | AWS credentials for DVC S3 remote |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS credentials for DVC S3 remote |
| `AWS_DEFAULT_REGION` | Yes | S3 bucket region |
| `DVC_REMOTE_BUCKET` | Yes | Full S3 path e.g. `s3://bucket/path` |
| `MLFLOW_TRACKING_URI` | Yes | MLflow server URI |
| `MLFLOW_EXPERIMENT` | No | Experiment name (default: `fraud-detection`) |
| `MODEL_THRESHOLD` | No | Decision threshold (default: `0.5`) |

---

## License

MIT
