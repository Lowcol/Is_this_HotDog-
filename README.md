# Is This HotDog

Hotdog vs not-hotdog classification project with:

- training/evaluation pipeline
- MLflow model logging
- FastAPI inference service
- Docker support for pipeline and API

## 1. Prerequisite

Install and start Docker Desktop.

## 2. Train and Evaluate (Docker)

Build image:

```bash
docker build -t hotdog-pipeline .
```

Run the full pipeline in a container:

```bash
docker run --rm -v "%cd%\mlruns:/app/mlruns" -v "%cd%\dataset:/app/dataset" hotdog-pipeline
```

Pipeline steps:

1. Import data
2. Train model
3. Evaluate model
4. Check deployment thresholds
5. Log model to MLflow when thresholds are met

## 3. Run API (Docker Compose)

Start API service:

```bash
docker compose up --build api
```

Stop services:

```bash
docker compose down
```

API endpoints:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 4. Test Prediction Endpoint

Single image:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

Health check:

```bash
curl "http://localhost:8000/health"
```

API will be available at:

- http://localhost:8000

## 5. Project Notes

- Class imbalance can reduce recall/precision for `hotdog` if training data is skewed.
- More training data and augmentation generally improve generalization.
- Transfer learning (MobileNetV2) is preferred over training a CNN from scratch with limited data.
