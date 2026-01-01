# Sign Language Classifier

Flask API that classifies a hand gesture from an uploaded image using MediaPipe hand landmarks and an XGBoost model.

## Endpoints

- GET / (health check)
- POST /predict (multipart form-data with key "image")

Example:

```bash
curl -X POST \
  -F "image=@/path/to/hand.jpg" \
  http://localhost:3000/predict
```

## Run locally (no Docker)

```bash
pip install -r requirements.txt
python api.py
```

## Run locally (Docker)

```bash
docker build -t signlanguage-api .
docker run -p 3000:10000 -e PORT=10000 signlanguage-api
```

## Deploy to Render

This repo includes a Dockerfile and render.yaml.

1) Push to GitHub
2) Render: New + -> Web Service -> Connect repo
3) Environment: Docker
4) Deploy

Render sets $PORT automatically. The container starts Gunicorn binding to 0.0.0.0:$PORT.

## Model files

These files must be present (included in this repo):
- hand_xgb_model.pkl
- scaler.pkl
- class_to_idx.pkl
