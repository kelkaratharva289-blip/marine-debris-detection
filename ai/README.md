# AI Inference & Preprocessing

This directory contains the AI inference pipeline, preprocessing modules,
anomaly classification, risk scoring, and edge deployment (ONNX/TensorRT)
utilities.

The actual implementation lives in `backend/app/ai/`. This directory serves
as the top-level reference and documentation for the AI subsystem.

## Components

| Module | Description |
|--------|-------------|
| `backend/app/ai/detector.py` | MarineDetector: full pipeline (preprocess -> infer -> anomaly -> risk) |
| `backend/app/ai/inference.py` | YOLO load/predict + marine class mapping |
| `backend/app/ai/preprocessing/` | Grayscale, denoise, CLAHE, normalize, letterbox |
| `backend/app/ai/anomaly.py` | Natural / Artificial / Uncertain classifier |
| `backend/app/ai/risk_engine.py` | 0-100 risk score + level buckets |
| `backend/app/ai/segmentation.py` | Optional mask pass (YOLO-Seg / U-Net) |
| `backend/app/ai/export.py` | PyTorch -> ONNX/TensorRT export |
| `backend/app/ai/benchmark.py` | Measured latency/FPS/accuracy CLI |
| `backend/app/ai/dataset/` | Dataset loading, validation, splitting |

## Running Inference

```bash
# From the backend directory, after uploading a scan via the API:
curl -X POST http://localhost:8000/api/v1/detections/run/<scan_id>

# Or trigger via the frontend UI at http://localhost:3000/scans
```

## Exporting Models

```bash
cd backend
python -m app.ai.export --model models/marine_debris_yolov8.pt --format onnx
python -m app.ai.export --model models/marine_debris_yolov8.pt --format engine
```

## Benchmarking

```bash
cd backend
python -m app.ai.benchmark \
  --models models/marine_debris_yolov8.pt \
  --images <dir/of/real/sonar/pngs> \
  --labels <dir/of/yolo/.txt/labels>
```
