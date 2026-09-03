# Generated Reports

This directory contains reports generated from real training runs, evaluations,
and inference results. Reports are produced by the training pipeline and
benchmark tools.

## Report Types

| File | Description |
|------|-------------|
| `report.json` | Full pipeline summary (config, split counts, metrics, detection summary) |
| `detections.csv` | Per-detection records with bbox, anomaly class, risk score |
| `per_image_counts.csv` | Detection count per test image |

## Generating Reports

```bash
# Full pipeline (produces reports in backend/data/training_outputs/):
cd backend
python train.py

# Copy reports to this directory for easy access:
copy data\training_outputs\report.json ..\reports\
copy data\training_outputs\detections.csv ..\reports\
```

## Benchmark Reports

```bash
python -m app.ai.benchmark \
  --models models/marine_debris_yolov8.pt \
  --images ../dataset/images/test \
  --labels ../dataset/labels/test
```

**This directory contains only results from actual model runs.
No synthetic, estimated, or placeholder data.**
