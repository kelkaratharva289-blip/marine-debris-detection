# Training & Evaluation Scripts

This directory contains training pipeline scripts and documentation.
The actual implementation lives in `backend/app/training/`.

## Pipeline Stages

1. **Validation** — Check real images and YOLO labels for integrity
2. **Split** — Stratified train/val/test split (70/15/15 default)
3. **Preprocessing** — Grayscale, denoise, CLAHE enhancement
4. **YOLO Training** — Ultralytics YOLOv8 training with real data
5. **Validation Metrics** — P / R / mAP50 / mAP50-95 on val split
6. **Test Metrics** — Evaluation + inference timing on test split
7. **Detection** — Confidence-based detection on test split
8. **Anomaly + Risk** — Natural/Artificial classification + risk scoring
9. **Reports** — JSON/CSV output with real measured results

## Running Training

```bash
# From the project root:
cd backend
python train.py

# With options:
python train.py --device cpu --epochs 50 --batch 8

# Dry run (1 epoch, real data):
python train.py --dry-run --device cpu
```

## Dataset Location

The training pipeline expects real data in `backend/data/raw/` by default.
Configure alternative locations with `--images-dir` and `--labels-dir`:

```bash
python train.py --images-dir ../dataset/images --labels-dir ../dataset/labels
```

## Output

Training artifacts and reports are written to `backend/data/training_outputs/`
by default (configurable via `--output-dir`).

**Only real datasets and real training runs produce output here.
No fabricated metrics or placeholder results.**
