# Real Sonar Dataset

Place your real side-scan sonar images and YOLO-format labels here.

## Required Structure

```
dataset/
├── images/
│   ├── train/       # Training sonar images (.png, .jpg, .jpeg, .bmp)
│   ├── val/         # Validation sonar images
│   └── test/        # Test sonar images
├── labels/
│   ├── train/       # Per-image .txt label files (YOLO format)
│   ├── val/
│   └── test/
├── data.yaml        # YOLO dataset config with class names
└── metadata/        # Optional: EXIF sidecar JSONs for geotagging
```

## Label Format

Each `.txt` label file contains one detection per line in YOLO normalized format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`ghost_net` = class 0):

```
0 0.45 0.32 0.12 0.08
0 0.71 0.55 0.09 0.06
```

## data.yaml

```yaml
names:
  0: ghost_net
  1: shipwreck
  2: pipe
  3: cylinder
  4: container
  5: other_debris
```

## Quick Start

1. Place raw images in `dataset/images/` (any structure — the training
   pipeline will validate and split them automatically).
2. Place matching YOLO labels in `dataset/labels/` (same filenames, `.txt` extension).
3. Run `python backend/train.py` from the project root.

**This directory must contain only real sonar imagery and real annotations.
No synthetic, generated, or placeholder data is permitted.**
