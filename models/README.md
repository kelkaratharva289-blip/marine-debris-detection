# Trained Models

Place your trained model weights here. The backend expects one of:

- `marine_debris_yolov8.pt` — PyTorch checkpoint (default)
- `marine_debris_yolov8.onnx` — ONNX export (requires `INFERENCE_BACKEND=onnx`)
- `marine_debris_yolov8.engine` — TensorRT engine (requires `INFERENCE_BACKEND=tensorrt`)

These files are produced by the training pipeline or by the export CLI:

```bash
cd backend

# After training, copy best weights:
copy ..\runs\detect\marine_debris_yolov8\weights\best.pt ..\models\marine_debris_yolov8.pt

# Or export to ONNX:
python -m app.ai.export --model ..\models\marine_debris_yolov8.pt --format onnx
```

**This directory must not contain fabricated or placeholder model files.**
Only real trained weights belong here.

The backend references this directory via the `YOLO_MODEL_PATH` setting
(`../models/marine_debris_yolov8.pt` relative to the `backend/` working
directory). The legacy `backend/models/` directory may also contain model
files used directly by older configurations.
