# SonicSweep — Marine Debris & Anomaly Detection

AI-powered underwater marine debris and anomaly detection using **Side-Scan
Sonar (SSS)** imagery. Upload sonar scans, run a YOLO-based detection pipeline
(preprocess -> inference -> anomaly classification -> risk scoring), and
review results on a web dashboard with a Leaflet geospatial map.

> **Honesty contract.** This project never fabricates AI results, GPS
> coordinates, accuracy, or benchmark numbers. If no trained model is present
> or coordinates cannot be read from real metadata, the API surfaces an error
> or *"Location unavailable"* rather than inventing data.

## Project Structure

```
marine-debris-detection/
├── frontend/               # Next.js React application (port 3000)
│   ├── app/                #   App Router pages (dashboard, scans, upload, analyze)
│   ├── components/         #   UI components (detection, map, layout, dashboard)
│   └── lib/                #   API client, types, utilities
├── backend/                # FastAPI application (port 8000)
│   ├── app/                #   Core application code
│   │   ├── api/v1/         #     REST endpoints (health, scans, detections)
│   │   ├── ai/             #     AI pipeline (detector, inference, preprocessing)
│   │   ├── core/           #     Config, database, settings
│   │   ├── models/         #     SQLAlchemy ORM models
│   │   ├── schemas/        #     Pydantic request/response schemas
│   │   ├── services/       #     Business logic
│   │   ├── training/       #     Training pipeline (split, train, evaluate, report)
│   │   └── utils/          #     Geotagging, mask rendering, geo helpers
│   ├── models/             #   Model weight files (YOLO .pt, ONNX, TensorRT)
│   ├── tests/              #   Pytest test suite
│   ├── train.py            #   CLI entry point for training
│   ├── requirements.txt    #   Python dependencies
│   └── requirements-edge.txt  # Optional ONNX/TensorRT deps
├── ai/                     # AI inference & preprocessing documentation
├── models/                 # Trained model weights (.pt, ONNX, TensorRT)
├── dataset/                # Real sonar images, YOLO labels, metadata
├── training/               # Training/evaluation scripts documentation
├── reports/                # Generated training/evaluation reports
├── docker-compose.yml      # PostGIS + backend + frontend orchestration
├── .env.example            # Root environment configuration template
├── setup.bat               # One-time project setup (venv, installs, .env)
├── run_backend.bat         # Start the FastAPI backend
├── run_frontend.bat        # Start the React frontend
├── run_all.bat             # Start backend + frontend together
└── README.md               # This file
```

## Quick Start (Windows)

### 1. Install Dependencies

```cmd
REM One-time setup — creates venv, installs all packages, copies .env
setup.bat
```

Or manually:

```cmd
REM Create Python virtual environment
python -m venv .venv

REM Activate the virtual environment
.venv\Scripts\activate

REM Install backend Python packages
pip install -r backend\requirements.txt

REM (Optional) Install edge deployment packages (ONNX/TensorRT)
pip install -r backend\requirements-edge.txt

REM Install frontend Node.js packages
cd frontend
npm install
cd ..
```

### 2. Create/Activate Python Virtual Environment

```cmd
REM Create (one-time only)
python -m venv .venv

REM Activate (run in every new terminal session)
.venv\Scripts\activate

REM Deactivate when done
deactivate
```

### 3. Start FastAPI Backend

```cmd
REM Option A: Use the convenience script
run_backend.bat

REM Option B: Manual start
.venv\Scripts\activate
cd backend
python main.py
```

The backend starts on **http://localhost:8000** with hot-reload.
API docs available at **http://localhost:8000/docs**.

### 4. Start React Frontend

```cmd
REM Option A: Use the convenience script
run_frontend.bat

REM Option B: Manual start
cd frontend
npm run dev
```

The frontend starts on **http://localhost:3000**.

### 5. Start Both Services

```cmd
REM Option A: Use the convenience script (opens two windows)
run_all.bat

REM Option B: Docker Compose (includes PostgreSQL + PostGIS)
docker compose up --build
```

### 6. Train the Model Using the Real Dataset

```cmd
REM Place your real sonar images and YOLO labels first:
REM   dataset/images/  -> .png, .jpg files
REM   dataset/labels/  -> matching .txt files (YOLO format)
REM
REM Then train:

REM Activate the virtual environment
.venv\Scripts\activate

REM Enter the backend directory
cd backend

REM Run the full training pipeline with defaults
python train.py

REM OR with custom hyperparameters
python train.py --epochs 50 --batch 8 --device cpu --imgsz 640

REM OR dry run (1 epoch on real data for a quick smoke test)
python train.py --dry-run --device cpu

REM OR point to the top-level dataset directory
python train.py --images-dir ../dataset/images --labels-dir ../dataset/labels
```

The pipeline runs all 9 stages on real data:
1. Validate dataset (images + labels)
2. Stratified train/val/test split (70/15/15)
3. Sonar preprocessing (grayscale, denoise, CLAHE)
4. YOLOv8 training (saves best.pt)
5. Validation metrics (P/R/mAP50/mAP50-95)
6. Test metrics + inference timing
7. Detection on test split
8. Anomaly classification + risk scoring
9. JSON/CSV report generation

Trained weights are saved to `backend/runs/detect/*/weights/best.pt`.
Copy to the root `models/marine_debris_yolov8.pt` for the backend to use:

```cmd
copy backend\runs\detect\marine_debris_yolov8\weights\best.pt models\marine_debris_yolov8.pt
```

### 7. Run AI Inference

**Via the web UI:**
1. Open http://localhost:3000/scans
2. Upload a sonar scan
3. Click "Run Detection" to trigger the AI pipeline

**Via the API:**

```cmd
REM Upload a scan
curl -X POST http://localhost:8000/api/v1/scans/upload -F "file=@path/to/sonar.png"

REM Run detection on the uploaded scan (replace <scan_id> with the returned ID)
curl -X POST http://localhost:8000/api/v1/detections/run/<scan_id>

REM Stateless analysis (no DB storage)
curl -X POST http://localhost:8000/api/v1/detections/analyze -F "file=@path/to/sonar.png"
```

**Export to ONNX/TensorRT for edge deployment:**

```cmd
cd backend

REM Export to ONNX
python -m app.ai.export --model ../models/marine_debris_yolov8.pt --format onnx

REM Export to TensorRT engine (requires CUDA GPU)
python -m app.ai.export --model ../models/marine_debris_yolov8.pt --format engine
```

Then set `INFERENCE_BACKEND=onnx` (or `tensorrt`) in your `.env` file.

**Benchmark on your hardware:**

```cmd
cd backend
python -m app.ai.benchmark ^
  --models ../models/marine_debris_yolov8.pt ^
  --images ../dataset/images/test ^
  --labels ../dataset/labels/test
```

### 8. Generate Reports

```cmd
REM Reports are auto-generated during training (step 6).
REM Find them in:
REM   backend\data\training_outputs\report.json
REM   backend\data\training_outputs\detections.csv
REM   backend\data\training_outputs\per_image_counts.csv

REM Copy to the reports/ directory for easy access:
copy backend\data\training_outputs\report.json reports\
copy backend\data\training_outputs\detections.csv reports\
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` in the project root (or `backend/.env`):

```cmd
copy .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://marine:marine@localhost:5432/marine_debris` | PostgreSQL connection |
| `YOLO_MODEL_PATH` | `models/marine_debris_yolov8.pt` | PyTorch checkpoint path |
| `INFERENCE_BACKEND` | `torch` | `torch` / `onnx` / `tensorrt` |
| `INFERENCE_DEVICE` | `None` (auto) | `cpu` or `cuda:0` |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.25` | YOLO confidence filter |
| `DETECTION_IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `SIMULATION_MODE` | `false` | Never enable in production |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000/api/v1` | Frontend API target |

### Database

PostgreSQL with PostGIS extension. Quick start with Docker:

```cmd
docker run -d --name marine-db -e POSTGRES_USER=marine -e POSTGRES_PASSWORD=marine -e POSTGRES_DB=marine_debris -p 5432:5432 postgis/postgis:16-3.4
```

Or use Docker Compose (includes DB + backend + frontend):

```cmd
docker compose up --build
```

## API Endpoints

Base path: `/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (includes DB connectivity) |
| GET | `/scans/` | List all scans |
| GET | `/scans/{id}` | Get a single scan |
| POST | `/scans/upload` | Upload a sonar image |
| DELETE | `/scans/{id}` | Delete a scan |
| GET | `/detections/` | List/filter detections |
| GET | `/detections/scan/{scan_id}` | Detections for a scan |
| POST | `/detections/run/{scan_id}` | Run AI pipeline on a scan |
| POST | `/detections/analyze` | Stateless image analysis |

## Dataset Format

Standard YOLO (Ultralytics) format:

```
dataset/
├── data.yaml          # class names: [ghost_net, shipwreck, pipe, ...]
├── images/            # sonar .png/.jpg files
│   ├── train/
│   ├── val/
│   └── test/
└── labels/            # matching .txt files, one box per line
    ├── train/
    ├── val/
    └── test/
```

Each label line: `<class_id> <x_center> <y_center> <width> <height>`
(normalized 0-1 fractions of image dimensions).

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind CSS + Leaflet |
| Backend | FastAPI + SQLAlchemy + Python 3.11 |
| Database | PostgreSQL 16 + PostGIS |
| AI | PyTorch + Ultralytics YOLOv8 |
| Edge AI | ONNX / TensorRT export + benchmarking |

## Testing

```cmd
.venv\Scripts\activate
cd backend
python -m pytest -q
```

## License

See the repository for licensing details.
