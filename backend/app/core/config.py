from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Marine Debris Detection API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql://marine:marine@localhost:5432/marine_debris"
    POSTGRES_USER: str = "marine"
    POSTGRES_PASSWORD: str = "marine"
    POSTGRES_DB: str = "marine_debris"

    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024

    YOLO_MODEL_PATH: str = "models/marine_debris_yolov8.pt"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.25
    DETECTION_IOU_THRESHOLD: float = 0.45
    # When the YOLO weights are unavailable and SIMULATION_MODE is enabled,
    # the detector returns realistic placeholder detections so the full
    # pipeline (preprocess -> infer -> anomaly -> risk -> geotag -> DB) can
    # be exercised end-to-end in a demo. DISABLED BY DEFAULT: placeholder
    # detections are NOT real AI output and must never be presented as such.
    SIMULATION_MODE: bool = False
    DETECTION_CLASSES: list[str] = [
        "ghost_net",
        "shipwreck",
        "pipe",
        "cylinder",
        "container",
        "other_debris",
    ]

    # Edge deployment: optional exported model backends (ONNX / TensorRT)
    # produced with `python -m app.ai.export`. When an artifact exists on
    # disk for the configured backend, the detector loads it in preference
    # to the raw PyTorch checkpoint. Verify real latency/accuracy gains with
    # `python -m app.ai.benchmark` — no speedup is assumed.
    INFERENCE_BACKEND: str = "torch"  # "torch" | "onnx" | "tensorrt"
    YOLO_ONNX_PATH: str = "models/marine_debris_yolov8.onnx"
    YOLO_ENGINE_PATH: str = "models/marine_debris_yolov8.engine"
    INFERENCE_DEVICE: str | None = None  # e.g. "cuda:0" (None = auto)

    # Optional segmentation (YOLO-Seg or U-Net). Disabled by default; enable
    # and point SEGMENTATION_MODEL_PATH at trained weights to get masks.
    SEGMENTATION_ENABLED: bool = False
    SEGMENTATION_MODEL_TYPE: str = "yolo-seg"  # "yolo-seg" | "unet"
    SEGMENTATION_MODEL_PATH: str = "models/marine_debris_yolov8-seg.pt"
    SEGMENTATION_CONFIDENCE_THRESHOLD: float = 0.25
    SEGMENTATION_IOU_THRESHOLD: float = 0.45
    # Directory where rendered mask-overlay PNGs are stored.
    MASK_OUTPUT_DIR: str = "./uploads/masks"

    # Anomaly classification: distinguishes Natural / Artificial / Uncertain
    # detected regions from hand-crafted sonar features + AI confidence.
    # Weights (sum to 1) control how much each feature group contributes.
    ANOMALY_ENABLED: bool = True
    ANOMALY_WEIGHT_SHAPE: float = 0.25
    ANOMALY_WEIGHT_TEXTURE: float = 0.20
    ANOMALY_WEIGHT_ACOUSTIC_SHADOW: float = 0.25
    ANOMALY_WEIGHT_AI: float = 0.30
    # Evidence margin below which a region is labelled "uncertain".
    ANOMALY_UNCERTAIN_THRESHOLD: float = 0.15

    # Shape feature parameters
    ANOMALY_ASPECT_RATIO_MIN: float = 1.0
    ANOMALY_ASPECT_RATIO_MAX: float = 4.0
    ANOMALY_COMPACTNESS_MIN: float = 0.05
    ANOMALY_COMPACTNESS_MAX: float = 0.8
    ANOMALY_CONCAVITY_MIN: float = -0.3
    ANOMALY_CONCAVITY_MAX: float = 0.4
    ANOMALY_SHAPE_GEOMETRY_WEIGHT: float = 0.5
    ANOMALY_SHAPE_CONVEXITY_WEIGHT: float = 0.35
    ANOMALY_SHAPE_ASPECT_WEIGHT: float = 0.15
    ANOMALY_SHAPE_ASPECT_DAMPEN: float = 0.5

    # Texture feature parameters
    ANOMALY_CANNY_LOW: int = 50
    ANOMALY_CANNY_HIGH: int = 150
    ANOMALY_TEXTURE_CONTRAST_MIN: float = 0.2
    ANOMALY_TEXTURE_CONTRAST_MAX: float = 2.5
    ANOMALY_TEXTURE_EDGE_MIN: float = 0.02
    ANOMALY_TEXTURE_EDGE_MAX: float = 0.4
    ANOMALY_TEXTURE_RELIABILITY_PIXELS: int = 900
    ANOMALY_TEXTURE_UNIFORM_CONTRAST_GATE: float = 0.35
    ANOMALY_TEXTURE_UNIFORM_SCORE_GATE: float = 0.7
    ANOMALY_TEXTURE_UNIFORM_DAMPEN: float = 0.4

    # Acoustic shadow feature parameters
    ANOMALY_SHADOW_SEARCH_MULTIPLIER: float = 2.5
    ANOMALY_SHADOW_SEARCH_MIN_PX: int = 4
    ANOMALY_SHADOW_FRACTION_MIN: float = 0.05
    ANOMALY_SHADOW_FRACTION_MAX: float = 0.6
    ANOMALY_SHADOW_DEPTH_MIN: float = 0.05
    ANOMALY_SHADOW_DEPTH_MAX: float = 0.9
    ANOMALY_SHADOW_STRIP_CONTRAST_MIN: float = 0.05
    ANOMALY_SHADOW_STRIP_CONTRAST_MAX: float = 0.9
    ANOMALY_SHADOW_FRACTION_WEIGHT: float = 0.4
    ANOMALY_SHADOW_DEPTH_WEIGHT: float = 0.3
    ANOMALY_SHADOW_CONTRAST_WEIGHT: float = 0.3
    ANOMALY_SHADOW_NEUTRAL_FRAC_GATE: float = 0.02
    ANOMALY_SHADOW_NEUTRAL_CONTRAST_GATE: float = 0.05

    # Marine Risk Scoring engine: computes a configurable 0-100 risk score
    # by fusing object type (per-class hazard prior), detector confidence,
    # estimated size and anomaly artificial probability into a risk level
    # (low | medium | high | critical). Weights are renormalised internally
    # over whatever inputs are available; being uncertain penalises the
    # artificial component so ambiguity never inflates risk.
    RISK_WEIGHT_OBJECT_TYPE: float = 0.25
    RISK_WEIGHT_CONFIDENCE: float = 0.25
    RISK_WEIGHT_SIZE: float = 0.25
    RISK_WEIGHT_ARTIFICIAL: float = 0.25
    RISK_UNCERTAIN_PENALTY: float = 0.30
    # Estimated-size mapping thresholds (bbox-area fraction of the image).
    # A bbox fraction at/below size_small maps to low risk, at/above
    # size_large maps to high risk, linear in between.
    RISK_SIZE_SMALL: float = 0.05
    RISK_SIZE_LARGE: float = 0.35
    # Risk level buckets (0-100).
    RISK_LOW_THRESHOLD: float = 49.0
    RISK_MEDIUM_THRESHOLD: float = 74.0
    RISK_HIGH_THRESHOLD: float = 89.0

    # Sonar preprocessing defaults
    PREPROCESS_IMG_SIZE: int = 640
    PREPROCESS_NORMALIZE_MEAN: float = 0.0
    PREPROCESS_NORMALIZE_STD: float = 255.0
    PREPROCESS_DENOISE_KERNEL: int = 5
    PREPROCESS_DENOISE_SIGMA_SPACE: float = 15.0
    PREPROCESS_DENOISE_SIGMA_COLOR: float = 45.0
    PREPROCESS_CLAHE_CLIP_LIMIT: float = 2.0
    PREPROCESS_CLAHE_GRID_SIZE: int = 8
    PREPROCESS_NORMALIZE_MIN: float = 0.0
    PREPROCESS_NORMALIZE_MAX: float = 1.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
