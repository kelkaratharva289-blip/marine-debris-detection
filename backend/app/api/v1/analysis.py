"""Direct sonar image analysis endpoint.

Accepts a sonar image upload and runs the full trained pipeline — YOLO
detection, natural/artificial anomaly classification and risk scoring — and
returns per-object results *without* requiring the image to be stored as a
scan or written to the database.

This is stateless: the uploaded bytes are written to a temporary file, passed
through :class:`MarineDetector`, then removed. No fabricated values are
returned — if the model is unavailable or the image cannot be decoded, a
clear HTTP error is raised instead.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ai.detector import MarineDetector, ModelNotAvailableError
from app.ai.preprocessing.pipeline import PreprocessingError
from app.core.config import settings
from app.schemas.analysis import (
    AnalysisResponse,
    build_analysis_result,
)
from app.utils.geotag import read_geotag

logger = logging.getLogger(__name__)

router = APIRouter()

# Content types / extensions we will attempt to decode as sonar images.
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}

_detector: MarineDetector | None = None


def _get_detector() -> MarineDetector:
    """Return a lazily-initialised, cached :class:`MarineDetector`."""
    global _detector
    if _detector is None:
        _detector = MarineDetector()
    return _detector


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyse a sonar image",
    description=(
        "Runs YOLO detection + anomaly classification + risk scoring on an "
        "uploaded sonar image and returns per-object results."
    ),
)
async def analyze_image(
    file: UploadFile = File(...),
) -> AnalysisResponse:
    """Analyse an uploaded sonar image.

    Args:
        file: The sonar image to analyse.

    Returns:
        :class:`AnalysisResponse` with one :class:`DetectionAnalysisResult`
        per detected object, plus the model artifact used.

    Raises:
        HTTPException 413: The upload exceeds ``MAX_UPLOAD_SIZE``.
        HTTPException 415: The file type is not a supported image format.
        HTTPException 422: The file cannot be decoded as an image.
        HTTPException 503: The YOLO model weights are unavailable.
    """
    _validate_type(file)

    # Enforce the size limit while streaming to disk (file.size may be None
    # for chunked transfers, so it must not be the only guard).
    temp_path = await _write_to_temp(file)

    try:
        detector = _get_detector()
        detections = detector.detect(temp_path)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {exc}",
        ) from exc
    except PreprocessingError as exc:
        # Image could not be decoded / preprocessed -> invalid image.
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image: {exc}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        # Decode failure / unreadable file.
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected pipeline error
        logger.exception("Unexpected error analysing image")
        raise HTTPException(
            status_code=502,
            detail=f"Analysis pipeline failed: {exc}",
        ) from exc
    finally:
        _cleanup(temp_path)

    # Resolve the uploaded image's real GPS metadata. If no source (EXIF or
    # sidecar) yields coordinates, geotag.available stays False and results
    # carry lat/lon=None -> the UI shows "Location unavailable".
    geotag = read_geotag(file_path=temp_path)
    anchor_lat, anchor_lon = geotag.sanitize()
    geo_source = geotag.source if anchor_lat is not None else None
    geo_timestamp = (
        geotag.timestamp.isoformat() if geotag.timestamp else None
    )

    results = [
        build_analysis_result(
            d,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            geo_source=geo_source,
            geo_timestamp=geo_timestamp,
        )
        for d in detections
    ]
    return AnalysisResponse(
        detections=results,
        count=len(results),
        model=_model_label(),
        processed=True,
    )

def _validate_type(file: UploadFile) -> None:
    """Reject files that are clearly not images before touching disk."""
    filename = (file.filename or "").lower()
    ext = os.path.splitext(filename)[1]
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext or 'unknown'}'. Supported: "
                + ", ".join(sorted(s for s in SUPPORTED_EXTENSIONS if s))
            ),
        )
    media = (file.content_type or "").lower()
    if media and media not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{media}'.",
        )


async def _write_to_temp(file: UploadFile) -> str:
    """Stream the upload to a uniquely-named temp file, enforcing size limit."""
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    fd, temp_path = tempfile.mkstemp(
        prefix="sonar_analyze_", suffix=suffix, dir=_tmp_dir()
    )
    os.close(fd)

    written = 0
    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="File too large")
                f.write(chunk)
    except BaseException:
        _cleanup(temp_path)
        raise

    if written == 0:
        _cleanup(temp_path)
        raise HTTPException(status_code=422, detail="Empty file")

    return temp_path


def _tmp_dir() -> str:
    """Return a writable temp directory for analysis uploads (created if needed)."""
    base = settings.UPLOAD_DIR if settings.UPLOAD_DIR else tempfile.gettempdir()
    d = os.path.join(base, ".analyze_tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _cleanup(temp_path: str | None) -> None:
    """Safely remove a temporary analysis file."""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:  # noqa: BLE001 - best-effort cleanup
            logger.debug("Could not remove temp file %s", temp_path)


def _model_label() -> str:
    """Human-readable label for the model artifact that would be used."""
    from app.ai.inference import resolve_inference_model_path

    path = resolve_inference_model_path(
        settings.YOLO_MODEL_PATH,
        onnx_path=getattr(settings, "YOLO_ONNX_PATH", None),
        engine_path=getattr(settings, "YOLO_ENGINE_PATH", None),
        backend=getattr(settings, "INFERENCE_BACKEND", "torch"),
    )
    if not path:
        if settings.SIMULATION_MODE:
            return "simulation"
        return "unavailable"
    return str(path)
