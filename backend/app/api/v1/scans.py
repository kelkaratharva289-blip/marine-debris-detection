import os
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.detection import Detection
from app.models.scan import Scan
from app.schemas.scan import ScanCreate, ScanList, ScanRead
from app.utils.geo import geom_from_lat_lon
from app.utils.geotag import read_geotag

router = APIRouter()


@router.get("/", response_model=list[ScanList])
def list_scans(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    scans = (
        db.query(
            Scan,
            func.count(Detection.id).label("detection_count"),
        )
        .outerjoin(Detection)
        .group_by(Scan.id)
        .order_by(Scan.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    results = []
    for scan, det_count in scans:
        item = ScanList.model_validate(scan)
        item.detection_count = det_count
        results.append(item)
    return results


@router.get("/{scan_id}", response_model=ScanRead)
def get_scan(scan_id: UUID, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@router.get("/{scan_id}/image")
def get_scan_image(scan_id: UUID, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    if not os.path.exists(scan.file_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(scan.file_path)


@router.post("/", response_model=ScanRead, status_code=201)
def create_scan(scan_data: ScanCreate, db: Session = Depends(get_db)):
    scan = Scan(**scan_data.model_dump())
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


@router.post("/upload", response_model=ScanRead, status_code=201)
async def upload_scan(
    file: UploadFile = File(...),
    name: str = "",
    description: str = "",
    location_name: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    depth: float | None = None,
    db: Session = Depends(get_db),
):
    if file.size is not None and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_name = _sanitise_filename(file.filename)
    file_path = os.path.join(
        settings.UPLOAD_DIR,
        f"{uuid.uuid4().hex[:8]}_{safe_name}",
    )

    # Stream the upload to disk while enforcing the size limit, since
    # ``file.size`` can be None (e.g. chunked transfer) and must not be
    # trusted as the only guard.
    written = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > settings.MAX_UPLOAD_SIZE:
                f.close()
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    # If the caller did not supply coordinates, try to read real GPS from the
    # file's embedded / sidecar metadata. Never fabricate coordinates: if no
    # source yields valid lat/lon, the scan keeps None.
    scan_lat, scan_lon = latitude, longitude
    geo_timestamp = None
    if scan_lat is None or scan_lon is None:
        geotag = read_geotag(file_path=file_path)
        scan_lat, scan_lon = geotag.sanitize()
        geo_timestamp = geotag.timestamp if (scan_lat, scan_lon) != (None, None) else None

    scan = Scan(
        name=name or safe_name,
        description=description,
        location_name=location_name,
        latitude=scan_lat,
        longitude=scan_lon,
        depth=depth,
        geom=geom_from_lat_lon(scan_lat, scan_lon),
        filename=safe_name,
        file_path=file_path,
        file_size=float(written),
        status="uploaded",
        created_at=geo_timestamp,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def _sanitise_filename(filename: str | None) -> str:
    """Strip directory components and unsafe characters from a client-supplied name."""
    if not filename:
        return "scan"
    base = os.path.basename(filename.replace("\\", "/"))
    base = "".join(c for c in base if c not in '<>:"/\\|?*' and ord(c) >= 32)
    return base.strip() or "scan"


@router.delete("/{scan_id}", status_code=204)
def delete_scan(scan_id: UUID, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    if os.path.exists(scan.file_path):
        os.remove(scan.file_path)
    db.delete(scan)
    db.commit()
    return None
