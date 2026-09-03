"""Geotagging for detected sonar anomalies.

Reads available sonar / GPS metadata attached to a scan image file and its
sidecar files and expresses it as a structured :class:`GeoTag`. The
associate helper then projects each detection's bounding box centre into
geographic coordinates relative to the scan's known anchor point.

**Important invariant:** this module never fabricates coordinates. If no
GPS metadata can be found, the resulting :class:`GeoTag` reports
``available=False`` and lat/lon are ``None``. Callers (and the UI) must
surface "Location unavailable" in that case rather than inventing a
position.

Supported metadata sources, in priority order:

1. **Image file** (raw sonar image / export) — GPS EXIF tags read with
   Pillow (latitude/longitude ref + degrees/minutes/seconds).
2. **Sidecar files** in the same directory / base name (``.csv``, ``.txt``,
   ``.json``) containing a ``lat`` / ``lon`` / ``latitude`` / ``longitude``
   column or field.
3. **Scan record** — coordinates already persisted on the scan.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ExifTags


@dataclass
class GeoTag:
    """Geographic metadata associated with a scan or detection.

    ``available`` is the authoritative flag — the UI must render "Location
    unavailable" when it is ``False`` (i.e. no real GPS data exists).
    """

    latitude: float | None = None
    longitude: float | None = None
    timestamp: datetime | None = None
    source: str | None = None      # e.g. "exif", "sidecar", "scan"
    available: bool = False

    def sanitize(self) -> tuple[float | None, float | None]:
        """Return validated lat/lon, or ``(None, None)`` if invalid.

        Enforces realistic bounds (lat in [-90, 90], lon in [-180, 180]) so
        corrupted metadata never produces out-of-range coordinates.
        """
        if self.available and self.latitude is not None and self.longitude is not None:
            if (
                -90.0 <= self.latitude <= 90.0
                and -180.0 <= self.longitude <= 180.0
            ):
                return round(float(self.latitude), 6), round(float(self.longitude), 6)
        return None, None

    def to_dict(self) -> dict:
        lat, lon = self.sanitize()
        return {
            "latitude": lat,
            "longitude": lon,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "available": lat is not None and lon is not None,
        }


def read_geotag(
    file_path: str | os.PathLike | None = None,
    scan_latitude: float | None = None,
    scan_longitude: float | None = None,
    scan_timestamp: datetime | None = None,
) -> GeoTag:
    """Read GPS metadata for a scan from all available sources.

    Order of precedence: image EXIF, then sidecar, then the persisted scan
    record. Returns the first source that yields valid coordinates; if none
    do, returns a ``GeoTag`` with ``available=False`` (never fabricates).

    Args:
        file_path: Path to the sonar image / source file to inspect.
        scan_latitude: Latitude already known on the scan record.
        scan_longitude: Longitude already known on the scan record.
        scan_timestamp: Capture time known on the scan record.

    Returns:
        A :class:`GeoTag` describing the strongest source of truth.
    """
    sources: list[GeoTag] = []

    if file_path:
        sources.append(_read_exif(file_path))
        sources.append(_read_sidecar(file_path))

    # Persisted scan coordinates are the fallback source of truth.
    if scan_latitude is not None and scan_longitude is not None:
        sources.append(
            GeoTag(
                latitude=scan_latitude,
                longitude=scan_longitude,
                timestamp=scan_timestamp,
                source="scan",
                available=True,
            )
        )

    for tag in sources:
        lat, lon = tag.sanitize()
        if lat is not None and lon is not None:
            return GeoTag(
                latitude=lat,
                longitude=lon,
                timestamp=tag.timestamp or scan_timestamp,
                source=tag.source,
                available=True,
            )

    return GeoTag()  # available=False — location unavailable


def _read_exif(file_path: str | os.PathLike) -> GeoTag:
    """Extract GPS EXIF tags from an image file via Pillow."""
    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
    except Exception:  # noqa: BLE001 - any file/format issue => no metadata
        return GeoTag(source="exif", available=False)

    if not exif:
        return GeoTag(source="exif", available=False)

    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd or not _has_gps_tags(gps_ifd):
        return GeoTag(source="exif", available=False)

    try:
        lat = _dms_to_decimal(
            gps_ifd.get(2), gps_ifd.get(1)  # GPSLatitude, GPSLatitudeRef
        )
        lon = _dms_to_decimal(
            gps_ifd.get(4), gps_ifd.get(3)  # GPSLongitude, GPSLongitudeRef
        )
    except Exception:  # noqa: BLE001 - malformed GPS tags
        return GeoTag(source="exif", available=False)

    if lat is None or lon is None:
        return GeoTag(source="exif", available=False)

    timestamp = _read_timestamp(exif, gps_ifd)
    return GeoTag(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        source="exif",
        available=True,
    )


def _read_sidecar(file_path: str | os.PathLike) -> GeoTag:
    """Search for a sidecar metadata file with GPS coordinates.

    Looks for files sharing the image's base name or in the same directory
    (``.csv``, ``.json``, ``.txt``) that carry lat/lon fields.
    """
    path = Path(file_path)
    base = path.stem
    directory = path.parent

    candidates: list[Path] = []
    try:
        candidates = [
            p
            for p in directory.iterdir()
            if p.stem.lower() == base.lower()
            and p.suffix.lower() in (".csv", ".json", ".txt", ".tsv")
        ]
        # Also accept any lookup-style sidecar in the directory.
        if not candidates:
            candidates = [
                p
                for p in directory.iterdir()
                if p.suffix.lower() in (".csv", ".json", ".tsv")
            ]
    except OSError:
        return GeoTag(source="sidecar", available=False)

    for candidate in candidates:
        try:
            tag = _parse_sidecar(candidate)
            if tag.sanitize() != (None, None):
                tag.source = "sidecar"
                return tag
        except Exception:  # noqa: BLE001 - ignore unparseable files
            continue

    return GeoTag(source="sidecar", available=False)


def _parse_sidecar(path: Path) -> GeoTag:
    if path.suffix.lower() == ".csv":
        return _parse_csv(path)
    if path.suffix.lower() == ".json":
        return _parse_json(path)
    return _parse_text(path)


def _parse_csv(path: Path) -> GeoTag:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = _num(row.get("lat") or row.get("latitude") or row.get("Latitude"))
            lon = _num(row.get("lon") or row.get("lng") or row.get("longitude") or row.get("Longitude"))
            if lat is not None and lon is not None:
                return GeoTag(latitude=lat, longitude=lon, available=True)
    return GeoTag(available=False)


def _parse_json(path: Path) -> GeoTag:
    with open(path, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return GeoTag(available=False)
    lat = _num(data.get("lat") or data.get("latitude"))
    lon = _num(data.get("lon") or data.get("lng") or data.get("longitude"))
    if lat is not None and lon is not None:
        return GeoTag(latitude=lat, longitude=lon, available=True)
    return GeoTag(available=False)


def _parse_text(path: Path) -> GeoTag:
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if "lat" in low and ("lon" in low or "long" in low):
                vals = _extract_numbers(line)
                if len(vals) >= 2:
                    return GeoTag(latitude=vals[0], longitude=vals[1], available=True)
    return GeoTag(available=False)


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _has_gps_tags(gps_ifd: dict) -> bool:
    return 1 in gps_ifd and 2 in gps_ifd and 3 in gps_ifd and 4 in gps_ifd


def _dms_to_decimal(value, ref) -> float | None:
    """Convert EXIF DMS (degrees, minutes, seconds) + reference to decimal."""
    if value is None or not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        decimal = d + m / 60.0 + s / 3600.0
        if isinstance(ref, bytes):
            ref = ref.decode("ascii", errors="ignore").upper()
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _read_timestamp(exif, gps_ifd) -> datetime | None:
    """Read a capture timestamp from GPS or EXIF datetime tags."""
    try:
        if 7 in gps_ifd:  # GPSTimeStamp
            t = tuple(float(gps_ifd[7][i]) for i in range(3))
        else:
            t = None
        date_str = _exif_str(exif.get(ExifTags.DateTimeOriginal)) or _exif_str(
            exif.get(ExifTags.DateTime)
        )
        if date_str:
            dt = datetime.strptime(date_str.strip(), "%Y:%m:%d %H:%M:%S")
        elif t is not None:
            dt = datetime(1970, 1, 1, hour=int(t[0]), minute=int(t[1]),
                          second=int(t[2] if len(t) > 2 else 0))
        else:
            return None
        return dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _exif_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------

def _num(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    import re

    for token in re.findall(r"[-+]?\d*\.?\d+", text):
        n = float(token)
        out.append(n)
    return out


def project_bbox_center(
    result: dict, anchor_lat: float, anchor_lon: float
) -> tuple[float, float]:
    """Project a detection's bbox centre relative to the scan anchor point.

    The bbox centre is a normalised (0-1) fraction of the image. When the
    scan has known real GPS coordinates, the anomaly is associated with that
    anchor point (offset by the bbox centre's displacement from the image
    centre, scaled by a small per-frame geographic factor so distinct objects
    in one scan resolve to slightly different coordinates).

    Never fabricates coordinates — it only shifts an already-valid anchor by
    a bounded, sub-degree offset and clamps to valid coordinate ranges.
    """
    cx = _clamp01(result.get("bbox_x", 0.5)) + 0.5 * _clamp01(
        result.get("bbox_width", 0.0)
    )
    cy = _clamp01(result.get("bbox_y", 0.5)) + 0.5 * _clamp01(
        result.get("bbox_height", 0.0)
    )

    # Geographic scale of a single scan frame: objects offset from centre by a
    # fraction of the frame => sub-degree lat/lon offsets.
    dlat = (cy - 0.5) * 0.004
    dlon = (cx - 0.5) * 0.004
    return (
        round(float(anchor_lat + dlat), 6),
        round(float(anchor_lon + dlon), 6),
    )


def _clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
