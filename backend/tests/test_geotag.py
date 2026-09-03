"""Unit tests for the geotagging module."""

import os

import numpy as np
import pytest

try:
    from PIL import Image
    from PIL.ExifTags import IFD
    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False

from app.utils.geotag import (
    GeoTag,
    read_geotag,
    project_bbox_center,
    _dms_to_decimal,
)


def test_sanitize_valid():
    tag = GeoTag(latitude=40.7128, longitude=-74.0060, available=True)
    assert tag.sanitize() == (40.7128, -74.0060)


def test_sanitize_rejects_out_of_bounds():
    tag = GeoTag(latitude=120.0, longitude=0.0, available=True)
    assert tag.sanitize() == (None, None)


def test_unavailable_by_default():
    tag = GeoTag()
    assert tag.available is False
    assert tag.sanitize() == (None, None)
    d = tag.to_dict()
    assert d["available"] is False
    assert d["latitude"] is None
    assert d["longitude"] is None


def test_never_fabricates_when_no_metadata(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.fromarray(np.full((20, 20, 3), 100, np.uint8)).save(img_path)
    tag = read_geotag(file_path=str(img_path))
    assert tag.available is False
    assert tag.sanitize() == (None, None)


def test_scan_fallback_when_no_file_metadata(tmp_path):
    # No EXIF/sidecar metadata, but the scan record has real coordinates.
    tag = read_geotag(
        file_path=str(tmp_path / "missing.png"),
        scan_latitude=-33.8688,
        scan_longitude=151.2093,
    )
    assert tag.available is True
    assert tag.source == "scan"
    lat, lon = tag.sanitize()
    assert lat == -33.8688
    assert lon == 151.2093


@pytest.mark.skipif(not HAS_PIL, reason="Pillow not available")
def test_reads_exif_gps(tmp_path):
    img_path = tmp_path / "geo.jpg"
    img = Image.fromarray(np.full((10, 10, 3), 0, np.uint8))
    gps = img.getexif()
    gps_ifd = gps.get_ifd(IFD.GPSInfo)
    gps_ifd[1] = "N"
    gps_ifd[2] = (40, 42, 46.08)
    gps_ifd[3] = "W"
    gps_ifd[4] = (74, 0, 21.6)
    img.save(img_path, exif=gps.get_ifd(IFD.GPSInfo) and gps)
    tag = read_geotag(file_path=str(img_path))
    assert tag.available is True
    assert tag.source == "exif"
    lat, lon = tag.sanitize()
    assert lat == pytest.approx(40.7128, abs=1e-4)
    assert lon == pytest.approx(-74.006, abs=1e-4)


def test_reads_sidecar_csv(tmp_path):
    base = tmp_path / "scan001"
    img_path = tmp_path / "scan001.png"
    Image.fromarray(np.full((10, 10, 3), 0, np.uint8)).save(img_path)
    with open(tmp_path / "scan001.csv", "w") as f:
        f.write("latitude,longitude\n40.5,-70.25\n")
    tag = read_geotag(file_path=str(img_path))
    assert tag.available is True
    assert tag.source == "sidecar"
    lat, lon = tag.sanitize()
    assert lat == 40.5
    assert lon == -70.25


def test_reads_sidecar_json(tmp_path):
    img_path = tmp_path / "scan002.png"
    Image.fromarray(np.full((10, 10, 3), 0, np.uint8)).save(img_path)
    with open(tmp_path / "scan002.json", "w") as f:
        f.write('{"latitude": -22.5, "longitude": 44.1}')
    tag = read_geotag(file_path=str(img_path))
    assert tag.available is True
    lat, lon = tag.sanitize()
    assert lat == -22.5
    assert lon == 44.1


def test_sidecar_takes_precedence_over_none_file(tmp_path):
    # CSV sidecar provides coords even though image has no EXIF.
    img_path = tmp_path / "scan003.png"
    Image.fromarray(np.full((10, 10, 3), 0, np.uint8)).save(img_path)
    with open(tmp_path / "scan003.csv", "w") as f:
        f.write("lat,lng\n10.0,20.0\n")
    tag = read_geotag(file_path=str(img_path))
    assert tag.available is True
    assert tag.sanitize() == (10.0, 20.0)


def test_dms_to_decimal():
    assert _dms_to_decimal((10, 30, 0), "N") == pytest.approx(10.5)
    assert _dms_to_decimal((10, 30, 0), "S") == pytest.approx(-10.5)
    assert _dms_to_decimal((0, 0, 0), "E") == pytest.approx(0.0)
    assert _dms_to_decimal(None, "N") is None


def test_projection_stays_valid():
    # Object at image centre maps to the anchor point itself.
    lat, lon = project_bbox_center(
        {"bbox_x": 0.4, "bbox_y": 0.4, "bbox_width": 0.2, "bbox_height": 0.2},
        10.0,
        20.0,
    )
    assert lat == pytest.approx(10.0, abs=1e-3)
    assert lon == pytest.approx(20.0, abs=1e-3)

    # Even at the extremes the projection stays within valid range.
    lat, lon = project_bbox_center({"bbox_x": 1.0, "bbox_y": 1.0}, 85.0, 179.0)
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0
