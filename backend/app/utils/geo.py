from typing import Optional

from geoalchemy2.elements import WKTElement


def point_to_wkt_element(longitude: float, latitude: float) -> WKTElement:
    return WKTElement(f"POINT({longitude} {latitude})", srid=4326)


def geom_from_lat_lon(
    latitude: Optional[float], longitude: Optional[float]
) -> Optional[WKTElement]:
    """Build a PostGIS POINT geometry from a valid lat/lon pair.

    Returns ``None`` when either coordinate is missing or out of range, so a
    detection with no GPS never gets a fabricated geometry.
    """
    if latitude is None or longitude is None:
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return point_to_wkt_element(longitude, latitude)


def calculate_bbox_area(width: float, height: float) -> float:
    return width * height


def severity_color(severity: str) -> str:
    colors = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#22c55e",
    }
    return colors.get(severity, "#6b7280")

