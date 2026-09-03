from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ScanBase(BaseModel):
    name: str
    description: str | None = None
    location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    depth: float | None = None
    scan_area_sqm: float | None = None


class ScanCreate(ScanBase):
    pass


class ScanRead(ScanBase):
    id: UUID
    filename: str
    file_path: str
    file_size: float | None = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ScanList(BaseModel):
    id: UUID
    name: str
    location_name: str | None = None
    status: str
    created_at: datetime
    detection_count: int = 0

    model_config = {"from_attributes": True}
