from fastapi import APIRouter

from app.api.v1 import analysis, detections, health, scans

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(scans.router, prefix="/scans", tags=["Scans"])
api_router.include_router(detections.router, prefix="/detections", tags=["Detections"])
api_router.include_router(analysis.router, prefix="/detections", tags=["Detections"])
