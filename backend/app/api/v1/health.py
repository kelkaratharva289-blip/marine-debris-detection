from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine

router = APIRouter()


@router.get("/health")
def health_check():
    """Report API health including real database connectivity.

    The API can boot without the database (schema creation is attempted on
    startup), so the health endpoint performs a live ``SELECT 1`` instead of
    reporting healthy unconditionally.
    """
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any connectivity error degrades status
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": "up" if db_ok else "down",
    }