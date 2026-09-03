import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1.router import api_router
from app.core.config import settings
import app.models  # noqa: F401 - register all tables
from app.core.database import Base, engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


def init_db() -> None:
    """Enable PostGIS and create tables on startup.

    Uses SQLAlchemy metadata so scans and detections (including the PostGIS
    geometry columns) are created if they do not already exist. No-op once
    the schema is present.

    The startup failure is logged (not swallowed) so operators can see why
    the backend started without a database connection.
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema ready")
    except Exception as exc:  # noqa: BLE001 - DB may be unreachable at boot
        logger.error("Database initialisation failed: %s", exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
