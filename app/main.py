"""
app/main.py
===========
FastAPI application entrypoint.

Production-aware setup:
  - Structured JSON logging via structlog
  - Request tracing (trace_id injected into every log line)
  - Graceful degradation: DB errors return 503 with structured body
  - No raw stack traces in responses
  - CORS enabled for dashboard

Startup sequence:
  1. init_db() — create tables
  2. Mount routers
  3. Register exception handlers
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.anomalies import detect_anomalies
from app.db import get_db, init_db
from app.health import check_health
from app.ingestion import ingest_events
from app.metrics import compute_funnel, compute_heatmap, compute_metrics
from app.models import (
    AnomaliesResponse,
    FunnelResponse,
    HealthResponse,
    HeatmapResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
)

# ── Structured logging setup ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Store Intelligence API")
    await init_db()
    logger.info("Database initialised")
    yield
    logger.info("Shutting down Store Intelligence API")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    description="Offline retail analytics — Apex Retail",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request tracing middleware ────────────────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    trace_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # Extract store_id from path if present
    path_parts = request.url.path.split("/")
    store_id = None
    if "stores" in path_parts:
        idx = path_parts.index("stores")
        if idx + 1 < len(path_parts):
            store_id = path_parts[idx + 1]

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        store_id=store_id,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error("database_error", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "database_unavailable",
            "message": "The database is temporarily unavailable. Please retry.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", error=str(exc), exc_type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    summary="Ingest a batch of detection events",
    description=(
        "Accepts up to 500 events per call. Idempotent by event_id — safe to call twice "
        "with the same payload. Returns partial success on malformed events."
    ),
)
async def ingest(
    body: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    structlog.contextvars.bind_contextvars(
        endpoint="ingest",
        event_count=len(body.events),
    )

    if body.events:
        store_ids = {e.store_id for e in body.events}
        structlog.contextvars.bind_contextvars(store_ids=list(store_ids))

    result = await ingest_events(body.events, db)

    logger.info(
        "ingest_complete",
        accepted=result.accepted,
        rejected=result.rejected,
        duplicates=len(result.duplicate_ids),
    )
    return result


@app.get(
    "/stores/{store_id}/metrics",
    response_model=MetricsResponse,
    summary="Real-time store metrics",
)
async def get_metrics(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> MetricsResponse:
    structlog.contextvars.bind_contextvars(endpoint="metrics", store_id=store_id)
    return await compute_metrics(store_id, db)


@app.get(
    "/stores/{store_id}/funnel",
    response_model=FunnelResponse,
    summary="Visitor conversion funnel",
)
async def get_funnel(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> FunnelResponse:
    structlog.contextvars.bind_contextvars(endpoint="funnel", store_id=store_id)
    return await compute_funnel(store_id, db)


@app.get(
    "/stores/{store_id}/heatmap",
    response_model=HeatmapResponse,
    summary="Zone visit heatmap",
)
async def get_heatmap(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> HeatmapResponse:
    structlog.contextvars.bind_contextvars(endpoint="heatmap", store_id=store_id)
    return await compute_heatmap(store_id, db)


@app.get(
    "/stores/{store_id}/anomalies",
    response_model=AnomaliesResponse,
    summary="Active operational anomalies",
)
async def get_anomalies(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnomaliesResponse:
    structlog.contextvars.bind_contextvars(endpoint="anomalies", store_id=store_id)
    return await detect_anomalies(store_id, db)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and feed status",
)
async def get_health(
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    structlog.contextvars.bind_contextvars(endpoint="health")
    return await check_health(db)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
    }