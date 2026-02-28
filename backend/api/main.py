"""
SolarGrid DZ — FastAPI Backend
Main application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import forecasts, installations, dispatch, carbon, maintenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("solargrid.api")


# ─────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("SolarGrid DZ API starting up...")

    # Start background Kafka pipeline
    from backend.data_pipeline.influx_client import get_influx_client
    from backend.data_pipeline.kafka_consumer import start_kafka_to_influx_pipeline

    influx = get_influx_client()
    app.state.influx = influx
    app.state.kafka_consumer = start_kafka_to_influx_pipeline(influx)

    # Start MQTT listener
    from backend.data_pipeline.mqtt_listener import start_mqtt_pipeline
    app.state.mqtt_listener = start_mqtt_pipeline(influx)

    # Initialise dispatch engine
    from optimizer.dispatch_engine import create_default_engine
    app.state.dispatch_engine = create_default_engine(n_districts=5, simulation_mode=True)

    logger.info("All services started successfully")
    yield

    # Shutdown
    logger.info("SolarGrid DZ API shutting down...")
    if hasattr(app.state, "kafka_consumer"):
        app.state.kafka_consumer.stop()
    if hasattr(app.state, "mqtt_listener"):
        app.state.mqtt_listener.stop()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="SolarGrid DZ API",
    description="Smart Solar Energy Management System for Urban Districts — Algeria",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(installations.router, prefix="/api/v1/installations", tags=["Installations"])
app.include_router(forecasts.router, prefix="/api/v1/forecasts", tags=["Forecasts"])
app.include_router(dispatch.router, prefix="/api/v1/dispatch", tags=["Dispatch"])
app.include_router(carbon.router, prefix="/api/v1/carbon", tags=["Carbon"])
app.include_router(maintenance.router, prefix="/api/v1/maintenance", tags=["Maintenance"])


# ─────────────────────────────────────────────
# Health & Root
# ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {"service": "SolarGrid DZ API", "version": "1.0.0", "status": "running"}


@app.get("/health", tags=["Root"])
async def health():
    return JSONResponse({"status": "healthy", "service": "solargrid-api"})
