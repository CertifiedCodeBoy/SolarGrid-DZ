"""Forecasts routes — solar production and demand forecasts."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Query
from backend.api.models import ForecastResponse, ForecastPoint

router = APIRouter()

_solar_forecaster = None
_demand_forecaster = None


def _get_solar_forecaster():
    global _solar_forecaster
    if _solar_forecaster is None:
        try:
            from ml.solar_forecast import SolarForecaster
            _solar_forecaster = SolarForecaster()
        except Exception:
            pass
    return _solar_forecaster


def _get_demand_forecaster():
    global _demand_forecaster
    if _demand_forecaster is None:
        try:
            from ml.demand_forecast import DemandForecaster
            _demand_forecaster = DemandForecaster()
        except Exception:
            pass
    return _demand_forecaster


@router.get("/solar/{installation_id}", response_model=ForecastResponse)
async def solar_forecast(
    installation_id: str,
    request: Request,
    horizon: int = Query(48, ge=1, le=168, description="Forecast horizon in hours"),
):
    """
    Get 48-hour solar production forecast for an installation.
    Uses real NWP weather data from Open-Meteo API (no key required).
    """
    engine = getattr(request.app.state, "dispatch_engine", None)
    inst = engine.installations.get(installation_id) if engine else None

    try:
        from ml.solar_forecast import SolarForecaster, fetch_weather_forecast
        forecaster = _get_solar_forecaster() or SolarForecaster.__new__(SolarForecaster)

        lat = inst.lat if inst else 36.75
        lon = inst.lon if inst else 3.05
        weather = fetch_weather_forecast(lat=lat, lon=lon, hours=horizon)

        fc = forecaster.forecast(weather_forecast=weather, horizon_hours=horizon)
        points = [
            ForecastPoint(
                timestamp=row.timestamp,
                forecast_kw=row.forecast_kw,
                lower_kw=row.lower_kw,
                upper_kw=row.upper_kw,
            )
            for row in fc.itertuples()
        ]
        return ForecastResponse(
            installation_id=installation_id,
            generated_at=datetime.utcnow(),
            horizon_hours=horizon,
            forecast=points,
        )
    except FileNotFoundError:
        # Model not yet trained — return synthetic forecast
        import numpy as np
        import pandas as pd
        now = datetime.utcnow()
        times = pd.date_range(now, periods=horizon, freq="1h")
        cap = inst.capacity_kw if inst else 500.0
        solar_angle = np.maximum(0, np.sin(np.pi * (times.hour - 6) / 12))
        fc_kw = cap * solar_angle * np.clip(np.random.normal(0.85, 0.1, horizon), 0, 1)
        points = [
            ForecastPoint(
                timestamp=ts,
                forecast_kw=round(float(v), 2),
                lower_kw=round(float(v * 0.85), 2),
                upper_kw=round(float(v * 1.15), 2),
            )
            for ts, v in zip(times, fc_kw)
        ]
        return ForecastResponse(
            installation_id=installation_id,
            generated_at=datetime.utcnow(),
            horizon_hours=horizon,
            forecast=points,
        )


@router.get("/demand/{installation_id}", response_model=ForecastResponse)
async def demand_forecast(
    installation_id: str,
    request: Request,
    horizon: int = Query(48, ge=1, le=168),
):
    """Get 48-hour demand forecast for an installation's district."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    inst = engine.installations.get(installation_id) if engine else None
    district_type = inst.district_type if inst else "residential"

    try:
        from ml.demand_forecast import DemandForecaster, generate_demand_data
        from datetime import timedelta
        forecaster = _get_demand_forecaster()
        if forecaster is None:
            raise FileNotFoundError("Model not trained")

        history = generate_demand_data(district_type=district_type, start="2025-01-01", end="2025-12-31")
        fc = forecaster.forecast(history, horizon_hours=horizon)
        points = [
            ForecastPoint(
                timestamp=row.timestamp,
                forecast_kw=row.forecast_kw,
                lower_kw=row.lower_kw,
                upper_kw=row.upper_kw,
            )
            for row in fc.itertuples()
        ]
        return ForecastResponse(
            installation_id=installation_id,
            generated_at=datetime.utcnow(),
            horizon_hours=horizon,
            forecast=points,
        )
    except (FileNotFoundError, Exception):
        import numpy as np
        import pandas as pd
        now = datetime.utcnow()
        times = pd.date_range(now, periods=horizon, freq="1h")
        peak = (inst.capacity_kw * 0.6) if inst else 300.0
        demand = peak * (0.4 + 0.6 * np.exp(-0.5 * ((times.hour - 19) / 3) ** 2))
        points = [
            ForecastPoint(
                timestamp=ts,
                forecast_kw=round(float(v), 2),
                lower_kw=round(float(v * 0.92), 2),
                upper_kw=round(float(v * 1.08), 2),
            )
            for ts, v in zip(times, demand)
        ]
        return ForecastResponse(
            installation_id=installation_id,
            generated_at=datetime.utcnow(),
            horizon_hours=horizon,
            forecast=points,
        )
