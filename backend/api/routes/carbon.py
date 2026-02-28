"""Carbon offset tracking routes."""

from datetime import datetime
from fastapi import APIRouter, Query
from backend.api.models import CarbonReport

router = APIRouter()

# Algeria national grid emission factor (kg CO₂/kWh)
GRID_CARBON_FACTOR = 0.65

# Equivalence constants
TREE_ABSORPTION_KG_YEAR = 21.0        # kg CO₂ absorbed per tree per year
CAR_EMISSIONS_KG_KM = 0.21           # kg CO₂ per km for average car
AVG_CAR_KM_YEAR = 15000.0


@router.get("/report", response_model=CarbonReport)
async def carbon_report(
    district_id: str = Query(None, description="Filter by district ID"),
    period_days: int = Query(30, ge=1, le=365),
    total_solar_kwh: float = Query(None, description="Override with real energy total"),
    self_consumption_pct: float = Query(83.0, ge=0, le=100),
    grid_import_reduction_pct: float = Query(41.0, ge=0, le=100),
):
    """
    Calculate carbon offset metrics for a district or the entire system.

    If `total_solar_kwh` is not provided, uses a simulated value based on
    period_days and the district's typical production.
    """
    if total_solar_kwh is None:
        # Simulate: 500 kW average system, ~4h equivalent full load per day
        base_kwh_per_day = 500 * 4.5
        total_solar_kwh = base_kwh_per_day * period_days * (self_consumption_pct / 100)

    # CO₂ avoided = solar used × grid carbon factor
    solar_used_kwh = total_solar_kwh * (self_consumption_pct / 100)
    co2_avoided_kg = solar_used_kwh * GRID_CARBON_FACTOR
    co2_avoided_tons = co2_avoided_kg / 1000.0

    # Equivalences
    equivalent_trees = co2_avoided_kg / (TREE_ABSORPTION_KG_YEAR / 365 * period_days)
    cars_off_road = co2_avoided_kg / (CAR_EMISSIONS_KG_KM * AVG_CAR_KM_YEAR / 365 * period_days)

    return CarbonReport(
        district_id=district_id,
        period_days=period_days,
        total_solar_kwh=round(total_solar_kwh, 2),
        co2_avoided_kg=round(co2_avoided_kg, 2),
        co2_avoided_tons=round(co2_avoided_tons, 4),
        equivalent_trees=round(equivalent_trees, 1),
        equivalent_cars_off_road=round(cars_off_road, 2),
        self_consumption_rate_pct=round(self_consumption_pct, 2),
        grid_import_reduction_pct=round(grid_import_reduction_pct, 2),
        generated_at=datetime.utcnow(),
    )


@router.get("/monthly")
async def monthly_carbon_breakdown(year: int = Query(2026)):
    """Monthly CO₂ avoidance breakdown — useful for city sustainability dashboard."""
    import numpy as np

    # Simulated monthly breakdown (kWh produced per month)
    monthly_irr = [3.8, 4.5, 5.6, 6.8, 7.5, 8.0, 8.2, 7.9, 6.7, 5.3, 4.0, 3.5]  # avg kWh/m²/day
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    results = []
    for i, (month, irr) in enumerate(zip(months, monthly_irr)):
        import calendar
        days = calendar.monthrange(year, i + 1)[1]
        solar_kwh = irr * days * 500 * 0.20 * 5  # simplified
        co2_kg = solar_kwh * 0.83 * GRID_CARBON_FACTOR
        results.append({
            "month": month,
            "year": year,
            "solar_kwh": round(solar_kwh, 0),
            "co2_avoided_kg": round(co2_kg, 0),
            "avg_irradiance_kwh_m2_day": irr,
        })
    return {"year": year, "monthly": results, "total_co2_avoided_kg": round(sum(r["co2_avoided_kg"] for r in results), 0)}


@router.get("/national-target")
async def national_target_progress():
    """Algeria 2030 renewable target progress tracker."""
    return {
        "target_year": 2030,
        "target_capacity_gw": 22.0,
        "current_capacity_gw": 0.56,  # as of 2026 estimate
        "progress_pct": round(0.56 / 22.0 * 100, 2),
        "solargrid_dz_contribution_mw": 5.0,  # simulated district deployments
        "years_remaining": 4,
        "annual_addition_needed_gw": round((22.0 - 0.56) / 4, 2),
    }
