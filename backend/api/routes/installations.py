"""Installations CRUD routes."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from backend.api.models import InstallationCreate, InstallationResponse, SystemOverview, DistrictEnergyBalance
import numpy as np

router = APIRouter()

# ─── In-memory registry (replace with PostgreSQL in production) ───────────────
_installations: dict[str, dict] = {}


@router.get("/", response_model=list[InstallationResponse])
async def list_installations(request: Request):
    """List all registered solar installations."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    if engine:
        result = []
        for inst_id, inst in engine.installations.items():
            state = engine.states.get(inst_id)
            result.append(InstallationResponse(
                installation_id=inst.installation_id,
                district_id=inst.district_id,
                capacity_kw=inst.capacity_kw,
                battery_capacity_kwh=inst.battery_capacity_kwh,
                district_type=inst.district_type,
                lat=inst.lat,
                lon=inst.lon,
                current_soc=state.current_soc if state else 0.5,
                last_updated=state.updated_at if state else None,
            ))
        return result
    return list(_installations.values())


@router.get("/{installation_id}", response_model=InstallationResponse)
async def get_installation(installation_id: str, request: Request):
    """Get details for a single installation."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    if engine and installation_id in engine.installations:
        inst = engine.installations[installation_id]
        state = engine.states[installation_id]
        return InstallationResponse(
            installation_id=inst.installation_id,
            district_id=inst.district_id,
            capacity_kw=inst.capacity_kw,
            battery_capacity_kwh=inst.battery_capacity_kwh,
            district_type=inst.district_type,
            lat=inst.lat,
            lon=inst.lon,
            current_soc=state.current_soc,
            last_updated=state.updated_at,
        )
    raise HTTPException(status_code=404, detail=f"Installation {installation_id} not found")


@router.get("/system/overview", response_model=SystemOverview)
async def system_overview(request: Request):
    """Get real-time system-level energy overview."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Dispatch engine not available")

    total_solar = 0.0
    total_demand = 0.0
    total_import = 0.0
    total_export = 0.0
    total_soc = 0.0
    district_balances = []

    for inst_id, inst in engine.installations.items():
        solar = float(engine._synthetic_solar(inst.capacity_kw)[0])
        demand = float(engine._synthetic_demand(inst)[0])
        state = engine.states[inst_id]
        net = solar - demand
        total_solar += solar
        total_demand += demand
        total_soc += state.current_soc
        district_balances.append(DistrictEnergyBalance(
            district_id=inst.district_id,
            district_type=inst.district_type,
            solar_kw=round(solar, 2),
            demand_kw=round(demand, 2),
            battery_soc=round(state.current_soc, 4),
            net_kw=round(net, 2),
            transfers_in_kw=0.0,
            transfers_out_kw=0.0,
        ))

    n = len(engine.installations)
    self_cons = (total_solar - max(0, total_solar - total_demand)) / max(total_solar, 1e-6) * 100
    co2 = total_solar * 0.45  # kg

    return SystemOverview(
        timestamp=datetime.utcnow(),
        n_installations=n,
        total_solar_kw=round(total_solar, 2),
        total_demand_kw=round(total_demand, 2),
        total_battery_soc_avg=round(total_soc / n, 4),
        total_grid_import_kw=round(max(0, total_demand - total_solar), 2),
        total_grid_export_kw=round(max(0, total_solar - total_demand), 2),
        system_self_consumption_pct=round(self_cons, 2),
        co2_avoided_today_kg=round(co2, 2),
        district_balances=district_balances,
    )
