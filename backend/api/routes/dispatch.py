"""Dispatch routes — MPC battery dispatch and district balancing."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Query
from backend.api.models import DispatchAction, DispatchSchedule
import numpy as np

router = APIRouter()


@router.post("/{installation_id}/action", response_model=DispatchAction)
async def run_dispatch(installation_id: str, request: Request):
    """
    Execute one MPC dispatch step for a single installation.
    Returns the immediate charge/discharge action.
    """
    engine = getattr(request.app.state, "dispatch_engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Dispatch engine not available")
    if installation_id not in engine.installations:
        raise HTTPException(status_code=404, detail=f"{installation_id} not found")

    record = engine.dispatch_installation(installation_id)
    return DispatchAction(
        installation_id=installation_id,
        action=record["action"],
        p_charge_kw=record["p_charge_kw"],
        p_discharge_kw=record["p_discharge_kw"],
        p_grid_import_kw=record["p_grid_import_kw"],
        p_grid_export_kw=record["p_grid_export_kw"],
        soc_after=record["soc_after"],
        timestamp=datetime.utcnow(),
    )


@router.get("/{installation_id}/schedule", response_model=DispatchSchedule)
async def get_dispatch_schedule(
    installation_id: str,
    request: Request,
    horizon: int = Query(24, ge=1, le=48),
):
    """
    Get the full MPC optimised dispatch schedule for the next N hours.
    """
    engine = getattr(request.app.state, "dispatch_engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Dispatch engine not available")
    if installation_id not in engine.installations:
        raise HTTPException(status_code=404, detail=f"{installation_id} not found")

    inst = engine.installations[installation_id]
    state = engine.states[installation_id]
    controller = engine.controllers[installation_id]

    solar_fc = engine._get_solar_forecast(inst)
    demand_fc = engine._get_demand_forecast(inst)
    result = controller.optimise(solar_fc, demand_fc, current_soc=state.current_soc)

    schedule_records = result["schedule"].head(horizon).to_dict(orient="records")
    return DispatchSchedule(
        installation_id=installation_id,
        n_hours=horizon,
        schedule=schedule_records,
        summary=result["summary"],
    )


@router.post("/all/dispatch")
async def dispatch_all(request: Request):
    """Dispatch all installations and compute cross-district energy transfers."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Dispatch engine not available")
    results = engine.dispatch_all()
    return results


@router.get("/history")
async def dispatch_history(
    request: Request,
    installation_id: str = Query(None),
    last_n: int = Query(100, ge=1, le=1000),
):
    """Get recent dispatch history."""
    engine = getattr(request.app.state, "dispatch_engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Dispatch engine not available")

    history = engine.history[-last_n:]
    if installation_id:
        history = [h for h in history if h.get("installation_id") == installation_id]

    return {"count": len(history), "history": history}
