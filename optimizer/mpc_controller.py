"""
SolarGrid DZ — MPC (Model Predictive Control) Battery Dispatch Controller
Optimises battery charge/discharge schedule to maximise solar self-consumption
and minimise grid import costs over a rolling 24-hour horizon.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cvxpy as cp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# System Configuration
# ─────────────────────────────────────────────

@dataclass
class BatteryConfig:
    """Battery energy storage system (BESS) configuration."""
    capacity_kwh: float = 500.0        # Total usable capacity
    max_charge_kw: float = 100.0       # Max charge rate
    max_discharge_kw: float = 100.0    # Max discharge rate
    charge_efficiency: float = 0.95    # Round-trip charge efficiency
    discharge_efficiency: float = 0.95
    soc_min: float = 0.10              # Minimum state of charge (10%)
    soc_max: float = 0.95              # Maximum state of charge (95%)
    initial_soc: float = 0.50          # Starting SoC
    degradation_cost_per_kwh: float = 0.005   # € per kWh cycled


@dataclass
class GridConfig:
    """Grid tariff and export parameters."""
    import_price_eur_kwh: float = 0.18   # Import from grid
    export_price_eur_kwh: float = 0.06   # Export to grid (feed-in tariff)
    max_import_kw: float = 500.0
    max_export_kw: float = 200.0
    carbon_factor_kg_kwh: float = 0.45   # kg CO2 per kWh grid import


@dataclass
class MPCConfig:
    """MPC rolling horizon configuration."""
    horizon_hours: int = 24             # Look-ahead window
    dt_hours: float = 1.0              # Time step (hours)
    self_consumption_weight: float = 2.0  # Reward for solar self-use
    curtailment_penalty: float = 1.0     # Penalty for wasted solar


# ─────────────────────────────────────────────
# MPC Optimiser
# ─────────────────────────────────────────────

class MPCController:
    """
    Model Predictive Control for battery dispatch.

    Solves a convex quadratic program at each timestep:
      Minimise:  grid_import_cost - solar_selfconsumption_reward + battery_degradation
      Subject to: energy balance, SoC bounds, charge/discharge rate limits
    """

    def __init__(
        self,
        battery: BatteryConfig = None,
        grid: GridConfig = None,
        mpc: MPCConfig = None,
    ):
        self.battery = battery or BatteryConfig()
        self.grid = grid or GridConfig()
        self.mpc = mpc or MPCConfig()
        self.current_soc = self.battery.initial_soc

    def optimise(
        self,
        solar_forecast_kw: np.ndarray,
        demand_forecast_kw: np.ndarray,
        current_soc: Optional[float] = None,
        grid_price_forecast: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Solve the MPC optimisation problem for the given forecasts.

        Args:
            solar_forecast_kw:  Solar production forecast [kW] length N
            demand_forecast_kw: Demand forecast [kW] length N
            current_soc:        Current battery state of charge [0–1]
            grid_price_forecast: Time-of-use tariff [€/kWh] length N

        Returns:
            dict with optimal dispatch schedule and economics.
        """
        N = min(len(solar_forecast_kw), len(demand_forecast_kw), self.mpc.horizon_hours)
        soc0 = current_soc if current_soc is not None else self.current_soc

        solar = solar_forecast_kw[:N]
        demand = demand_forecast_kw[:N]
        dt = self.mpc.dt_hours
        C = self.battery.capacity_kwh
        P_ch_max = self.battery.max_charge_kw
        P_dis_max = self.battery.max_discharge_kw
        eta_c = self.battery.charge_efficiency
        eta_d = self.battery.discharge_efficiency
        soc_min = self.battery.soc_min
        soc_max = self.battery.soc_max
        deg_cost = self.battery.degradation_cost_per_kwh

        # Import price (time-varying or flat)
        if grid_price_forecast is not None:
            p_import = grid_price_forecast[:N]
        else:
            p_import = np.full(N, self.grid.import_price_eur_kwh)
        p_export = np.full(N, self.grid.export_price_eur_kwh)

        # ── Decision Variables ─────────────────────────
        P_charge = cp.Variable(N, nonneg=True)      # Battery charging [kW]
        P_discharge = cp.Variable(N, nonneg=True)   # Battery discharging [kW]
        P_grid_im = cp.Variable(N, nonneg=True)     # Grid import [kW]
        P_grid_ex = cp.Variable(N, nonneg=True)     # Grid export [kW]
        P_curtail = cp.Variable(N, nonneg=True)     # Curtailed solar [kW]
        soc = cp.Variable(N + 1)                    # State of charge [0–1]

        # ── Constraints ────────────────────────────────
        constraints = [soc[0] == soc0]

        for t in range(N):
            # SoC dynamics
            constraints.append(
                soc[t + 1] == soc[t]
                + (eta_c * P_charge[t] - P_discharge[t] / eta_d) * dt / C
            )
            # Energy balance: solar + discharge + import = demand + charge + export + curtailment
            net_solar = solar[t] - P_curtail[t]
            constraints.append(
                net_solar + P_discharge[t] + P_grid_im[t]
                == demand[t] + P_charge[t] + P_grid_ex[t]
            )

        # Bounds
        constraints += [
            P_charge <= P_ch_max,
            P_discharge <= P_dis_max,
            P_grid_im <= self.grid.max_import_kw,
            P_grid_ex <= self.grid.max_export_kw,
            P_curtail <= solar,
            soc[1:] >= soc_min,
            soc[1:] <= soc_max,
        ]

        # ── Objective ──────────────────────────────────
        # Grid cost
        grid_cost = cp.sum(cp.multiply(p_import, P_grid_im) * dt)
        grid_revenue = cp.sum(cp.multiply(p_export, P_grid_ex) * dt)

        # Battery degradation cost
        batt_deg = deg_cost * cp.sum(P_charge + P_discharge) * dt

        # Curtailment penalty (waste is bad)
        curtail_pen = self.mpc.curtailment_penalty * cp.sum(P_curtail) * dt

        # Self-consumption reward (higher weight → prefer using own solar)
        self_consumption = cp.sum(
            cp.minimum(solar - P_curtail, demand + P_charge)
        ) * dt * self.mpc.self_consumption_weight * self.grid.import_price_eur_kwh

        objective = cp.Minimize(
            grid_cost - grid_revenue + batt_deg + curtail_pen - self_consumption
        )

        # ── Solve ──────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"MPC solver status: {prob.status} — using fallback dispatch")
            return self._fallback_dispatch(solar, demand, soc0, N, dt)

        # ── Extract Results ────────────────────────────
        schedule = pd.DataFrame({
            "solar_kw": solar,
            "demand_kw": demand,
            "p_charge_kw": np.round(np.maximum(0, P_charge.value), 2),
            "p_discharge_kw": np.round(np.maximum(0, P_discharge.value), 2),
            "p_grid_import_kw": np.round(np.maximum(0, P_grid_im.value), 2),
            "p_grid_export_kw": np.round(np.maximum(0, P_grid_ex.value), 2),
            "curtailed_kw": np.round(np.maximum(0, P_curtail.value), 2),
            "soc": np.round(soc.value[1:], 4),
            "grid_import_price": p_import,
        })

        # Economics summary
        total_import_cost = float(np.sum(schedule["p_grid_import_kw"] * p_import * dt))
        total_export_revenue = float(np.sum(schedule["p_grid_export_kw"] * p_export * dt))
        total_solar_used = float(np.sum(np.minimum(solar, demand)))
        total_solar = float(np.sum(solar))
        self_consumption_rate = total_solar_used / max(total_solar, 1e-6)
        grid_import_reduction = 1 - (np.sum(schedule["p_grid_import_kw"]) / max(np.sum(demand), 1e-6))
        co2_avoided_kg = total_solar_used * self.grid.carbon_factor_kg_kwh

        summary = {
            "status": prob.status,
            "total_import_cost_eur": round(total_import_cost, 2),
            "total_export_revenue_eur": round(total_export_revenue, 2),
            "net_cost_eur": round(total_import_cost - total_export_revenue, 2),
            "self_consumption_rate_pct": round(self_consumption_rate * 100, 2),
            "grid_import_reduction_pct": round(grid_import_reduction * 100, 2),
            "co2_avoided_kg": round(co2_avoided_kg, 2),
            "curtailed_energy_kwh": round(float(np.sum(schedule["curtailed_kw"] * dt)), 2),
            "final_soc": round(float(soc.value[-1]), 4),
        }

        # Update internal SoC (first timestep's result)
        self.current_soc = float(soc.value[1])

        return {"schedule": schedule, "summary": summary}

    def _fallback_dispatch(
        self,
        solar: np.ndarray,
        demand: np.ndarray,
        soc0: float,
        N: int,
        dt: float,
    ) -> dict:
        """Rule-based fallback: charge when surplus, discharge when deficit."""
        C = self.battery.capacity_kwh
        soc = soc0
        records = []

        for t in range(N):
            net = solar[t] - demand[t]  # positive = surplus, negative = deficit
            p_charge = p_discharge = p_import = p_export = p_curtail = 0.0

            if net > 0:  # surplus → charge battery
                charge_headroom = (self.battery.soc_max - soc) * C / dt
                p_charge = min(net, self.battery.max_charge_kw, charge_headroom)
                remaining_surplus = net - p_charge
                p_export = min(remaining_surplus, self.grid.max_export_kw)
                p_curtail = max(0, remaining_surplus - p_export)
            else:  # deficit → discharge battery
                discharge_avail = (soc - self.battery.soc_min) * C / dt
                p_discharge = min(-net, self.battery.max_discharge_kw, discharge_avail)
                p_import = max(0, -net - p_discharge)

            soc += (p_charge * self.battery.charge_efficiency - p_discharge / self.battery.discharge_efficiency) * dt / C
            soc = np.clip(soc, self.battery.soc_min, self.battery.soc_max)

            records.append({
                "solar_kw": solar[t], "demand_kw": demand[t],
                "p_charge_kw": round(p_charge, 2), "p_discharge_kw": round(p_discharge, 2),
                "p_grid_import_kw": round(p_import, 2), "p_grid_export_kw": round(p_export, 2),
                "curtailed_kw": round(p_curtail, 2), "soc": round(soc, 4),
            })

        schedule = pd.DataFrame(records)
        summary = {
            "status": "fallback",
            "total_import_cost_eur": float(round(schedule["p_grid_import_kw"].sum() * self.grid.import_price_eur_kwh * dt, 2)),
            "self_consumption_rate_pct": 0.0,
            "final_soc": round(soc, 4),
        }
        return {"schedule": schedule, "summary": summary}

    def get_next_action(
        self,
        solar_forecast_kw: np.ndarray,
        demand_forecast_kw: np.ndarray,
        current_soc: float,
    ) -> dict:
        """
        Return only the immediate (t=0) dispatch decision for real-time control.
        """
        result = self.optimise(solar_forecast_kw, demand_forecast_kw, current_soc)
        row = result["schedule"].iloc[0]
        return {
            "action": "charge" if row["p_charge_kw"] > row["p_discharge_kw"] else
                      "discharge" if row["p_discharge_kw"] > row["p_charge_kw"] else "idle",
            "p_charge_kw": row["p_charge_kw"],
            "p_discharge_kw": row["p_discharge_kw"],
            "p_grid_import_kw": row["p_grid_import_kw"],
            "p_grid_export_kw": row["p_grid_export_kw"],
            "soc_after": row["soc"],
            "summary": result["summary"],
        }


# ─────────────────────────────────────────────
# District Energy Transfer (P2P Balancing)
# ─────────────────────────────────────────────

def balance_districts(district_states: list[dict]) -> list[dict]:
    """
    Compute optimal energy transfers between districts to minimise curtailment.

    Args:
        district_states: List of {district_id, surplus_kw, deficit_kw, priority}

    Returns:
        List of {from_district, to_district, transfer_kw, reason}
    """
    PRIORITY_WEIGHTS = {"hospital": 3.0, "school": 2.0, "residential": 1.5, "commercial": 1.0}

    surplus = [(d["district_id"], d.get("surplus_kw", 0)) for d in district_states if d.get("surplus_kw", 0) > 0]
    deficit = [(d["district_id"], d.get("deficit_kw", 0), d.get("district_type", "residential"))
               for d in district_states if d.get("deficit_kw", 0) > 0]

    # Sort by priority
    deficit.sort(key=lambda x: -PRIORITY_WEIGHTS.get(x[2], 1.0))

    transfers = []
    surplus_remaining = {d: s for d, s in surplus}

    for (def_id, def_kw, def_type) in deficit:
        needed = def_kw
        for sur_id in list(surplus_remaining.keys()):
            if needed <= 0:
                break
            available = surplus_remaining[sur_id]
            if available <= 0:
                continue
            transfer = min(needed, available)
            transfers.append({
                "from_district": sur_id,
                "to_district": def_id,
                "transfer_kw": round(transfer, 2),
                "reason": f"Priority coverage for {def_type}",
            })
            surplus_remaining[sur_id] -= transfer
            needed -= transfer

    return transfers


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPC Battery Dispatch Controller")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--capacity-kwh", type=float, default=500.0)
    args = parser.parse_args()

    np.random.seed(42)
    solar = np.clip(np.random.normal(200, 80, args.horizon), 0, None)
    demand = np.clip(np.random.normal(250, 50, args.horizon), 50, None)

    batt = BatteryConfig(capacity_kwh=args.capacity_kwh)
    mpc = MPCController(battery=batt)

    print(f"\nRunning MPC for {args.horizon}-hour horizon...")
    result = mpc.optimise(solar, demand, current_soc=0.5)

    print("\nOptimal dispatch schedule:")
    print(result["schedule"].to_string(max_rows=24))
    print("\nEconomics summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
