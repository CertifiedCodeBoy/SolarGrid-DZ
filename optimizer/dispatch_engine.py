"""
SolarGrid DZ — Dispatch Engine
Orchestrates ML forecasts → MPC controller → real-time BESS dispatch decisions.
Runs as a continuous loop (every hour) or in simulation mode.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from optimizer.mpc_controller import MPCController, BatteryConfig, GridConfig, MPCConfig, balance_districts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dispatch_engine")

# ─────────────────────────────────────────────
# District & Installation Registry
# ─────────────────────────────────────────────

@dataclass
class SolarInstallation:
    installation_id: str
    district_id: str
    capacity_kw: float
    battery_capacity_kwh: float
    district_type: str = "residential"
    lat: float = 36.75
    lon: float = 3.05


@dataclass
class DispatchState:
    installation_id: str
    current_soc: float
    last_action: str = "idle"
    last_dispatch_kw: float = 0.0
    total_solar_kwh: float = 0.0
    total_import_kwh: float = 0.0
    total_export_kwh: float = 0.0
    total_co2_avoided_kg: float = 0.0
    updated_at: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Dispatch Engine
# ─────────────────────────────────────────────

class DispatchEngine:
    def __init__(
        self,
        installations: list[SolarInstallation],
        horizon_hours: int = 24,
        simulation_mode: bool = False,
    ):
        self.installations = {inst.installation_id: inst for inst in installations}
        self.horizon = horizon_hours
        self.simulation_mode = simulation_mode
        self.states: dict[str, DispatchState] = {}
        self.controllers: dict[str, MPCController] = {}
        self.history: list[dict] = []

        # Lazy-import forecasters to avoid startup cost
        self._solar_forecaster = None
        self._demand_forecaster = None

        for inst in installations:
            self.states[inst.installation_id] = DispatchState(
                installation_id=inst.installation_id,
                current_soc=0.50,
            )
            self.controllers[inst.installation_id] = MPCController(
                battery=BatteryConfig(
                    capacity_kwh=inst.battery_capacity_kwh,
                    max_charge_kw=inst.battery_capacity_kwh * 0.2,
                    max_discharge_kw=inst.battery_capacity_kwh * 0.2,
                ),
                grid=GridConfig(),
                mpc=MPCConfig(horizon_hours=self.horizon),
            )

        logger.info(f"Dispatch engine initialised with {len(installations)} installations")

    # ─── Forecasting ─────────────────────────

    def _get_solar_forecast(self, inst: SolarInstallation) -> np.ndarray:
        """Get 48-hour solar production forecast [kW]."""
        try:
            from ml.solar_forecast import SolarForecaster, fetch_weather_forecast
            if self._solar_forecaster is None:
                self._solar_forecaster = SolarForecaster()
            weather = fetch_weather_forecast(lat=inst.lat, lon=inst.lon, hours=self.horizon)
            fc = self._solar_forecaster.forecast(weather_forecast=weather, horizon_hours=self.horizon)
            return fc["forecast_kw"].values
        except Exception as e:
            logger.warning(f"Solar forecast unavailable for {inst.installation_id}: {e} → using synthetic")
            return self._synthetic_solar(inst.capacity_kw)

    def _get_demand_forecast(self, inst: SolarInstallation) -> np.ndarray:
        """Get 48-hour demand forecast [kW]."""
        try:
            from ml.demand_forecast import DemandForecaster, generate_demand_data
            if self._demand_forecaster is None:
                self._demand_forecaster = DemandForecaster()
            history = generate_demand_data(district_type=inst.district_type, start="2025-01-01", end="2025-12-31")
            fc = self._demand_forecaster.forecast(history, horizon_hours=self.horizon)
            return fc["forecast_kw"].values
        except Exception as e:
            logger.warning(f"Demand forecast unavailable for {inst.installation_id}: {e} → using synthetic")
            return self._synthetic_demand(inst)

    @staticmethod
    def _synthetic_solar(capacity_kw: float) -> np.ndarray:
        now = datetime.utcnow()
        hours = np.arange(48)
        hour_of_day = (now.hour + hours) % 24
        solar = capacity_kw * np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        solar += np.random.normal(0, capacity_kw * 0.05, 48)
        return np.clip(solar, 0, capacity_kw)

    @staticmethod
    def _synthetic_demand(inst: SolarInstallation) -> np.ndarray:
        peak = inst.capacity_kw * 0.6
        now = datetime.utcnow()
        hours = np.arange(48)
        hour_of_day = (now.hour + hours) % 24
        demand = peak * (0.4 + 0.6 * np.exp(-0.5 * ((hour_of_day - 19) / 3) ** 2))
        demand += np.random.normal(0, peak * 0.05, 48)
        return np.clip(demand, peak * 0.2, peak * 1.1)

    # ─── Single-step Dispatch ────────────────

    def dispatch_installation(self, installation_id: str) -> dict:
        """Run MPC for a single installation and return the dispatch action."""
        inst = self.installations[installation_id]
        state = self.states[installation_id]
        controller = self.controllers[installation_id]

        solar_fc = self._get_solar_forecast(inst)
        demand_fc = self._get_demand_forecast(inst)
        action = controller.get_next_action(solar_fc, demand_fc, state.current_soc)

        # Update state
        dt = 1.0  # 1 hour
        state.current_soc = action["soc_after"]
        state.last_action = action["action"]
        state.last_dispatch_kw = action["p_charge_kw"] if action["action"] == "charge" else action["p_discharge_kw"]
        state.total_solar_kwh += float(solar_fc[0]) * dt
        state.total_import_kwh += float(action["p_grid_import_kw"]) * dt
        state.total_export_kwh += float(action["p_grid_export_kw"]) * dt
        state.total_co2_avoided_kg += float(solar_fc[0]) * 0.45 * dt  # 0.45 kg/kWh
        state.updated_at = datetime.utcnow()

        record = {
            "ts": datetime.utcnow().isoformat(),
            "installation_id": installation_id,
            "district_id": inst.district_id,
            **action,
            "solar_forecast_kw": round(float(solar_fc[0]), 2),
            "demand_forecast_kw": round(float(demand_fc[0]), 2),
            "soc": state.current_soc,
        }
        self.history.append(record)
        logger.info(
            f"[{installation_id}] action={action['action']} "
            f"charge={action['p_charge_kw']} kW discharge={action['p_discharge_kw']} kW "
            f"import={action['p_grid_import_kw']} kW SOC={state.current_soc:.2%}"
        )
        return record

    def dispatch_all(self) -> list[dict]:
        """Dispatch all installations and compute cross-district balancing."""
        results = []
        district_states = []

        for inst_id in self.installations:
            rec = self.dispatch_installation(inst_id)
            results.append(rec)
            inst = self.installations[inst_id]
            surplus = max(0, rec["solar_forecast_kw"] - rec["demand_forecast_kw"] - rec["p_charge_kw"])
            deficit = max(0, rec["demand_forecast_kw"] - rec["solar_forecast_kw"] - rec["p_discharge_kw"])
            district_states.append({
                "district_id": inst.district_id,
                "district_type": inst.district_type,
                "surplus_kw": surplus,
                "deficit_kw": deficit,
            })

        # Cross-district balancing
        transfers = balance_districts(district_states)
        if transfers:
            logger.info(f"District transfers: {transfers}")

        return {"dispatches": results, "district_transfers": transfers}

    # ─── Simulation ──────────────────────────

    def simulate(self, days: int = 30) -> pd.DataFrame:
        """
        Run a multi-day simulation stepping hour-by-hour.

        Returns:
            DataFrame with hourly system-level KPIs.
        """
        logger.info(f"Starting {days}-day simulation...")
        hours = days * 24
        rows = []

        for h in range(hours):
            hourly_totals = {
                "hour": h,
                "total_solar_kw": 0.0,
                "total_demand_kw": 0.0,
                "total_charge_kw": 0.0,
                "total_discharge_kw": 0.0,
                "total_import_kw": 0.0,
                "total_export_kw": 0.0,
                "total_curtailed_kw": 0.0,
                "avg_soc": 0.0,
            }

            for inst_id, inst in self.installations.items():
                solar = self._synthetic_solar(inst.capacity_kw)
                demand = self._synthetic_demand(inst)
                state = self.states[inst_id]
                controller = self.controllers[inst_id]

                action = controller.get_next_action(solar, demand, state.current_soc)
                state.current_soc = float(action["soc_after"])

                hourly_totals["total_solar_kw"] += solar[0]
                hourly_totals["total_demand_kw"] += demand[0]
                hourly_totals["total_charge_kw"] += action["p_charge_kw"]
                hourly_totals["total_discharge_kw"] += action["p_discharge_kw"]
                hourly_totals["total_import_kw"] += action["p_grid_import_kw"]
                hourly_totals["total_export_kw"] += action["p_grid_export_kw"]
                hourly_totals["avg_soc"] += state.current_soc

            n = len(self.installations)
            hourly_totals["avg_soc"] /= n
            rows.append(hourly_totals)

        df = pd.DataFrame(rows)
        df["self_consumption_rate"] = (
            (df["total_solar_kw"] - df["total_export_kw"]) / df["total_solar_kw"].clip(lower=1e-6)
        ).clip(0, 1)

        total_solar = df["total_solar_kw"].sum()
        total_import = df["total_import_kw"].sum()
        total_demand = df["total_demand_kw"].sum()

        logger.info(
            f"\nSimulation complete ({days} days):"
            f"\n  Total solar generated: {total_solar:.0f} kWh"
            f"\n  Total grid import:     {total_import:.0f} kWh"
            f"\n  Self-consumption rate: {(1 - total_import / max(total_demand, 1)) * 100:.1f}%"
            f"\n  CO₂ avoided:           {(total_solar - df['total_export_kw'].sum()) * 0.45:.0f} kg"
        )
        return df

    # ─── Continuous Loop ─────────────────────

    async def run_async(self, interval_seconds: int = 3600):
        """Run the dispatch engine continuously, executing every `interval_seconds`."""
        logger.info(f"Dispatch engine starting — interval={interval_seconds}s")
        while True:
            try:
                results = self.dispatch_all()
                logger.info(f"Dispatch cycle complete: {len(results['dispatches'])} installations")
            except Exception as exc:
                logger.error(f"Dispatch cycle error: {exc}", exc_info=True)
            await asyncio.sleep(interval_seconds)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_default_engine(n_districts: int = 5, simulation_mode: bool = True) -> DispatchEngine:
    """Create a demo engine with simulated installations."""
    district_types = ["residential", "commercial", "industrial", "hospital", "school"]
    installations = []
    for i in range(n_districts):
        d_type = district_types[i % len(district_types)]
        installations.append(SolarInstallation(
            installation_id=f"INST-{i+1:03d}",
            district_id=f"DZ-ALG-{i+1:02d}",
            capacity_kw=float(np.random.choice([200, 350, 500, 750, 1000])),
            battery_capacity_kwh=float(np.random.choice([200, 400, 600, 1000])),
            district_type=d_type,
        ))
    return DispatchEngine(installations, simulation_mode=simulation_mode)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SolarGrid DZ — Dispatch Engine")
    parser.add_argument("--mode", choices=["run", "simulate"], default="simulate")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--districts", type=int, default=5)
    args = parser.parse_args()

    engine = create_default_engine(n_districts=args.districts, simulation_mode=True)

    if args.mode == "simulate":
        sim_df = engine.simulate(days=args.days)
        print("\nHourly simulation summary (first 48h):")
        print(sim_df.head(48).to_string())

    elif args.mode == "run":
        asyncio.run(engine.run_async(interval_seconds=3600))
