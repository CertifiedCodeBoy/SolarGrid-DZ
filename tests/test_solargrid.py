"""
SolarGrid DZ — Backend Integration Tests
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ─────────────────────────────────────────────
# ML Model Tests
# ─────────────────────────────────────────────

class TestSolarForecast:
    def test_generate_synthetic_data(self):
        from ml.solar_forecast import generate_synthetic_data
        df = generate_synthetic_data(start="2024-01-01", end="2024-01-31")
        assert len(df) == 31 * 24
        assert "production_kw" in df.columns
        assert (df["production_kw"] >= 0).all()

    def test_feature_engineering(self):
        from ml.solar_forecast import generate_synthetic_data, engineer_features
        df = generate_synthetic_data(start="2024-01-01", end="2024-02-28")
        df_feat = engineer_features(df)
        assert "hour_sin" in df_feat.columns
        assert "solar_elevation" in df_feat.columns
        assert df_feat.isnull().sum().sum() == 0

    def test_solar_elevation_daytime(self):
        from ml.solar_forecast import _solar_elevation
        import pandas as pd
        # Noon on summer solstice in Algiers
        ts = pd.DatetimeIndex(["2024-06-21 12:00:00"])
        elev = _solar_elevation(ts)
        assert elev[0] > 60  # Should be very high in summer


class TestDemandForecast:
    def test_generate_demand_data(self):
        from ml.demand_forecast import generate_demand_data
        df = generate_demand_data(district_type="residential", start="2024-01-01", end="2024-01-31")
        assert len(df) == 31 * 24
        assert "demand_kw" in df.columns
        assert (df["demand_kw"] > 0).all()

    def test_feature_engineering(self):
        from ml.demand_forecast import generate_demand_data, engineer_demand_features
        df = generate_demand_data(start="2024-01-01", end="2024-03-31")
        df_feat = engineer_demand_features(df)
        assert "hour_sin" in df_feat.columns
        assert "demand_lag_1h" in df_feat.columns
        assert df_feat.isnull().sum().sum() == 0


class TestFaultDetection:
    def test_generate_panel_data(self):
        from ml.fault_detection import generate_panel_data
        df = generate_panel_data(n_panels=5, n_days=30)
        assert "panel_id" in df.columns
        assert "power_w" in df.columns
        assert "is_fault" in df.columns

    def test_fault_injection(self):
        from ml.fault_detection import generate_panel_data
        df = generate_panel_data(n_panels=20, n_days=30, anomaly_rate=0.3)
        assert df["is_fault"].sum() > 0, "Should have some faults injected"


# ─────────────────────────────────────────────
# MPC Optimiser Tests
# ─────────────────────────────────────────────

class TestMPCController:
    def test_fallback_dispatch(self):
        from optimizer.mpc_controller import MPCController, BatteryConfig
        batt = BatteryConfig(capacity_kwh=100.0, max_charge_kw=20.0, max_discharge_kw=20.0)
        mpc = MPCController(battery=batt)

        solar = np.full(24, 50.0)   # flat 50 kW
        demand = np.full(24, 60.0)  # flat 60 kW (deficit)
        result = mpc._fallback_dispatch(solar, demand, soc0=0.8, N=24, dt=1.0)

        assert "schedule" in result
        assert len(result["schedule"]) == 24
        # Should be discharging to cover deficit
        total_discharge = result["schedule"]["p_discharge_kw"].sum()
        assert total_discharge > 0

    def test_energy_balance(self):
        from optimizer.mpc_controller import MPCController, BatteryConfig
        batt = BatteryConfig(capacity_kwh=200.0, max_charge_kw=40.0, max_discharge_kw=40.0)
        mpc = MPCController(battery=batt)

        solar = np.array([0]*8 + [100]*8 + [0]*8)   # solar only at midday
        demand = np.full(24, 50.0)
        result = mpc._fallback_dispatch(solar, demand, soc0=0.5, N=24, dt=1.0)

        sched = result["schedule"]
        # Check energy balance: solar + discharge + import = demand + charge + export + curtail
        for _, row in sched.iterrows():
            lhs = row["solar_kw"] + row["p_discharge_kw"] + row["p_grid_import_kw"]
            rhs = row["demand_kw"] + row["p_charge_kw"] + row["p_grid_export_kw"] + row["curtailed_kw"]
            assert abs(lhs - rhs) < 1e-3, f"Energy balance violated: lhs={lhs:.3f} rhs={rhs:.3f}"

    def test_soc_bounds(self):
        from optimizer.mpc_controller import MPCController, BatteryConfig
        batt = BatteryConfig(capacity_kwh=100.0, soc_min=0.1, soc_max=0.95)
        mpc = MPCController(battery=batt)

        solar = np.full(24, 200.0)  # large excess
        demand = np.full(24, 30.0)
        result = mpc._fallback_dispatch(solar, demand, soc0=0.5, N=24, dt=1.0)

        for soc in result["schedule"]["soc"]:
            assert 0.09 <= soc <= 0.96, f"SoC {soc} out of bounds"


# ─────────────────────────────────────────────
# FastAPI Tests
# ─────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.main import app
        # Override lifespan for testing (skip external connections)
        from contextlib import asynccontextmanager
        from fastapi import FastAPI

        @asynccontextmanager
        async def test_lifespan(app):
            from optimizer.dispatch_engine import create_default_engine
            app.state.dispatch_engine = create_default_engine(n_districts=3)
            yield

        test_app = FastAPI(lifespan=test_lifespan)
        from backend.api.routes import forecasts, installations, dispatch, carbon, maintenance
        test_app.include_router(installations.router, prefix="/api/v1/installations")
        test_app.include_router(forecasts.router, prefix="/api/v1/forecasts")
        test_app.include_router(dispatch.router, prefix="/api/v1/dispatch")
        test_app.include_router(carbon.router, prefix="/api/v1/carbon")
        test_app.include_router(maintenance.router, prefix="/api/v1/maintenance")
        return TestClient(test_app)

    def test_list_installations(self, client):
        resp = client.get("/api/v1/installations/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_carbon_report(self, client):
        resp = client.get("/api/v1/carbon/report?period_days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert "co2_avoided_kg" in data
        assert data["co2_avoided_kg"] > 0

    def test_national_target(self, client):
        resp = client.get("/api/v1/carbon/national-target")
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_year"] == 2030
        assert data["target_capacity_gw"] == 22.0
