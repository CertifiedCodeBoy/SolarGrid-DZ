"""SolarGrid DZ — Pydantic models for API request/response validation."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class InstallationBase(BaseModel):
    installation_id: str
    district_id: str
    capacity_kw: float = Field(..., gt=0)
    battery_capacity_kwh: float = Field(..., ge=0)
    district_type: str = "residential"
    lat: float = 36.75
    lon: float = 3.05


class InstallationCreate(InstallationBase):
    pass


class InstallationResponse(InstallationBase):
    current_soc: float = 0.5
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True


class SolarTelemetry(BaseModel):
    installation_id: str
    power_kw: float
    dc_voltage_v: float = 380.0
    dc_current_a: float = 0.0
    temperature_c: float = 25.0
    timestamp: Optional[datetime] = None


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_kw: float
    lower_kw: float
    upper_kw: float


class ForecastResponse(BaseModel):
    installation_id: str
    generated_at: datetime
    horizon_hours: int
    forecast: list[ForecastPoint]


class DispatchAction(BaseModel):
    installation_id: str
    action: str  # "charge" | "discharge" | "idle"
    p_charge_kw: float
    p_discharge_kw: float
    p_grid_import_kw: float
    p_grid_export_kw: float
    soc_after: float
    timestamp: datetime


class DispatchSchedule(BaseModel):
    installation_id: str
    n_hours: int
    schedule: list[dict]
    summary: dict


class CarbonReport(BaseModel):
    district_id: Optional[str] = None
    period_days: int
    total_solar_kwh: float
    co2_avoided_kg: float
    co2_avoided_tons: float
    equivalent_trees: float
    equivalent_cars_off_road: float
    self_consumption_rate_pct: float
    grid_import_reduction_pct: float
    generated_at: datetime


class PanelHealthEntry(BaseModel):
    panel_id: str
    anomaly_rate: float
    avg_anomaly_score: float
    max_anomaly_score: float
    dominant_fault: str
    maintenance_priority: str  # "OK" | "MEDIUM" | "HIGH" | "CRITICAL"


class MaintenanceReport(BaseModel):
    installation_id: str
    n_panels_checked: int
    n_panels_flagged: int
    report: list[PanelHealthEntry]
    generated_at: datetime


class DistrictEnergyBalance(BaseModel):
    district_id: str
    district_type: str
    solar_kw: float
    demand_kw: float
    battery_soc: float
    net_kw: float  # positive = surplus
    transfers_in_kw: float
    transfers_out_kw: float


class SystemOverview(BaseModel):
    timestamp: datetime
    n_installations: int
    total_solar_kw: float
    total_demand_kw: float
    total_battery_soc_avg: float
    total_grid_import_kw: float
    total_grid_export_kw: float
    system_self_consumption_pct: float
    co2_avoided_today_kg: float
    district_balances: list[DistrictEnergyBalance]
