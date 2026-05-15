"""Validation utilities for the prediction pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from lustra.prediction.fuel_model import FuelModel, get_fuel_model
from lustra.prediction.rothermel import SpreadResult, compute_spread
from lustra.prediction.weather import WeatherSnapshot


@dataclass(frozen=True)
class ReferenceScenario:
    name: str
    fuel_code: str
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_10m_ms: float
    slope_deg: float = 0.0
    aspect_deg: float = 0.0
    live_herb_moisture_pct: float = 60.0
    live_woody_moisture_pct: float = 90.0
    expected_ros_m_per_min_band: Tuple[float, float] = (0.0, float("inf"))
    citation: str = ""


@dataclass(frozen=True)
class ReferenceResult:
    scenario: ReferenceScenario
    fuel: FuelModel
    spread: SpreadResult
    ros_m_per_min: float
    passed: bool


REFERENCE_SCENARIOS: List[ReferenceScenario] = [
    ReferenceScenario("GR2 low wind cool dry", "GR2", 25.0, 30.0, 2.0,
        expected_ros_m_per_min_band=(0.5, 8.0)),
    ReferenceScenario("GR2 hot dry high wind", "GR2", 32.0, 20.0, 5.0,
        expected_ros_m_per_min_band=(3.0, 25.0)),
    ReferenceScenario("SH5 dry shrub on slope", "SH5", 30.0, 25.0, 4.0,
        slope_deg=15.0, aspect_deg=180.0,
        expected_ros_m_per_min_band=(5.0, 60.0)),
    ReferenceScenario("NB1 nonburnable", "NB1", 35.0, 10.0, 10.0,
        expected_ros_m_per_min_band=(0.0, 0.0)),
    ReferenceScenario("GR2 calm", "GR2", 15.0, 80.0, 0.0,
        expected_ros_m_per_min_band=(0.0, 2.0)),
]


def run_reference_scenario(scenario: ReferenceScenario) -> ReferenceResult:
    fuel = get_fuel_model(scenario.fuel_code)
    weather = WeatherSnapshot(0.0, 0.0, "reference",
        scenario.temperature_c, scenario.relative_humidity_pct,
        scenario.wind_speed_10m_ms, 270.0)
    spread = compute_spread(
        fuel, weather, slope_deg=scenario.slope_deg, aspect_deg=scenario.aspect_deg,
        live_herb_moisture_pct=scenario.live_herb_moisture_pct,
        live_woody_moisture_pct=scenario.live_woody_moisture_pct,
    )
    ros = spread.rate_of_spread_ms * 60.0
    lo, hi = scenario.expected_ros_m_per_min_band
    return ReferenceResult(scenario, fuel, spread, ros, lo <= ros <= hi)


def run_all_reference_scenarios() -> List[ReferenceResult]:
    return [run_reference_scenario(s) for s in REFERENCE_SCENARIOS]


def sorensen_index(predicted_ring_lonlat, observed_ring_lonlat, *, grid_resolution_m: float = 5.0) -> float:
    """Sorensen (Dice) similarity, 2*|A and B| / (|A|+|B|)."""
    if len(predicted_ring_lonlat) < 3 or len(observed_ring_lonlat) < 3:
        return 0.0
    all_lon = [p[0] for p in list(predicted_ring_lonlat) + list(observed_ring_lonlat)]
    all_lat = [p[1] for p in list(predicted_ring_lonlat) + list(observed_ring_lonlat)]
    centroid_lat = float(np.mean(all_lat))
    centroid_lon = float(np.mean(all_lon))
    earth_r = 6_371_000.0
    m_per_lat = math.pi * earth_r / 180.0
    m_per_lon = m_per_lat * max(0.1, math.cos(math.radians(centroid_lat)))

    def to_xy(ring):
        return np.array([((lon - centroid_lon) * m_per_lon, (lat - centroid_lat) * m_per_lat) for lon, lat in ring], dtype=np.float64)

    pred_xy = to_xy(predicted_ring_lonlat)
    obs_xy = to_xy(observed_ring_lonlat)
    x_min = float(min(pred_xy[:, 0].min(), obs_xy[:, 0].min())) - grid_resolution_m
    x_max = float(max(pred_xy[:, 0].max(), obs_xy[:, 0].max())) + grid_resolution_m
    y_min = float(min(pred_xy[:, 1].min(), obs_xy[:, 1].min())) - grid_resolution_m
    y_max = float(max(pred_xy[:, 1].max(), obs_xy[:, 1].max())) + grid_resolution_m
    n_cols = max(2, int(math.ceil((x_max - x_min) / grid_resolution_m)))
    n_rows = max(2, int(math.ceil((y_max - y_min) / grid_resolution_m)))
    xs = x_min + (np.arange(n_cols) + 0.5) * grid_resolution_m
    ys = y_min + (np.arange(n_rows) + 0.5) * grid_resolution_m
    grid_x, grid_y = np.meshgrid(xs, ys)

    def raster(ring_xy):
        px = ring_xy[:, 0]; py = ring_xy[:, 1]
        inside = np.zeros_like(grid_x, dtype=bool)
        n = ring_xy.shape[0]
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            cond = ((yi > grid_y) != (yj > grid_y)) & (grid_x < (xj - xi) * (grid_y - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            j = i
        return inside

    a = raster(pred_xy)
    b = raster(obs_xy)
    aA, aB = int(a.sum()), int(b.sum())
    inter = int(np.logical_and(a, b).sum())
    return 0.0 if aA + aB == 0 else float(2.0 * inter / (aA + aB))


def format_report(results: Optional[List[ReferenceResult]] = None) -> str:
    if results is None:
        results = run_all_reference_scenarios()
    lines = ["Reference scenario suite:"]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lo, hi = r.scenario.expected_ros_m_per_min_band
        lines.append(f"  [{status}] {r.scenario.name:30s}  ROS={r.ros_m_per_min:6.2f} m/min  band=({lo:.2f}, {hi:.2f})")
    return "\n".join(lines)
