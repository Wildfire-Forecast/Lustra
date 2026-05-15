"""Rothermel surface fire spread - thin SI adapter over pyretechnics.

The actual physics comes from pyretechnics' validated Rothermel
implementation; this module handles unit conversion, builds the
6-slot moisture array, and returns an SI :class:`SpreadResult`.

References
----------
Rothermel, R. C. (1972). RP INT-115.
Andrews, P. L. (2018). RMRS-GTR-371.
Scott, J. H. and Burgan, R. E. (2005). RMRS-GTR-153.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pyretechnics.fuel_models as _pf
import pyretechnics.surface_fire as _ps

from lustra.prediction.fuel_model import FuelModel
from lustra.prediction.weather import (
    WeatherSnapshot, midflame_wind_speed, one_hour_dead_fuel_moisture,
)


@dataclass(frozen=True)
class SpreadResult:
    rate_of_spread_ms: float
    direction_of_max_spread_deg: float
    fireline_intensity_kw_per_m: float
    flame_length_m: float
    length_to_width_ratio: float
    eccentricity: float
    midflame_wind_ms: float
    one_hour_fuel_moisture_pct: float

    @classmethod
    def zero(cls, *, midflame_wind_ms: float = 0.0, one_hour_fuel_moisture_pct: float = 0.0) -> "SpreadResult":
        return cls(0.0, float("nan"), 0.0, 0.0, 1.0, 0.0, midflame_wind_ms, one_hour_fuel_moisture_pct)


def compute_spread(
    fuel: FuelModel,
    weather: WeatherSnapshot,
    *,
    slope_deg: float = 0.0,
    aspect_deg: float = 0.0,
    canopy_cover_frac: float = 0.0,
    canopy_height_m: float = 0.0,
    live_herb_moisture_pct: float = 60.0,
    live_woody_moisture_pct: float = 90.0,
    moisture_10hr_pct: Optional[float] = None,
    moisture_100hr_pct: Optional[float] = None,
    surface_lw_ratio_model: str = "behave",
) -> SpreadResult:
    if not fuel.is_burnable:
        return SpreadResult.zero()

    one_hr_pct = one_hour_dead_fuel_moisture(weather.temperature_c, weather.relative_humidity_pct)
    ten_hr_pct = moisture_10hr_pct if moisture_10hr_pct is not None else (one_hr_pct + 1.0)
    hundred_hr_pct = moisture_100hr_pct if moisture_100hr_pct is not None else (one_hr_pct + 2.0)

    moisture_array = (
        one_hr_pct / 100.0,
        ten_hr_pct / 100.0,
        hundred_hr_pct / 100.0,
        one_hr_pct / 100.0,
        live_herb_moisture_pct / 100.0,
        live_woody_moisture_pct / 100.0,
    )

    raw_model = _pf.get_fuel_model(int(fuel.number))
    moist_model = _pf.moisturize(raw_model, moisture_array)

    midflame_ms = midflame_wind_speed(
        weather.wind_speed_10m_ms,
        fuel_bed_depth_m=fuel.fuel_bed_depth_m,
        canopy_cover_frac=canopy_cover_frac,
        canopy_height_m=canopy_height_m,
    )
    midflame_m_per_min = midflame_ms * 60.0
    slope_rise_run = math.tan(math.radians(max(0.0, slope_deg)))

    no_wind_no_slope = _ps.calc_surface_fire_behavior_no_wind_no_slope(moist_model)
    max_behavior = _ps.calc_surface_fire_behavior_max(
        no_wind_no_slope, midflame_m_per_min, float(weather.wind_direction_10m_deg),
        slope_rise_run, float(aspect_deg), surface_lw_ratio_model=surface_lw_ratio_model,
    )

    ros_m_per_min = float(max_behavior["max_spread_rate"])
    direction_x, direction_y, _ = max_behavior["max_spread_direction"]
    bearing_deg = float("nan") if ros_m_per_min <= 0.0 else (math.degrees(math.atan2(direction_x, direction_y))) % 360.0

    return SpreadResult(
        rate_of_spread_ms=ros_m_per_min / 60.0,
        direction_of_max_spread_deg=bearing_deg,
        fireline_intensity_kw_per_m=float(max_behavior["max_fireline_intensity"]),
        flame_length_m=float(max_behavior["max_flame_length"]),
        length_to_width_ratio=float(max_behavior["length_to_width_ratio"]),
        eccentricity=float(max_behavior["eccentricity"]),
        midflame_wind_ms=midflame_ms,
        one_hour_fuel_moisture_pct=one_hr_pct,
    )
