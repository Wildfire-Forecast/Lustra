"""Weather inputs for the Rothermel fire spread model.

Pulls current observations from Open-Meteo (no API key required) and exposes
helpers that convert raw meteorology into the variables a Rothermel/FARSITE
pipeline actually consumes:

- 10 m open wind  ->  midflame wind, via the Albini and Baughman (1979)
  wind adjustment factor as parameterised in Andrews (2012, RMRS-GTR-266).
- 2 m air temperature and relative humidity  ->  1-hr dead fuel moisture,
  via the Simard (1968) equilibrium moisture content equations.

References
----------
Albini, F. A. and Baughman, R. G. (1979). Estimating windspeeds for
    predicting wildland fire behavior. USDA Forest Service Research Paper
    INT-221.
Andrews, P. L. (2012). Modeling wind adjustment factor and midflame wind
    speed for Rothermel's surface fire spread model. USDA Forest Service
    General Technical Report RMRS-GTR-266.
Simard, A. J. (1968). The moisture content of forest fuels - I. A review of
    the basic concepts. Canadian Department of Forestry and Rural
    Development, Forest Fire Research Institute, Information Report FF-X-14.
Open-Meteo (https://open-meteo.com) - free weather API, CC-BY 4.0.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_FEET_PER_METER = 3.28084


@dataclass(frozen=True)
class WeatherSnapshot:
    """Point-in-time weather at a single lat/lon."""

    latitude: float
    longitude: float
    timestamp_iso: str
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_10m_ms: float
    wind_direction_10m_deg: float
    fetched_at_unix: float = field(default_factory=lambda: time.time())

    @property
    def wind_vector_10m_ms(self) -> Tuple[float, float]:
        bearing_rad = math.radians(self.wind_direction_10m_deg)
        u_east = -self.wind_speed_10m_ms * math.sin(bearing_rad)
        v_north = -self.wind_speed_10m_ms * math.cos(bearing_rad)
        return u_east, v_north


class WeatherProvider:
    def __init__(
        self,
        *,
        cache_ttl_s: float = 600.0,
        grid_resolution_deg: float = 0.05,
        timeout_s: float = 10.0,
    ) -> None:
        self.cache_ttl_s = float(cache_ttl_s)
        self.grid_resolution_deg = float(grid_resolution_deg)
        self.timeout_s = float(timeout_s)
        self._cache: Dict[Tuple[float, float], WeatherSnapshot] = {}

    def get(self, latitude: float, longitude: float) -> WeatherSnapshot:
        key = self._cache_key(latitude, longitude)
        cached = self._cache.get(key)
        if cached is not None and (time.time() - cached.fetched_at_unix) <= self.cache_ttl_s:
            return cached
        snapshot = self._fetch(latitude, longitude)
        self._cache[key] = snapshot
        return snapshot

    def _cache_key(self, latitude: float, longitude: float) -> Tuple[float, float]:
        res = self.grid_resolution_deg
        return (round(latitude / res) * res, round(longitude / res) * res)

    def _fetch(self, latitude: float, longitude: float) -> WeatherSnapshot:
        params = {
            "latitude": f"{latitude:.5f}",
            "longitude": f"{longitude:.5f}",
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "lustra-prediction/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload.get("current")
        if not isinstance(current, dict):
            raise RuntimeError(f"Open-Meteo response missing 'current' block: {payload!r}")
        return WeatherSnapshot(
            latitude=float(payload.get("latitude", latitude)),
            longitude=float(payload.get("longitude", longitude)),
            timestamp_iso=str(current.get("time", "")),
            temperature_c=float(current["temperature_2m"]),
            relative_humidity_pct=float(current["relative_humidity_2m"]),
            wind_speed_10m_ms=float(current["wind_speed_10m"]),
            wind_direction_10m_deg=float(current["wind_direction_10m"]),
        )


def _ten_m_to_twenty_ft_wind(wind_10m_ms: float, roughness_length_m: float = 0.03) -> float:
    if wind_10m_ms <= 0.0:
        return 0.0
    twenty_ft_m = 20.0 / _FEET_PER_METER
    return wind_10m_ms * math.log(twenty_ft_m / roughness_length_m) / math.log(10.0 / roughness_length_m)


def midflame_wind_speed(
    wind_10m_ms: float,
    *,
    fuel_bed_depth_m: float,
    canopy_cover_frac: float = 0.0,
    canopy_height_m: float = 0.0,
    crown_fill_frac: float = 0.0,
    roughness_length_m: float = 0.03,
) -> float:
    """Midflame wind (m/s) via Albini-Baughman (1979) WAF, Andrews 2012 eqs. 47/49."""
    if fuel_bed_depth_m <= 0.0:
        raise ValueError("fuel_bed_depth_m must be positive")
    wind_20ft_ms = _ten_m_to_twenty_ft_wind(wind_10m_ms, roughness_length_m=roughness_length_m)
    fuel_depth_ft = fuel_bed_depth_m * _FEET_PER_METER

    if canopy_cover_frac <= 0.05:
        waf = 1.83 / math.log((20.0 + 0.36 * fuel_depth_ft) / (0.13 * fuel_depth_ft))
    else:
        if canopy_height_m <= 0.0:
            raise ValueError("canopy_height_m must be positive when canopy_cover_frac > 0.05")
        canopy_height_ft = canopy_height_m * _FEET_PER_METER
        if crown_fill_frac <= 0.0:
            crown_fill_frac = min(1.0, canopy_cover_frac * canopy_height_ft / 20.0)
        waf = 0.555 / (math.sqrt(crown_fill_frac * canopy_height_ft) * math.log((20.0 + 0.36 * canopy_height_ft) / (0.13 * canopy_height_ft)))

    waf = max(0.0, min(waf, 1.0))
    return wind_20ft_ms * waf


def one_hour_dead_fuel_moisture(temperature_c: float, relative_humidity_pct: float) -> float:
    """Simard (1968) EMC equations as a 1-hr dead fuel moisture proxy. Returns %."""
    rh = max(0.0, min(100.0, float(relative_humidity_pct)))
    temp_f = temperature_c * 9.0 / 5.0 + 32.0
    if rh < 10.0:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f
    elif rh <= 50.0:
        emc = 2.22749 + 0.160107 * rh - 0.014784 * temp_f
    else:
        emc = 21.0606 + 0.005565 * rh * rh - 0.00035 * rh * temp_f - 0.483199 * rh
    return max(1.0, emc)
