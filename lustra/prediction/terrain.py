"""Terrain inputs (elevation, slope, aspect) for Rothermel.

Uses Open-Topo-Data SRTM30m + Horn (1981) central differences.
Aggressively grid-cached because the public endpoint allows only
1000 calls/day.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


OPEN_TOPO_DATA_URL = "https://api.opentopodata.org/v1/srtm30m"
_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class TerrainSample:
    latitude: float
    longitude: float
    elevation_m: float
    slope_deg: float
    aspect_deg: float


class TerrainProvider:
    def __init__(self, *, neighbor_spacing_m: float = 30.0, cache_grid_deg: float = 0.001,
                 request_pause_s: float = 1.05, timeout_s: float = 15.0) -> None:
        self.neighbor_spacing_m = float(neighbor_spacing_m)
        self.cache_grid_deg = float(cache_grid_deg)
        self.request_pause_s = float(request_pause_s)
        self.timeout_s = float(timeout_s)
        self._cache: Dict[Tuple[float, float], TerrainSample] = {}
        self._last_request_at = 0.0

    def get(self, latitude: float, longitude: float) -> TerrainSample:
        key = self._cache_key(latitude, longitude)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sample = self._fetch(latitude, longitude)
        self._cache[key] = sample
        return sample

    def _cache_key(self, latitude: float, longitude: float) -> Tuple[float, float]:
        res = self.cache_grid_deg
        return (round(latitude / res) * res, round(longitude / res) * res)

    def _fetch(self, latitude: float, longitude: float) -> TerrainSample:
        meters_per_deg_lat = math.pi * _EARTH_RADIUS_M / 180.0
        meters_per_deg_lon = meters_per_deg_lat * max(0.1, math.cos(math.radians(latitude)))
        d_lat = self.neighbor_spacing_m / meters_per_deg_lat
        d_lon = self.neighbor_spacing_m / meters_per_deg_lon
        points = [
            (latitude, longitude),
            (latitude + d_lat, longitude),
            (latitude - d_lat, longitude),
            (latitude, longitude + d_lon),
            (latitude, longitude - d_lon),
        ]
        elevations = self._query_elevations(points)
        center, north, south, east, west = elevations
        dz_dy = (north - south) / (2.0 * self.neighbor_spacing_m)
        dz_dx = (east - west) / (2.0 * self.neighbor_spacing_m)
        slope_deg = math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))
        if dz_dx == 0.0 and dz_dy == 0.0:
            aspect_deg = 0.0
        else:
            aspect_deg = math.degrees(math.atan2(-dz_dx, -dz_dy)) % 360.0
        return TerrainSample(latitude, longitude, float(center), float(slope_deg), float(aspect_deg))

    def _query_elevations(self, points: Iterable[Tuple[float, float]]) -> List[float]:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_pause_s:
            time.sleep(self.request_pause_s - elapsed)
        locations = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
        url = f"{OPEN_TOPO_DATA_URL}?{urllib.parse.urlencode({'locations': locations, 'interpolation': 'cubic'})}"
        request = urllib.request.Request(url, headers={"User-Agent": "lustra-prediction/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._last_request_at = time.time()
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 5:
            raise RuntimeError(f"Unexpected Open-Topo-Data response: {payload!r}")
        return [float(r["elevation"]) if r.get("elevation") is not None else 0.0 for r in results]
