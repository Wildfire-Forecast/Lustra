"""End-to-end fire spread prediction.

PredictionEngine ties weather, fuel-model classification, terrain,
Rothermel and propagation together. It consumes the FireTracker
GeoJSON output and optionally the dry-zone tracker output, so dry
polygons drive a heavier fuel model inside their footprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lustra.prediction.fuel_model import FuelModel, classify_drone_class, get_fuel_model
from lustra.prediction.propagation import (
    PropagationConfig, SpreadFn, propagate_geojson,
)
from lustra.prediction.rothermel import SpreadResult, compute_spread
from lustra.prediction.terrain import TerrainProvider, TerrainSample
from lustra.prediction.weather import WeatherProvider, WeatherSnapshot


FuelClassifier = Callable[[float, float], FuelModel]


def _default_fuel_classifier(_lat: float, _lon: float) -> FuelModel:
    return classify_drone_class("default")


def _point_in_ring(lon: float, lat: float, ring_lonlat: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon for a single ring."""
    inside = False
    n = len(ring_lonlat)
    j = n - 1
    for i in range(n):
        xi, yi = ring_lonlat[i][0], ring_lonlat[i][1]
        xj, yj = ring_lonlat[j][0], ring_lonlat[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def build_dry_zone_fuel_classifier(
    dry_geojson: Optional[Dict],
    *,
    inside_fuel_code: str = "GR4",
    outside_fuel_code: str = "GR1",
) -> FuelClassifier:
    """Classifier that returns a faster-spreading fuel model inside dry polygons.

    Drone "dry zone" detections are interpreted as patches of cured dry
    grass; inside any polygon Feature in ``dry_geojson`` -> ``GR4``
    (moderate-load dry grass, fast ROS), outside -> ``GR1`` (sparse dry
    grass, much slower). Override the codes per deployment.
    """
    inside_fuel = get_fuel_model(inside_fuel_code)
    outside_fuel = get_fuel_model(outside_fuel_code)
    rings: List[List[List[float]]] = []
    if dry_geojson:
        for feature in dry_geojson.get("features", []):
            geom = feature.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates") or []
            if coords:
                rings.append(coords[0])

    def _fn(latitude: float, longitude: float) -> FuelModel:
        for ring in rings:
            if _point_in_ring(longitude, latitude, ring):
                return inside_fuel
        return outside_fuel

    return _fn


@dataclass
class PredictionEngine:
    weather_provider: WeatherProvider = field(default_factory=WeatherProvider)
    terrain_provider: Optional[TerrainProvider] = None
    fuel_classifier: FuelClassifier = field(default=_default_fuel_classifier)
    propagation_config: PropagationConfig = field(default_factory=PropagationConfig)
    canopy_cover_frac: float = 0.0
    canopy_height_m: float = 0.0
    live_herb_moisture_pct: float = 60.0
    live_woody_moisture_pct: float = 90.0
    use_terrain: bool = False
    weather_override: Optional[WeatherSnapshot] = None

    def __post_init__(self) -> None:
        if self.use_terrain and self.terrain_provider is None:
            self.terrain_provider = TerrainProvider()

    def predict(
        self,
        fire_geojson: Dict,
        *,
        horizons_min: Sequence[float] = (15.0, 30.0, 60.0),
        dry_geojson: Optional[Dict] = None,
    ) -> Dict:
        horizons_s = [float(h) * 60.0 for h in horizons_min if h > 0]
        if not horizons_s or not fire_geojson.get("features"):
            return {"type": "FeatureCollection", "features": []}

        classifier = self.fuel_classifier
        if dry_geojson is not None and dry_geojson.get("features"):
            classifier = build_dry_zone_fuel_classifier(dry_geojson)
        spread_fn = self._build_spread_fn(classifier)
        return propagate_geojson(
            fire_geojson, horizons_s=horizons_s, spread_fn=spread_fn,
            config=self.propagation_config, method="huygens",
        )

    def diagnose(self, latitude: float, longitude: float) -> Tuple[FuelModel, WeatherSnapshot, Optional[TerrainSample], SpreadResult]:
        fuel = self.fuel_classifier(latitude, longitude)
        weather = self.weather_override if self.weather_override is not None else self.weather_provider.get(latitude, longitude)
        terrain = self.terrain_provider.get(latitude, longitude) if self.use_terrain and self.terrain_provider else None
        slope_deg = terrain.slope_deg if terrain else 0.0
        aspect_deg = terrain.aspect_deg if terrain else 0.0
        spread = compute_spread(
            fuel, weather, slope_deg=slope_deg, aspect_deg=aspect_deg,
            canopy_cover_frac=self.canopy_cover_frac, canopy_height_m=self.canopy_height_m,
            live_herb_moisture_pct=self.live_herb_moisture_pct,
            live_woody_moisture_pct=self.live_woody_moisture_pct,
        )
        return fuel, weather, terrain, spread

    def _build_spread_fn(self, classifier: FuelClassifier) -> SpreadFn:
        weather = self.weather_provider
        terrain = self.terrain_provider if self.use_terrain else None
        canopy_cover = self.canopy_cover_frac
        canopy_height = self.canopy_height_m
        herb = self.live_herb_moisture_pct
        woody = self.live_woody_moisture_pct
        override = self.weather_override

        def _fn(latitude: float, longitude: float) -> SpreadResult:
            fuel = classifier(latitude, longitude)
            snapshot = override if override is not None else weather.get(latitude, longitude)
            if terrain is not None:
                t = terrain.get(latitude, longitude)
                slope_deg, aspect_deg = t.slope_deg, t.aspect_deg
            else:
                slope_deg, aspect_deg = 0.0, 0.0
            return compute_spread(
                fuel, snapshot, slope_deg=slope_deg, aspect_deg=aspect_deg,
                canopy_cover_frac=canopy_cover, canopy_height_m=canopy_height,
                live_herb_moisture_pct=herb, live_woody_moisture_pct=woody,
            )

        return _fn
