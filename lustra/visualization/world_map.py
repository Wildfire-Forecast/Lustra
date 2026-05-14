"""Folium-based world map UI for rendering fire zones.

This module is intentionally decoupled from the simulator. It accepts GeoJSON
that can be produced by a future YOLO pipeline and renders it on an interactive
folium map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import folium


@dataclass
class WorldMapConfig:
    """Configuration for the world map UI."""

    center_lat: float = 39.0
    center_lon: float = 35.0
    zoom_start: int = 10
    tiles_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    tiles_attribution: str = "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors"
    tiles_name: str = "OpenStreetMap"
    tiles_subdomains: str = "abc"


class WorldMapUI:
    """Simple Folium world map UI for rendering fire zones."""

    def __init__(self, config: Optional[WorldMapConfig] = None) -> None:
        self.config = config or WorldMapConfig()
        self._map = folium.Map(
            location=[self.config.center_lat, self.config.center_lon],
            zoom_start=self.config.zoom_start,
            tiles=None,
        )
        folium.TileLayer(
            tiles=self.config.tiles_url,
            attr=self.config.tiles_attribution,
            name=self.config.tiles_name,
            subdomains=self.config.tiles_subdomains,
            max_zoom=19,
        ).add_to(self._map)

    def add_fire_zones_geojson(
        self,
        geojson: Dict[str, Any],
        *,
        name: str = "Fire Zones",
        style: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add fire zones as GeoJSON to the map.

        Args:
            geojson: A GeoJSON FeatureCollection or Feature.
            name: Layer name.
            style: Optional folium style dict (e.g., color, weight, fillOpacity).
        """
        style = style or {"color": "#ff4500", "weight": 2, "fillOpacity": 0.35}
        layer = folium.GeoJson(geojson, name=name, style_function=lambda _: style)
        layer.add_to(self._map)

    def add_markers(
        self,
        points: Iterable[Dict[str, Any]],
        *,
        name: str = "Detections",
        icon_color: str = "red",
    ) -> None:
        """Optional helper to add marker points.

        Each point should include: {"lat": float, "lon": float, "popup": str}
        """
        feature_group = folium.FeatureGroup(name=name)
        for p in points:
            folium.Marker(
                location=[p["lat"], p["lon"]],
                popup=p.get("popup"),
                icon=folium.Icon(color=icon_color),
            ).add_to(feature_group)
        feature_group.add_to(self._map)

    def save_html(self, output_path: str) -> None:
        """Save the map to an HTML file."""
        folium.LayerControl().add_to(self._map)
        self._map.save(output_path)

    @property
    def folium_map(self) -> folium.Map:
        """Expose the internal folium map for custom use."""
        return self._map
