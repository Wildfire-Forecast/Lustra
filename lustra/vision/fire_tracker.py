"""Fire detection tracker and GeoJSON export utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class FireTrack:
    track_id: int
    centroid_xy: np.ndarray
    polygon_xy: List[Tuple[float, float]]
    last_seen_s: float
    hits: int
    conf_avg: float


class FireTracker:
    def __init__(
        self,
        *,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        distance_threshold_m: float = 6.0,
        ttl_s: float = 4.0,
        min_hits: int = 2,
        smoothing: float = 0.7,
    ) -> None:
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.distance_threshold_m = float(distance_threshold_m)
        self.ttl_s = float(ttl_s)
        self.min_hits = int(min_hits)
        self.smoothing = float(smoothing)
        self._tracks: List[FireTrack] = []
        self._next_id = 1
        self._meters_per_deg_lat = 111_111.0
        self._meters_per_deg_lon = 111_111.0 * max(0.1, cos(radians(self.origin_lat)))

    def update(self, detections: Iterable[Dict[str, object]], timestamp_s: float) -> None:
        for det in detections:
            poly = det.get("polygon_xy")
            conf = float(det.get("confidence", 0.0))
            poly_xy = self._sanitize_polygon(poly)
            if len(poly_xy) < 3:
                continue

            centroid = np.mean(np.array(poly_xy, dtype=np.float32), axis=0)
            track = self._find_nearest_track(centroid)
            if track is None:
                self._tracks.append(
                    FireTrack(
                        track_id=self._next_id,
                        centroid_xy=centroid,
                        polygon_xy=poly_xy,
                        last_seen_s=timestamp_s,
                        hits=1,
                        conf_avg=conf,
                    )
                )
                self._next_id += 1
            else:
                track.centroid_xy = (self.smoothing * track.centroid_xy) + ((1.0 - self.smoothing) * centroid)
                track.polygon_xy = poly_xy
                track.last_seen_s = timestamp_s
                track.hits += 1
                track.conf_avg = (track.conf_avg * (track.hits - 1) + conf) / track.hits

        self._tracks = [t for t in self._tracks if (timestamp_s - t.last_seen_s) <= self.ttl_s]

    def to_geojson(self, timestamp_s: float) -> Dict[str, object]:
        features = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue
            if (timestamp_s - track.last_seen_s) > self.ttl_s:
                continue

            ring = [self._world_to_lonlat(xy) for xy in track.polygon_xy]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "track_id": track.track_id,
                        "confidence": float(track.conf_avg),
                        "last_seen": float(track.last_seen_s),
                        "hits": track.hits,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring],
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}

    def world_xy_to_lonlat(self, xy: Tuple[float, float]) -> List[float]:
        return self._world_to_lonlat(xy)

    def get_active_tracks(self, timestamp_s: float, *, include_unconfirmed: bool = False) -> List[Dict[str, object]]:
        tracks = []
        for track in self._tracks:
            age = float(timestamp_s - track.last_seen_s)
            if age > self.ttl_s:
                continue
            if (track.hits < self.min_hits) and (not include_unconfirmed):
                continue
            tracks.append(
                {
                    "track_id": track.track_id,
                    "hits": track.hits,
                    "confidence": float(track.conf_avg),
                    "age_s": age,
                    "centroid_xy": track.centroid_xy.copy(),
                }
            )
        tracks.sort(key=lambda t: (t["hits"], -t["age_s"]), reverse=True)
        return tracks

    def _world_to_lonlat(self, xy: Tuple[float, float]) -> List[float]:
        x, y = xy
        lat = self.origin_lat + (y / self._meters_per_deg_lat)
        lon = self.origin_lon + (x / self._meters_per_deg_lon)
        return [float(lon), float(lat)]

    def _find_nearest_track(self, centroid: np.ndarray) -> Optional[FireTrack]:
        best_track = None
        best_dist = None
        for track in self._tracks:
            dist = float(np.linalg.norm(track.centroid_xy - centroid))
            if dist <= self.distance_threshold_m and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_track = track
        return best_track

    @staticmethod
    def _sanitize_polygon(poly: object) -> List[Tuple[float, float]]:
        if not poly:
            return []
        points: List[Tuple[float, float]] = []
        for pt in poly:
            if pt is None:
                continue
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = pt[0], pt[1]
            elif hasattr(pt, "tolist"):
                arr = pt.tolist()
                if len(arr) < 2:
                    continue
                x, y = arr[0], arr[1]
            else:
                continue
            if x is None or y is None:
                continue
            points.append((float(x), float(y)))
        return points
