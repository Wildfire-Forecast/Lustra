"""Fire perimeter propagation.

Two algorithms are provided:

- :func:`propagate_huygens` (default): each input perimeter vertex is
  treated as an ignition point that emits an Anderson (1983) elliptical
  wave after time t. The predicted perimeter is the convex envelope of
  the union of those ellipses. This is Finney (1998) FARSITE-style
  propagation; no timestepping error, no vertex blow-up at long horizons.
- :func:`propagate` (Richards 1990): outward-normal march with timestep.
  Exposed for comparison and strongly time-varying weather.

The ignition-point ellipse has the rear focus at the vertex, semi-major
along the head-fire direction, and shape set by the local L/W ratio.
Polar form with focus at origin (head angle theta):

    R(theta) = R_max * (1 - e) / (1 - e * cos(theta))

with R(0)=R_max at the head and R(pi)=R_max*(1-e)/(1+e) at the back,
matching the Anderson (1983) head/back ratio.

References
----------
Anderson, H. E. (1983). RP INT-305.
Richards, G. D. (1990). Int. J. Numer. Methods Eng. 30, 1163-1179.
Finney, M. A. (1998). RP RMRS-RP-4 (FARSITE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lustra.prediction.rothermel import SpreadResult


_EARTH_RADIUS_M = 6_371_000.0

SpreadFn = Callable[[float, float], SpreadResult]


@dataclass(frozen=True)
class PropagationConfig:
    timestep_s: float = 60.0
    target_vertex_spacing_m: float = 5.0
    max_vertex_spacing_m: float = 15.0
    min_vertex_spacing_m: float = 1.0
    max_vertices_per_polygon: int = 4000


@dataclass(frozen=True)
class PredictedPerimeter:
    track_id: int
    horizon_s: float
    ring_lonlat: List[Tuple[float, float]]
    area_m2: float


def propagate_huygens(
    initial_perimeter_lonlat: Sequence[Tuple[float, float]],
    *,
    track_id: int,
    horizons_s: Sequence[float],
    spread_fn: SpreadFn,
    samples_per_ellipse: int = 36,
) -> List[PredictedPerimeter]:
    horizons_sorted = sorted({float(h) for h in horizons_s if h > 0})
    if not horizons_sorted:
        return []

    ring = _close_ring(list(initial_perimeter_lonlat))
    if len(ring) < 4:
        return []

    centroid_lon = float(np.mean([p[0] for p in ring[:-1]]))
    centroid_lat = float(np.mean([p[1] for p in ring[:-1]]))
    meters_per_deg_lat = math.pi * _EARTH_RADIUS_M / 180.0
    meters_per_deg_lon = meters_per_deg_lat * max(0.1, math.cos(math.radians(centroid_lat)))

    def xy_to_lonlat(x: float, y: float) -> Tuple[float, float]:
        return (centroid_lon + x / meters_per_deg_lon, centroid_lat + y / meters_per_deg_lat)

    vertices_xy = np.array(
        [((lon - centroid_lon) * meters_per_deg_lon, (lat - centroid_lat) * meters_per_deg_lat) for lon, lat in ring[:-1]],
        dtype=np.float64,
    )

    spreads: List[Optional[Tuple[float, float, float, np.ndarray]]] = []
    any_growth = False
    for x, y in vertices_xy:
        lon, lat = xy_to_lonlat(float(x), float(y))
        s = spread_fn(lat, lon)
        if not math.isfinite(s.direction_of_max_spread_deg) or s.rate_of_spread_ms <= 0.0:
            spreads.append(None)
            continue
        any_growth = True
        head_rad = math.radians(s.direction_of_max_spread_deg)
        head_unit = np.array([math.sin(head_rad), math.cos(head_rad)], dtype=np.float64)
        spreads.append((float(s.rate_of_spread_ms), float(s.eccentricity), float(s.length_to_width_ratio), head_unit))

    if not any_growth:
        return []

    # Baseline hull area of just the seed vertices (no propagation). The
    # propagated hull must exceed this to count as "real" growth; for an
    # irregular concave input polygon, this prevents the hull from being
    # reported as a "prediction" when nothing actually moved.
    seed_hull = _convex_hull_ring(vertices_xy.copy())
    seed_hull_area = _ring_area_m2(seed_hull) if seed_hull.shape[0] >= 3 else 0.0

    results: List[PredictedPerimeter] = []
    for horizon_s in horizons_sorted:
        points_xy: List[np.ndarray] = []
        for vertex_xy, spread in zip(vertices_xy, spreads):
            if spread is None:
                points_xy.append(vertex_xy.copy())
                continue
            ros_max, e, _lw, head_unit = spread
            a = ros_max * horizon_s / max(1.0 + e, 1e-6)
            b = a * math.sqrt(max(1.0 - e * e, 0.0))
            c = a * e
            center = vertex_xy + c * head_unit
            perp_unit = np.array([head_unit[1], -head_unit[0]], dtype=np.float64)
            for k in range(samples_per_ellipse):
                phi = 2.0 * math.pi * k / samples_per_ellipse
                local_x = a * math.cos(phi)
                local_y = b * math.sin(phi)
                points_xy.append(center + local_x * head_unit + local_y * perp_unit)

        if not points_xy:
            continue
        all_points = np.array(points_xy, dtype=np.float64)
        hull_ring = _convex_hull_ring(all_points)
        if hull_ring.shape[0] < 3:
            continue
        area_m2 = _ring_area_m2(hull_ring)
        if area_m2 <= seed_hull_area * 1.001:
            continue

        ring_lonlat = [xy_to_lonlat(float(x), float(y)) for x, y in hull_ring]
        ring_lonlat.append(ring_lonlat[0])
        results.append(PredictedPerimeter(track_id, float(horizon_s), ring_lonlat, float(area_m2)))
    return results


def _convex_hull_ring(points_xy: np.ndarray) -> np.ndarray:
    pts = points_xy[np.lexsort((points_xy[:, 1], points_xy[:, 0]))]
    if pts.shape[0] <= 2:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def propagate(
    initial_perimeter_lonlat: Sequence[Tuple[float, float]],
    *,
    track_id: int,
    horizons_s: Sequence[float],
    spread_fn: SpreadFn,
    config: Optional[PropagationConfig] = None,
) -> List[PredictedPerimeter]:
    cfg = config or PropagationConfig()
    horizons_sorted = sorted({float(h) for h in horizons_s if h > 0})
    if not horizons_sorted:
        return []

    ring = _close_ring(list(initial_perimeter_lonlat))
    if len(ring) < 4:
        return []

    centroid_lon = float(np.mean([p[0] for p in ring[:-1]]))
    centroid_lat = float(np.mean([p[1] for p in ring[:-1]]))
    meters_per_deg_lat = math.pi * _EARTH_RADIUS_M / 180.0
    meters_per_deg_lon = meters_per_deg_lat * max(0.1, math.cos(math.radians(centroid_lat)))

    def lonlat_to_xy(lon, lat):
        return ((lon - centroid_lon) * meters_per_deg_lon, (lat - centroid_lat) * meters_per_deg_lat)

    def xy_to_lonlat(x, y):
        return (centroid_lon + x / meters_per_deg_lon, centroid_lat + y / meters_per_deg_lat)

    vertices_xy = np.array([lonlat_to_xy(lon, lat) for lon, lat in ring[:-1]], dtype=np.float64)
    vertices_xy = _resample_ring(vertices_xy, cfg)
    initial_area_m2 = _ring_area_m2(vertices_xy)

    results: List[PredictedPerimeter] = []
    elapsed_s = 0.0
    for horizon_s in horizons_sorted:
        while elapsed_s + cfg.timestep_s * 0.5 < horizon_s:
            dt = min(cfg.timestep_s, horizon_s - elapsed_s)
            vertices_xy = _advance_one_step(vertices_xy, dt, xy_to_lonlat, spread_fn)
            vertices_xy = _resample_ring(vertices_xy, cfg)
            elapsed_s += dt
            if vertices_xy.shape[0] < 4:
                break
        area_m2 = _ring_area_m2(vertices_xy)
        if area_m2 <= initial_area_m2 * 1.001:
            continue
        ring_lonlat = [xy_to_lonlat(float(x), float(y)) for x, y in vertices_xy]
        ring_lonlat.append(ring_lonlat[0])
        results.append(PredictedPerimeter(track_id, float(horizon_s), ring_lonlat, float(area_m2)))
    return results


def propagate_geojson(
    fire_geojson: Dict,
    *,
    horizons_s: Sequence[float],
    spread_fn: SpreadFn,
    config: Optional[PropagationConfig] = None,
    method: str = "huygens",
) -> Dict:
    features = []
    for feature in fire_geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue
        coords = geometry.get("coordinates") or []
        if not coords:
            continue
        outer_ring = coords[0]
        track_id = int(feature.get("properties", {}).get("track_id", 0))
        ring_lonlat = [(float(p[0]), float(p[1])) for p in outer_ring]
        if method == "huygens":
            predicted = propagate_huygens(ring_lonlat, track_id=track_id, horizons_s=horizons_s, spread_fn=spread_fn)
        else:
            predicted = propagate(ring_lonlat, track_id=track_id, horizons_s=horizons_s, spread_fn=spread_fn, config=config)
        for p in predicted:
            features.append({
                "type": "Feature",
                "properties": {
                    "track_id": p.track_id,
                    "horizon_s": p.horizon_s,
                    "horizon_min": round(p.horizon_s / 60.0, 2),
                    "area_m2": p.area_m2,
                },
                "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in p.ring_lonlat]]},
            })
    return {"type": "FeatureCollection", "features": features}


def _close_ring(ring):
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _advance_one_step(vertices_xy, dt_s, xy_to_lonlat, spread_fn):
    n = vertices_xy.shape[0]
    new_xy = np.empty_like(vertices_xy)
    for i in range(n):
        prev_pt = vertices_xy[(i - 1) % n]
        curr_pt = vertices_xy[i]
        next_pt = vertices_xy[(i + 1) % n]
        tangent = next_pt - prev_pt
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-9:
            new_xy[i] = curr_pt
            continue
        tangent /= tangent_norm
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
        lon, lat = xy_to_lonlat(float(curr_pt[0]), float(curr_pt[1]))
        spread = spread_fn(lat, lon)
        if not math.isfinite(spread.direction_of_max_spread_deg) or spread.rate_of_spread_ms <= 0.0:
            new_xy[i] = curr_pt
            continue
        head_rad = math.radians(spread.direction_of_max_spread_deg)
        head_vec = np.array([math.sin(head_rad), math.cos(head_rad)], dtype=np.float64)
        cos_theta = float(np.dot(normal, head_vec))
        e = float(spread.eccentricity)
        ros_max = float(spread.rate_of_spread_ms)
        denom = 1.0 - e * cos_theta
        ros_normal = ros_max if denom <= 1e-6 else ros_max * (1.0 - e) / denom
        ros_normal = max(ros_normal, 0.0)
        new_xy[i] = curr_pt + normal * (ros_normal * dt_s)
    return new_xy


def _resample_ring(vertices_xy, cfg):
    n = vertices_xy.shape[0]
    if n < 3:
        return vertices_xy
    resampled = []
    for i in range(n):
        a = vertices_xy[i]
        b = vertices_xy[(i + 1) % n]
        resampled.append(a)
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length > cfg.max_vertex_spacing_m:
            n_inserts = int(math.ceil(length / cfg.target_vertex_spacing_m)) - 1
            for k in range(1, n_inserts + 1):
                t = k / (n_inserts + 1)
                resampled.append(a + t * seg)
    out = np.array(resampled, dtype=np.float64)
    if out.shape[0] <= 3:
        return out
    kept = [out[0]]
    for i in range(1, out.shape[0]):
        if float(np.linalg.norm(out[i] - kept[-1])) >= cfg.min_vertex_spacing_m:
            kept.append(out[i])
    if len(kept) > cfg.max_vertices_per_polygon:
        stride = math.ceil(len(kept) / cfg.max_vertices_per_polygon)
        kept = kept[::stride]
    return np.array(kept, dtype=np.float64)


def _ring_area_m2(vertices_xy):
    if vertices_xy.shape[0] < 3:
        return 0.0
    xs = vertices_xy[:, 0]
    ys = vertices_xy[:, 1]
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
