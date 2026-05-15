"""Fire detection tracker with per-track evidence grids.

Each track owns a small 2D float grid sized to fit just its own observed area
plus a margin. Detections rasterize positive evidence into the grid; the
camera footprint rasterizes negative evidence wherever it observes a cell but
no detection covers it. Polygons exported to the map are extracted as
contours of the thresholded grid. Memory scales with active fire area only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FireTrack:
    track_id: int
    centroid_xy: np.ndarray
    grid: np.ndarray                      # float32, shape (rows, cols), values in [0, 1]
    grid_origin_xy: Tuple[float, float]   # world XY of cell (0, 0)'s top-left corner
    cell_size_m: float
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
        miss_dt_cap_s: float = 0.2,
        # Per-track grid parameters
        cell_size_m: float = 1.0,
        initial_margin_m: float = 2.0,
        growth_chunk_cells: int = 16,
        gain_per_s: float = 2.0,
        decay_per_s: float = 1.0,
        activation_threshold: float = 0.5,
        death_threshold: float = 0.1,
        merge_overlap_min_cells: int = 4,
        merge_dilate_cells: int = 2,
        simplify_eps_m: float = 0.5,
        min_contour_area_cells: int = 2,
    ) -> None:
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.distance_threshold_m = float(distance_threshold_m)
        self.ttl_s = float(ttl_s)
        self.min_hits = int(min_hits)
        self.smoothing = float(smoothing)
        self.miss_dt_cap_s = float(miss_dt_cap_s)

        self.cell_size_m = float(cell_size_m)
        self.initial_margin_m = float(initial_margin_m)
        self.growth_chunk_cells = int(growth_chunk_cells)
        self.gain_per_s = float(gain_per_s)
        self.decay_per_s = float(decay_per_s)
        self.activation_threshold = float(activation_threshold)
        self.death_threshold = float(death_threshold)
        self.merge_overlap_min_cells = int(merge_overlap_min_cells)
        self.merge_dilate_cells = int(merge_dilate_cells)
        self.simplify_eps_m = float(simplify_eps_m)
        self.min_contour_area_cells = int(min_contour_area_cells)

        self._tracks: List[FireTrack] = []
        self._next_id = 1
        self._last_update_s: Optional[float] = None
        self._last_miss_check_s: Optional[float] = None

        self._meters_per_deg_lat = 111_111.0
        self._meters_per_deg_lon = 111_111.0 * max(0.1, cos(radians(self.origin_lat)))

    # ------------------------------------------------------------------ public

    def update(self, detections: Iterable[Dict[str, object]], timestamp_s: float) -> None:
        # Per-frame gain delta in evidence units. Capping dt avoids huge jumps
        # after pauses (e.g. window minimized, debugger break).
        if self._last_update_s is None:
            dt = 0.0
        else:
            dt = min(self.miss_dt_cap_s, max(0.0, timestamp_s - self._last_update_s))
        self._last_update_s = timestamp_s

        for det in detections:
            poly = det.get("polygon_xy")
            conf = float(det.get("confidence", 0.0))
            poly_xy = self._sanitize_polygon(poly)
            if len(poly_xy) < 3:
                continue

            centroid = np.mean(np.array(poly_xy, dtype=np.float32), axis=0)
            track = self._find_match_for_detection(centroid, poly_xy)

            if track is None:
                track = self._spawn_track(centroid, poly_xy, conf, timestamp_s)
            else:
                track.centroid_xy = (
                    self.smoothing * track.centroid_xy
                    + (1.0 - self.smoothing) * centroid
                )
                track.last_seen_s = timestamp_s
                track.hits += 1
                track.conf_avg = (track.conf_avg * (track.hits - 1) + conf) / track.hits
                self._ensure_grid_covers(track, poly_xy)

            # Rasterize detection polygon and apply gain. Use a per-call dt
            # of at least one frame's worth so the very first detection on a
            # newborn track immediately reaches the activation threshold.
            gain_dt = dt if dt > 0.0 else (1.0 / 30.0)
            self._apply_gain(track, poly_xy, conf, gain_dt)

        # Merge tracks whose grids now overlap.
        self._merge_overlapping_tracks(timestamp_s)

        # TTL prune for unconfirmed tracks only; confirmed persist.
        self._tracks = [
            t for t in self._tracks
            if t.hits >= self.min_hits or (timestamp_s - t.last_seen_s) <= self.ttl_s
        ]

    def mark_misses(
        self,
        footprint_xy: List[Tuple[float, float]],
        frame_weight: float,
        timestamp_s: float,
        detection_polys: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> None:
        """Apply per-cell decay to tracks whose grids overlap the camera footprint."""
        if footprint_xy is None or len(footprint_xy) < 3 or frame_weight <= 0.0:
            self._last_miss_check_s = timestamp_s
            return

        if self._last_miss_check_s is None:
            self._last_miss_check_s = timestamp_s
            return
        dt = min(self.miss_dt_cap_s, max(0.0, timestamp_s - self._last_miss_check_s))
        self._last_miss_check_s = timestamp_s
        if dt <= 0.0:
            return

        decay_amount = self.decay_per_s * frame_weight * dt

        survivors: List[FireTrack] = []
        for track in self._tracks:
            # Just-updated tracks already had gain applied in update();
            # skip miss accumulation for them this frame.
            if track.last_seen_s >= timestamp_s - 1e-6:
                survivors.append(track)
                continue

            # Skip tracks whose grids don't overlap the footprint's AABB.
            fp_arr = np.asarray(footprint_xy, dtype=np.float32)
            fp_minx = float(fp_arr[:, 0].min())
            fp_miny = float(fp_arr[:, 1].min())
            fp_maxx = float(fp_arr[:, 0].max())
            fp_maxy = float(fp_arr[:, 1].max())
            tminx, tminy, tmaxx, tmaxy = self._track_aabb(track)
            if fp_maxx < tminx or fp_minx > tmaxx or fp_maxy < tminy or fp_miny > tmaxy:
                survivors.append(track)
                continue

            footprint_mask = self._rasterize_polygon_into_grid(track, footprint_xy)
            if footprint_mask is None or not footprint_mask.any():
                survivors.append(track)
                continue

            detected_mask = None
            if detection_polys:
                detected_mask = np.zeros_like(track.grid, dtype=np.uint8)
                for det_poly in detection_polys:
                    m = self._rasterize_polygon_into_grid(track, det_poly)
                    if m is not None:
                        detected_mask |= m

            if detected_mask is not None:
                decay_region = footprint_mask.astype(bool) & ~detected_mask.astype(bool)
            else:
                decay_region = footprint_mask.astype(bool)

            track.grid[decay_region] -= decay_amount
            np.clip(track.grid, 0.0, 1.0, out=track.grid)

            # If no cell is meaningfully alive anymore, the track is dead.
            if float(track.grid.max()) < self.death_threshold:
                continue
            survivors.append(track)

        self._tracks = survivors

    def to_geojson(self, timestamp_s: float) -> Dict[str, object]:
        features: List[Dict[str, object]] = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue
            polygons = self._extract_polygons(track)
            for poly_world in polygons:
                ring = [self._world_to_lonlat(xy) for xy in poly_world]
                if not ring:
                    continue
                if ring[0] != ring[-1]:
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

    def get_active_tracks(
        self,
        timestamp_s: float,
        *,
        include_unconfirmed: bool = False,
    ) -> List[Dict[str, object]]:
        out = []
        for track in self._tracks:
            age = float(timestamp_s - track.last_seen_s)
            if age > self.ttl_s:
                continue
            if (track.hits < self.min_hits) and (not include_unconfirmed):
                continue
            out.append(
                {
                    "track_id": track.track_id,
                    "hits": track.hits,
                    "confidence": float(track.conf_avg),
                    "age_s": age,
                    "centroid_xy": track.centroid_xy.copy(),
                }
            )
        out.sort(key=lambda t: (t["hits"], -t["age_s"]), reverse=True)
        return out

    # ---------------------------------------------------------------- internals

    def _spawn_track(
        self,
        centroid: np.ndarray,
        poly_xy: List[Tuple[float, float]],
        conf: float,
        timestamp_s: float,
    ) -> FireTrack:
        grid, origin = self._allocate_grid_for_polygon(poly_xy)
        track = FireTrack(
            track_id=self._next_id,
            centroid_xy=centroid,
            grid=grid,
            grid_origin_xy=origin,
            cell_size_m=self.cell_size_m,
            last_seen_s=timestamp_s,
            hits=1,
            conf_avg=conf,
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _allocate_grid_for_polygon(
        self,
        poly_xy: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        xs = [p[0] for p in poly_xy]
        ys = [p[1] for p in poly_xy]
        minx = min(xs) - self.initial_margin_m
        miny = min(ys) - self.initial_margin_m
        maxx = max(xs) + self.initial_margin_m
        maxy = max(ys) + self.initial_margin_m
        cols = max(1, int(np.ceil((maxx - minx) / self.cell_size_m)))
        rows = max(1, int(np.ceil((maxy - miny) / self.cell_size_m)))
        grid = np.zeros((rows, cols), dtype=np.float32)
        return grid, (float(minx), float(miny))

    def _track_aabb(self, track: FireTrack) -> Tuple[float, float, float, float]:
        rows, cols = track.grid.shape
        x0, y0 = track.grid_origin_xy
        return (
            x0,
            y0,
            x0 + cols * track.cell_size_m,
            y0 + rows * track.cell_size_m,
        )

    def _ensure_grid_covers(
        self,
        track: FireTrack,
        poly_xy: List[Tuple[float, float]],
    ) -> None:
        """Grow the track's grid if poly_xy extends past its current bounds."""
        x0, y0, x1, y1 = self._track_aabb(track)
        xs = [p[0] for p in poly_xy]
        ys = [p[1] for p in poly_xy]
        pminx = min(xs) - self.initial_margin_m
        pminy = min(ys) - self.initial_margin_m
        pmaxx = max(xs) + self.initial_margin_m
        pmaxy = max(ys) + self.initial_margin_m
        if pminx >= x0 and pminy >= y0 and pmaxx <= x1 and pmaxy <= y1:
            return  # Already fits.

        # Grow by chunks past what's strictly required to avoid frequent
        # reallocations as the fire spreads.
        chunk_m = self.growth_chunk_cells * self.cell_size_m
        new_x0 = x0
        new_y0 = y0
        new_x1 = x1
        new_y1 = y1
        if pminx < x0:
            new_x0 = x0 - chunk_m * int(np.ceil((x0 - pminx) / chunk_m))
        if pminy < y0:
            new_y0 = y0 - chunk_m * int(np.ceil((y0 - pminy) / chunk_m))
        if pmaxx > x1:
            new_x1 = x1 + chunk_m * int(np.ceil((pmaxx - x1) / chunk_m))
        if pmaxy > y1:
            new_y1 = y1 + chunk_m * int(np.ceil((pmaxy - y1) / chunk_m))

        new_cols = max(1, int(np.ceil((new_x1 - new_x0) / self.cell_size_m)))
        new_rows = max(1, int(np.ceil((new_y1 - new_y0) / self.cell_size_m)))
        new_grid = np.zeros((new_rows, new_cols), dtype=np.float32)

        # Copy old grid into the right slice of the new grid.
        rows, cols = track.grid.shape
        col_off = int(round((x0 - new_x0) / self.cell_size_m))
        row_off = int(round((y0 - new_y0) / self.cell_size_m))
        new_grid[row_off : row_off + rows, col_off : col_off + cols] = track.grid

        track.grid = new_grid
        track.grid_origin_xy = (float(new_x0), float(new_y0))

    def _rasterize_polygon_into_grid(
        self,
        track: FireTrack,
        poly_xy: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Return a uint8 mask matching track.grid with 1s inside poly_xy."""
        if not poly_xy or len(poly_xy) < 3:
            return None
        rows, cols = track.grid.shape
        x0, y0 = track.grid_origin_xy
        pts = np.empty((len(poly_xy), 2), dtype=np.int32)
        for i, (x, y) in enumerate(poly_xy):
            pts[i, 0] = int(np.floor((x - x0) / track.cell_size_m))
            pts[i, 1] = int(np.floor((y - y0) / track.cell_size_m))
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _apply_gain(
        self,
        track: FireTrack,
        poly_xy: List[Tuple[float, float]],
        conf: float,
        dt: float,
    ) -> None:
        mask = self._rasterize_polygon_into_grid(track, poly_xy)
        if mask is None:
            return
        gain = self.gain_per_s * max(0.0, conf) * dt
        if gain <= 0.0:
            return
        track.grid[mask.astype(bool)] += gain
        np.clip(track.grid, 0.0, 1.0, out=track.grid)

    def _find_match_for_detection(
        self,
        centroid: np.ndarray,
        poly_xy: List[Tuple[float, float]],
    ) -> Optional[FireTrack]:
        """Associate a detection with an existing track.

        Strong match: detection overlaps the track's *active* cells.
        Weak match: detection's AABB overlaps a brand-new track with no
        active cells yet (so the second detection on a newborn track can
        still find it before it confirms).
        Otherwise: centroid distance within the configured threshold.
        """
        best_overlap: Optional[FireTrack] = None
        best_overlap_dist: Optional[float] = None
        best_close: Optional[FireTrack] = None
        best_close_dist: Optional[float] = None

        pminx = min(p[0] for p in poly_xy)
        pminy = min(p[1] for p in poly_xy)
        pmaxx = max(p[0] for p in poly_xy)
        pmaxy = max(p[1] for p in poly_xy)

        for track in self._tracks:
            dist = float(np.linalg.norm(track.centroid_xy - centroid))
            x0, y0, x1, y1 = self._track_aabb(track)
            aabb_overlap = not (pmaxx < x0 or pminx > x1 or pmaxy < y0 or pminy > y1)

            strong = False
            weak = False
            if aabb_overlap:
                active_any = bool((track.grid >= self.activation_threshold).any())
                mask = self._rasterize_polygon_into_grid(track, poly_xy)
                grid_overlap_cells = 0
                if mask is not None:
                    grid_overlap_cells = int(
                        ((track.grid >= self.activation_threshold) & mask.astype(bool)).sum()
                    )
                if grid_overlap_cells > 0:
                    strong = True
                elif not active_any:
                    # Track hasn't accumulated active cells yet; AABB overlap
                    # is the best signal we have for newborn tracks.
                    weak = True

            if strong or weak:
                if best_overlap_dist is None or dist < best_overlap_dist:
                    best_overlap_dist = dist
                    best_overlap = track
            elif dist <= self.distance_threshold_m and (
                best_close_dist is None or dist < best_close_dist
            ):
                best_close_dist = dist
                best_close = track

        return best_overlap if best_overlap is not None else best_close

    def _merge_overlapping_tracks(self, timestamp_s: float) -> None:
        """Fuse track pairs whose grids overlap by at least merge_overlap_min_cells."""
        i = 0
        while i < len(self._tracks):
            j = i + 1
            while j < len(self._tracks):
                if self._should_merge(self._tracks[i], self._tracks[j]):
                    self._tracks[i] = self._merge_two(self._tracks[i], self._tracks[j])
                    del self._tracks[j]
                    # Restart inner loop after a merge, in case the union pulled
                    # in another previously-non-overlapping track.
                    j = i + 1
                else:
                    j += 1
            i += 1

    def _should_merge(self, a: FireTrack, b: FireTrack) -> bool:
        """Merge only when both tracks' active (dilated) cells overlap.

        AABB overlap alone — including the empty padding margin — is not a
        merge signal, so two nearby-but-clearly-separate fires won't fuse.
        """
        ax0, ay0, ax1, ay1 = self._track_aabb(a)
        bx0, by0, bx1, by1 = self._track_aabb(b)
        dilate_m = self.merge_dilate_cells * self.cell_size_m
        # Inflate the candidate intersection region by the dilation radius so we
        # see active cells that are near the seam.
        ox0 = max(ax0, bx0) - dilate_m
        oy0 = max(ay0, by0) - dilate_m
        ox1 = min(ax1, bx1) + dilate_m
        oy1 = min(ay1, by1) + dilate_m
        if ox1 <= ox0 or oy1 <= oy0:
            return False
        cols = int(np.ceil((ox1 - ox0) / self.cell_size_m))
        rows = int(np.ceil((oy1 - oy0) / self.cell_size_m))
        if cols <= 0 or rows <= 0:
            return False

        a_mask = self._sample_active_mask_into(a, ox0, oy0, rows, cols)
        b_mask = self._sample_active_mask_into(b, ox0, oy0, rows, cols)
        if not a_mask.any() or not b_mask.any():
            return False

        if self.merge_dilate_cells > 0:
            k = self.merge_dilate_cells * 2 + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            a_mask = cv2.dilate(a_mask, kernel)
            b_mask = cv2.dilate(b_mask, kernel)

        overlap_cells = int(((a_mask > 0) & (b_mask > 0)).sum())
        return overlap_cells >= self.merge_overlap_min_cells

    def _sample_active_mask_into(
        self,
        track: FireTrack,
        x0: float,
        y0: float,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """Project the track's active-cell mask into a target region.

        The target region has world top-left (x0, y0) and dimensions rows×cols
        at cell_size_m resolution. Cells outside the track stay zero.
        """
        mask = np.zeros((rows, cols), dtype=np.uint8)
        tx0, ty0 = track.grid_origin_xy
        col_off = int(round((tx0 - x0) / self.cell_size_m))
        row_off = int(round((ty0 - y0) / self.cell_size_m))
        tr, tc = track.grid.shape

        dst_col0 = max(0, col_off)
        dst_row0 = max(0, row_off)
        dst_col1 = min(cols, col_off + tc)
        dst_row1 = min(rows, row_off + tr)
        if dst_col1 <= dst_col0 or dst_row1 <= dst_row0:
            return mask

        src_col0 = dst_col0 - col_off
        src_row0 = dst_row0 - row_off
        src_col1 = src_col0 + (dst_col1 - dst_col0)
        src_row1 = src_row0 + (dst_row1 - dst_row0)

        active = (track.grid[src_row0:src_row1, src_col0:src_col1] >= self.activation_threshold)
        mask[dst_row0:dst_row1, dst_col0:dst_col1] = active.astype(np.uint8)
        return mask

    def _merge_two(self, a: FireTrack, b: FireTrack) -> FireTrack:
        ax0, ay0, ax1, ay1 = self._track_aabb(a)
        bx0, by0, bx1, by1 = self._track_aabb(b)
        ux0 = min(ax0, bx0); uy0 = min(ay0, by0)
        ux1 = max(ax1, bx1); uy1 = max(ay1, by1)
        cols = max(1, int(np.ceil((ux1 - ux0) / self.cell_size_m)))
        rows = max(1, int(np.ceil((uy1 - uy0) / self.cell_size_m)))
        new_grid = np.zeros((rows, cols), dtype=np.float32)

        def blit(track: FireTrack) -> None:
            x0, y0 = track.grid_origin_xy
            col_off = int(round((x0 - ux0) / self.cell_size_m))
            row_off = int(round((y0 - uy0) / self.cell_size_m))
            r, c = track.grid.shape
            dest = new_grid[row_off : row_off + r, col_off : col_off + c]
            np.maximum(dest, track.grid, out=dest)

        blit(a)
        blit(b)

        # Keep the older (smaller) track id for continuity; sum hits, max conf.
        if a.track_id <= b.track_id:
            keeper_id = a.track_id
            keeper_last_seen = max(a.last_seen_s, b.last_seen_s)
        else:
            keeper_id = b.track_id
            keeper_last_seen = max(a.last_seen_s, b.last_seen_s)
        total_hits = a.hits + b.hits
        merged_conf = max(a.conf_avg, b.conf_avg)
        merged_centroid = (a.centroid_xy * a.hits + b.centroid_xy * b.hits) / max(1, total_hits)

        return FireTrack(
            track_id=keeper_id,
            centroid_xy=merged_centroid.astype(np.float32),
            grid=new_grid,
            grid_origin_xy=(float(ux0), float(uy0)),
            cell_size_m=self.cell_size_m,
            last_seen_s=keeper_last_seen,
            hits=total_hits,
            conf_avg=float(merged_conf),
        )

    def _extract_polygons(self, track: FireTrack) -> List[List[Tuple[float, float]]]:
        mask = (track.grid >= self.activation_threshold).astype(np.uint8)
        if not mask.any():
            return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[List[Tuple[float, float]]] = []
        eps_px = max(1.0, self.simplify_eps_m / track.cell_size_m)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area_cells:
                continue
            simplified = cv2.approxPolyDP(contour, eps_px, closed=True)
            if len(simplified) < 3:
                continue
            poly: List[Tuple[float, float]] = []
            x0, y0 = track.grid_origin_xy
            for pt in simplified:
                col, row = float(pt[0][0]), float(pt[0][1])
                wx = x0 + (col + 0.5) * track.cell_size_m
                wy = y0 + (row + 0.5) * track.cell_size_m
                poly.append((wx, wy))
            out.append(poly)
        return out

    # -------------------------------------------------------- lon/lat & sanitize

    def _world_to_lonlat(self, xy: Tuple[float, float]) -> List[float]:
        x, y = xy
        lat = self.origin_lat + (y / self._meters_per_deg_lat)
        lon = self.origin_lon + (x / self._meters_per_deg_lon)
        return [float(lon), float(lat)]

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
