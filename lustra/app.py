import os
import threading
import time
import csv
from collections import deque

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from .config import get_project_paths
from .visualization import draw_topdown_map
from .visualization.world_map import WorldMapConfig, WorldMapUI
from .vision.camera import get_camera_image, get_parallel_stereo_views
from .vision.fire_tracker import FireTracker
from .vision.geometry import camera_ray_to_world, get_camera_basis, intersect_ray_with_ground, pixel_to_camera_ray
from .vision.stereo import StereoProcessor
from .world import WorldBuilder

os.environ["KMP_WARNINGS"] = "0"


class LustraApp:
    def __init__(self, verbose=False, drone_height_m=18.0):
        print("[startup] Initializing Lustra app...", flush=True)
        self.verbose = bool(verbose)
        self.drone_height_m = float(drone_height_m)
        self.left_window_name = "Left Eye (Reference)"
        self.default_window_name = "Lustra (Default View)"
        self.paths = get_project_paths()
        os.makedirs(self.paths.captured_images_dir, exist_ok=True)

        self.clicked_point = None
        self.pending_click_point = None
        self.clicked_ground_point = None
        self.clicked_depth_value = None
        self.clicked_range_value = None
        self.clicked_error_band_value = np.nan
        self.saved_ground_points = []

        self.render_width, self.render_height = 640, 640
        self.fov = 60
        self.near_val, self.far_val = 0.1, 120.0

        self.cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.cam_target = np.array([8.0, 8.0, 0.0], dtype=np.float32)
        self.base_eye_pos = np.array([20.0, 20.0, self.drone_height_m], dtype=np.float32)
        self.baseline_m = 0.30

        fov_rad = np.deg2rad(self.fov)
        self.fy = (self.render_height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = self.fy * (self.render_width / self.render_height)

        self.detector = None
        self._detector_thread = None
        self._detector_started = False
        self._detector_ready = False
        self._detector_error = None
        self._detector_wait_printed = False

        self.stereo_processor = StereoProcessor(self.fx, self.baseline_m)
        self.num_disp = self.stereo_processor.num_disp

        self.width = self.render_width - self.num_disp
        self.height = self.render_height
        self.cx = self.render_width / 2.0 - self.num_disp
        self.cy = self.render_height / 2.0

        self.conf_threshold = 0.40

        self.move_speed = 4
        self.img_counter = self._next_image_counter()
        self.frame_i = 0
        self.show_stereo = True
        self.fire_body_id = None
        self.clicked_abs_errors_m = deque(maxlen=5000)
        self.clicked_rel_errors_pct = deque(maxlen=5000)
        self.depth_compare_csv = os.path.join(self.paths.captured_images_dir, "clicked_depth_comparisons.csv")
        self.last_clicked_comparison = None
        self.current_left_eye = self.base_eye_pos.copy()
        self.current_left_target = self.cam_target.copy()
        self.current_left_depth_m = None
        self.current_left_seg_mask = None
        self._default_move_key = None
        self._default_move_ttl_s = 0.18
        self._default_move_last_seen = 0.0
        self.controls_panel = self.make_controls_panel()
        self.fire_map_path = os.path.join(self.paths.root_dir, "fire_map.html")
        self.map_origin_lat = 39.0
        self.map_origin_lon = 35.0
        self.fire_map_config = WorldMapConfig(center_lat=self.map_origin_lat, center_lon=self.map_origin_lon, zoom_start=10)
        self.fire_tracker = FireTracker(
            origin_lat=self.map_origin_lat,
            origin_lon=self.map_origin_lon,
            distance_threshold_m=6.0,
            ttl_s=4.0,
            min_hits=2,
        )
        self._fire_map_interval_s = 1.0
        self._last_fire_map_write = 0.0
        self._fire_snapshot = []
        self._init_depth_compare_csv()

    def _next_image_counter(self) -> int:
        import re
        existing = [
            f for f in os.listdir(self.paths.captured_images_dir)
            if re.match(r"rect_left_(\d+)\.png", f)
        ]
        if not existing:
            return 0
        indices = [int(re.match(r"rect_left_(\d+)\.png", f).group(1)) for f in existing]
        return max(indices) + 1

    def make_controls_panel(self):
        panel = np.full((260, 420, 3), 18, dtype=np.uint8)
        lines = [
            "Controls",
            "W/X: forward/back",
            "A/D: left/right",
            "R/F: up/down",
            "C: save ground point",
            "T: save images",
            "G: stereo windows (verbose)",
            "Q: quit",
        ]
        y = 34
        for i, line in enumerate(lines):
            scale = 0.75 if i == 0 else 0.58
            color = (220, 230, 255) if i == 0 else (225, 225, 225)
            thickness = 2 if i == 0 else 1
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += 34 if i == 0 else 28
        return panel

    def _show_clean_window(self, name, image):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(name, image)

    def _show_default_window(self, image):
        cv2.namedWindow(self.default_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.default_window_name, image)

    def _load_detector_worker(self):
        try:
            print("[startup] Importing YOLO runtime...", flush=True)
            from .vision.detection import YoloDetector

            print(f"[startup] Loading YOLO model: {self.paths.model_path}", flush=True)
            self.detector = YoloDetector(self.paths.model_path, default_confidence=0.40)
            self._detector_ready = True
            print("[startup] YOLO ready.", flush=True)
        except Exception as exc:
            self._detector_error = exc
            print(f"[startup] YOLO failed to load: {exc}", flush=True)

    def _start_detector_loading(self):
        if self._detector_started:
            return
        self._detector_started = True
        self._detector_thread = threading.Thread(target=self._load_detector_worker, daemon=True)
        self._detector_thread.start()

    def setup_simulation(self):
        print("[startup] Connecting to PyBullet GUI...", flush=True)
        p.connect(p.GUI, options="--disable-example-browser")
        p.setInternalSimFlags(0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        p.setGravity(0, 0, -9.8)
        pybullet_data.getDataPath()
        p.setAdditionalSearchPath(self.paths.assets_dir)

        print("[startup] Building world (this can take a bit)...", flush=True)
        wb = WorldBuilder(self.paths.assets_dir)
        wb.setup_base_world()
        wb.build_biome_world(tile_size=4, grid_range=25)
        ## fire spawner
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) ##better performance
        self.fire_body_id = wb.spawn_fire(center_pos=[25,25, 1], grid_size=7, max_radius=0.5, max_scale=20)
        self.fire_body_id = wb.spawn_fire(center_pos=[70,-10, 1], grid_size=8, max_radius=0.5, max_scale=15)
        self.fire_body_id = wb.spawn_fire(center_pos=[-50,-25, 1], grid_size=6, max_radius=0.5, max_scale=25)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.resetDebugVisualizerCamera(
            cameraDistance=40,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=self.cam_target.tolist(),
        )

    def print_controls(self):
        print("Intrinsics:", "fx=", self.fx, "fy=", self.fy, "cx=", self.cx, "cy=", self.cy)
        print("Mode:", "VERBOSE (-v)" if self.verbose else "DEFAULT")
        print(f"Drone height (m): {self.drone_height_m:.2f}")
        print("------ Drone Controls -----")
        print("Press 'w' to move forwards.")
        print("Press 'x' to move backwards.")
        print("Press 'a' to move left.")
        print("Press 'd' to move right.")
        print("Press 'r' to move up.")
        print("Press 'f' to move down.")
        print("------ Stereo / Save / Quit -----")
        print("Press 'g' to toggle stereo windows.")
        print("Press 't' to capture images.")
        print("Press 'q' to quit.")
        print("Press 'c' to save current clicked ground point.")
        print(f"Clicked-point depth comparisons are logged to: {self.depth_compare_csv}")

    def _init_depth_compare_csv(self):
        if os.path.exists(self.depth_compare_csv) and os.path.getsize(self.depth_compare_csv) > 0:
            return
        with open(self.depth_compare_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "timestamp_unix",
                    "frame",
                    "pixel_x",
                    "pixel_y",
                    "estimated_range_m",
                    "true_range_m",
                    "abs_error_m",
                    "rel_error_pct",
                    "eye_x",
                    "eye_y",
                    "eye_z",
                    "hit_x",
                    "hit_y",
                    "hit_z",
                    "hit_body_id",
                ]
            )

    def compute_range_from_depth(self, depth_z, px, py):
        if not np.isfinite(depth_z):
            return np.nan
        ray_cam = pixel_to_camera_ray(px, py, self.fx, self.fy, self.cx, self.cy)
        if ray_cam[2] <= 1e-6:
            return np.nan
        return float(depth_z / ray_cam[2])

    def get_true_range_at_pixel(self, px, py):
        if self.current_left_depth_m is None or self.current_left_seg_mask is None:
            return np.nan, None, -1
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return np.nan, None, -1

        depth_z = float(self.current_left_depth_m[py, px])
        if not np.isfinite(depth_z) or depth_z <= 0.0:
            return np.nan, None, -1

        ray_cam = pixel_to_camera_ray(px, py, self.fx, self.fy, self.cx, self.cy)
        if ray_cam[2] <= 1e-6:
            return np.nan, None, -1
        true_range_m = float(depth_z / ray_cam[2])

        ray_world = camera_ray_to_world(ray_cam, self.current_left_eye, self.current_left_target, self.cam_up)
        hit_pos = self.current_left_eye + ray_world * true_range_m

        seg_val = int(self.current_left_seg_mask[py, px])
        if seg_val < 0:
            hit_body_id = -1
        else:
            hit_body_id = seg_val & ((1 << 24) - 1)
        return true_range_m, hit_pos, hit_body_id

    def build_clicked_comparison(self, pred_range_m, eye_pos, frame_i, px, py):
        if not np.isfinite(pred_range_m):
            return None
        true_range_m, hit_pos, hit_body_id = self.get_true_range_at_pixel(px, py)
        if not np.isfinite(true_range_m):
            return None
        abs_error_m = abs(pred_range_m - true_range_m)
        rel_error_pct = (abs_error_m / max(true_range_m, 1e-6)) * 100.0
        self.clicked_abs_errors_m.append(abs_error_m)
        self.clicked_rel_errors_pct.append(rel_error_pct)
        return {
            "timestamp_unix": time.time(),
            "frame": frame_i,
            "pixel_x": int(px),
            "pixel_y": int(py),
            "estimated_range_m": float(pred_range_m),
            "true_range_m": true_range_m,
            "abs_error_m": abs_error_m,
            "rel_error_pct": float(rel_error_pct),
            "eye_x": float(eye_pos[0]),
            "eye_y": float(eye_pos[1]),
            "eye_z": float(eye_pos[2]),
            "hit_x": float(hit_pos[0]),
            "hit_y": float(hit_pos[1]),
            "hit_z": float(hit_pos[2]),
            "hit_body_id": hit_body_id,
        }

    def append_clicked_comparison_to_csv(self, row):
        with open(self.depth_compare_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    f"{row['timestamp_unix']:.6f}",
                    row["frame"],
                    row["pixel_x"],
                    row["pixel_y"],
                    f"{row['estimated_range_m']:.6f}",
                    f"{row['true_range_m']:.6f}",
                    f"{row['abs_error_m']:.6f}",
                    f"{row['rel_error_pct']:.4f}",
                    f"{row['eye_x']:.6f}",
                    f"{row['eye_y']:.6f}",
                    f"{row['eye_z']:.6f}",
                    f"{row['hit_x']:.6f}",
                    f"{row['hit_y']:.6f}",
                    f"{row['hit_z']:.6f}",
                    row["hit_body_id"],
                ]
            )

    def get_clicked_error_band(self, percentile=95):
        if len(self.clicked_abs_errors_m) < 20:
            return np.nan
        return float(np.percentile(np.array(self.clicked_abs_errors_m, dtype=np.float32), percentile))

    def make_depth_comparison_panel(self):
        panel_h, panel_w = 430, 640
        panel = np.full((panel_h, panel_w, 3), 22, dtype=np.uint8)
        cv2.putText(panel, "Depth Comparison", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 255), 2)

        if self.last_clicked_comparison is None:
            cv2.putText(panel, "Click on Left Eye to start depth comparison...", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            return panel

        c = self.last_clicked_comparison
        abs_errors = np.array(self.clicked_abs_errors_m, dtype=np.float32)
        mean_err = float(np.mean(abs_errors)) if len(abs_errors) else np.nan
        med_err = float(np.median(abs_errors)) if len(abs_errors) else np.nan
        p95_err = self.get_clicked_error_band(95)

        stats_lines = [
            f"Pixel: ({c['pixel_x']}, {c['pixel_y']}) | Hit body: {c['hit_body_id']}) | Drone height: {self.drone_height_m:.2f} m",
            f"Estimated range: {c['estimated_range_m']:.2f} m",
            f"True range:      {c['true_range_m']:.2f} m",
            f"Abs error:       {c['abs_error_m']:.2f} m",
            f"Rel error:       {c['rel_error_pct']:.1f} %",
            f"Samples: {len(self.clicked_abs_errors_m)} | Mean: {mean_err:.2f} | Median: {med_err:.2f} | P95: {p95_err:.2f}",
        ]
        y = 72
        for line in stats_lines:
            cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
            y += 26

        def draw_trend(series, x0, y0, w, h, title, unit, color):
            cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
            tail = np.array(series, dtype=np.float32)[-200:]
            if len(tail) < 2:
                cv2.putText(panel, f"{title}: not enough samples", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
                return

            y_max = max(1.0, float(np.percentile(tail, 95)) * 1.2, float(np.max(tail)))
            prev = None
            for i, value in enumerate(tail):
                x = x0 + int((i / (len(tail) - 1)) * (w - 1))
                y_plot = y0 + h - 1 - int((min(float(value), y_max) / y_max) * (h - 1))
                point = (x, y_plot)
                if prev is not None:
                    cv2.line(panel, prev, point, color, 2)
                prev = point
            cv2.putText(
                panel,
                f"{title} (last {len(tail)}), y-max {y_max:.1f} {unit}",
                (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

        x0 = 20
        y0 = y + 8
        w = panel_w - 40
        chart_gap = 12
        available_h = max(120, panel_h - y0 - 18)
        chart_h = max(55, int((available_h - chart_gap) / 2))
        y1 = y0 + chart_h + chart_gap

        draw_trend(self.clicked_abs_errors_m, x0, y0, w, chart_h, "Abs error trend", "m", (40, 210, 80))
        draw_trend(self.clicked_rel_errors_pct, x0, y1, w, chart_h, "Rel error trend", "%", (70, 180, 255))

        return panel

    def on_mouse(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_click_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")

    def on_mouse_default_view(self, event, x, y, flags, param):
        del param
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < self.width and 0 <= y < self.height:
            self.on_mouse(event, x, y, flags, None)

    def make_default_view_panel(self, debug_left, topdown_map, fire_snapshot):
        depth_panel = self.make_depth_comparison_panel()
        left_h, left_w = debug_left.shape[:2]
        right_w = left_w
        v_spacer = np.full((left_h, 8, 3), 35, dtype=np.uint8)
        h_spacer = np.full((8, right_w, 3), 35, dtype=np.uint8)

        top_h = int((left_h - 8) * 0.40)
        bottom_h = left_h - 8 - top_h

        top_block = np.full((top_h, right_w, 3), 18, dtype=np.uint8)
        pad = 10
        map_w = int(right_w * 0.52)
        map_h = max(80, top_h - (2 * pad))
        topdown_resized = cv2.resize(topdown_map, (map_w, map_h), interpolation=cv2.INTER_LINEAR)
        top_block[pad : pad + map_h, pad : pad + map_w] = topdown_resized

        controls = [
            "Controls",
            "W/X: forward/back",
            "A/D: left/right",
            "R/F: up/down",
            "C: save ground point",
            "T: save images",
            "Q: quit",
        ]
        text_y = 24
        for i, line in enumerate(controls):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.56 if i == 0 else 0.45
            color = (220, 230, 255) if i == 0 else (220, 220, 220)
            thickness = 2 if i == 0 else 1
            (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
            tx = max(map_w + 2 * pad, right_w - pad - tw)
            cv2.putText(top_block, line, (tx, text_y), font, scale, color, thickness)
            text_y += th + (8 if i == 0 else 7)

        tracker_x0 = max(map_w + 2 * pad, right_w - 260)
        tracker_y0 = min(text_y + 6, top_h - 24)
        tracker_w = right_w - tracker_x0 - pad
        tracker_h = max(60, top_h - tracker_y0 - pad)
        cv2.rectangle(top_block, (tracker_x0, tracker_y0), (tracker_x0 + tracker_w, tracker_y0 + tracker_h), (90, 90, 90), 1)
        cv2.putText(top_block, "Fire Tracker", (tracker_x0 + 8, tracker_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 255), 1)

        line_y = tracker_y0 + 40
        if not fire_snapshot:
            cv2.putText(top_block, "No active fires", (tracker_x0 + 8, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        else:
            for entry in fire_snapshot[:6]:
                track_id = entry.get("track_id")
                hits = entry.get("hits")
                conf = entry.get("confidence")
                age = entry.get("age_s")
                label = f"#{track_id} hits={hits} conf={conf:.2f} age={age:.1f}s"
                cv2.putText(top_block, label, (tracker_x0 + 8, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                line_y += 18
                if line_y > tracker_y0 + tracker_h - 8:
                    break

        depth_resized = cv2.resize(depth_panel, (right_w, bottom_h), interpolation=cv2.INTER_LINEAR)
        right_column = np.vstack([top_block, h_spacer, depth_resized])

        return np.hstack([debug_left, v_spacer, right_column])

    def project_bbox_corners_to_ground(self, x1, y1, x2, y2):
        corners_img = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        ground_pts = []
        eye_pos = self.current_left_eye
        target_pos = self.current_left_target
        for (u, v) in corners_img:
            ray_cam = pixel_to_camera_ray(u, v, self.fx, self.fy, self.cx, self.cy)
            ray_world = camera_ray_to_world(ray_cam, eye_pos, target_pos, self.cam_up)
            ground_pt = intersect_ray_with_ground(eye_pos, ray_world, ground_z=0.0)
            ground_pts.append(ground_pt)
        return ground_pts

    @staticmethod
    def is_fire_detection(class_name: str) -> bool:
        return "fire" in str(class_name).lower()

    def estimate_range_from_bbox_edges(self, depth_m, x1, y1, x2, y2):
        edge_points = [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
            ((x1 + x2) // 2, y1),
            ((x1 + x2) // 2, y2),
            (x1, (y1 + y2) // 2),
            (x2, (y1 + y2) // 2),
        ]
        ranges = []
        for px, py in edge_points:
            z = self.stereo_processor.patch_median_depth(depth_m, int(px), int(py), half_size=6)
            r = self.compute_range_from_depth(z, int(px), int(py))
            if np.isfinite(r):
                ranges.append(r)
        if not ranges:
            return np.nan
        return float(np.median(np.array(ranges, dtype=np.float32)))

    def update_fire_map(self, fire_polygons, timestamp_s):
        self.fire_tracker.update(fire_polygons, timestamp_s)
        self._fire_snapshot = self.fire_tracker.get_active_tracks(timestamp_s, include_unconfirmed=True)
        if timestamp_s - self._last_fire_map_write < self._fire_map_interval_s:
            return

        drone_lon, drone_lat = self.fire_tracker.world_xy_to_lonlat((self.base_eye_pos[0], self.base_eye_pos[1]))
        self.fire_map_config.center_lat = drone_lat
        self.fire_map_config.center_lon = drone_lon

        geojson = self.fire_tracker.to_geojson(timestamp_s)
        map_ui = WorldMapUI(self.fire_map_config)
        map_ui.add_markers(
            [
                {
                    "lat": drone_lat,
                    "lon": drone_lon
                }
            ],
            name="Drone",
            icon_color="blue",
        )
        if geojson.get("features"):
            map_ui.add_fire_zones_geojson(
                geojson,
                style={"color": "#ff0000", "weight": 2, "fillOpacity": 0.4},
            )
        map_ui.save_html(self.fire_map_path)
        self._last_fire_map_write = timestamp_s

    def process_click(self, debug_left, depth_m):
        if self.pending_click_point is not None:
            self.clicked_point = self.pending_click_point
            self.pending_click_point = None
            px, py = self.clicked_point
            z_click = self.stereo_processor.patch_median_depth(depth_m, px, py, half_size=8)
            self.clicked_depth_value = z_click
            click_range_m = self.compute_range_from_depth(z_click, px, py)
            self.clicked_range_value = click_range_m
            comparison = self.build_clicked_comparison(click_range_m, self.current_left_eye, self.frame_i, px, py)
            if comparison is not None:
                self.last_clicked_comparison = comparison
                self.append_clicked_comparison_to_csv(comparison)
            self.clicked_error_band_value = self.get_clicked_error_band(percentile=95)

            ray_cam = pixel_to_camera_ray(px, py, self.fx, self.fy, self.cx, self.cy)
            ray_world = camera_ray_to_world(ray_cam, self.current_left_eye, self.current_left_target, self.cam_up)
            ground_pt = intersect_ray_with_ground(self.current_left_eye, ray_world, ground_z=0.0)
            self.clicked_ground_point = ground_pt

            if self.clicked_ground_point is not None:
                print(
                    f"Clicked depth: {z_click:.2f} m" if np.isfinite(z_click) else "Clicked depth: nan",
                    f"| Ground point: ({self.clicked_ground_point[0]:.2f}, {ground_pt[1]:.2f}, {ground_pt[2]:.2f})",
                )

        if self.clicked_point is None:
            return

        px, py = self.clicked_point

        cv2.circle(debug_left, (px, py), 7, (0, 0, 255), -1)
        if np.isfinite(self.clicked_range_value):
            if np.isfinite(self.clicked_error_band_value):
                depth_label = f"{self.clicked_range_value:.2f} m ±{self.clicked_error_band_value:.1f}"
            else:
                depth_label = f"{self.clicked_range_value:.2f} m"
        else:
            depth_label = "nan"
        cv2.putText(debug_left, depth_label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.clicked_ground_point is not None:
            ground_label = f"({self.clicked_ground_point[0]:.2f}, {self.clicked_ground_point[1]:.2f})"
            cv2.putText(
                debug_left,
                ground_label,
                (px + 10, py + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    def handle_movement(self, keys):
        _, right, _ = get_camera_basis(self.base_eye_pos, self.cam_target, self.cam_up)
        raw_forward = self.cam_target - self.base_eye_pos
        forward_horizontal = np.array([raw_forward[0], raw_forward[1], 0])
        if np.linalg.norm(forward_horizontal) > 0:
            forward_horizontal /= np.linalg.norm(forward_horizontal)

        if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
            self.base_eye_pos += forward_horizontal * self.move_speed
            self.cam_target += forward_horizontal * self.move_speed
        if ord("x") in keys and keys[ord("x")] & p.KEY_IS_DOWN:
            self.base_eye_pos -= forward_horizontal * self.move_speed
            self.cam_target -= forward_horizontal * self.move_speed
        if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
            self.base_eye_pos -= right * self.move_speed
            self.cam_target -= right * self.move_speed
        if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
            self.base_eye_pos += right * self.move_speed
            self.cam_target += right * self.move_speed
        if ord("r") in keys and keys[ord("r")] & p.KEY_IS_DOWN:
            self.base_eye_pos[2] += self.move_speed
            self.cam_target[2] += self.move_speed
        if ord("f") in keys and keys[ord("f")] & p.KEY_IS_DOWN:
            self.base_eye_pos[2] -= self.move_speed
            self.cam_target[2] -= self.move_speed
        self.drone_height_m = float(self.base_eye_pos[2])

        _, right, _ = get_camera_basis(self.base_eye_pos, self.cam_target, self.cam_up)
        raw_forward = self.cam_target - self.base_eye_pos
        forward_horizontal = np.array([raw_forward[0], raw_forward[1], 0])
        if np.linalg.norm(forward_horizontal) > 0:
            forward_horizontal /= np.linalg.norm(forward_horizontal)

        if self._default_move_key == ord("w"):
            self.base_eye_pos += forward_horizontal * self.move_speed
            self.cam_target += forward_horizontal * self.move_speed
        elif self._default_move_key == ord("x"):
            self.base_eye_pos -= forward_horizontal * self.move_speed
            self.cam_target -= forward_horizontal * self.move_speed
        elif self._default_move_key == ord("a"):
            self.base_eye_pos -= right * self.move_speed
            self.cam_target -= right * self.move_speed
        elif self._default_move_key == ord("d"):
            self.base_eye_pos += right * self.move_speed
            self.cam_target += right * self.move_speed
        elif self._default_move_key == ord("r"):
            self.base_eye_pos[2] += self.move_speed
            self.cam_target[2] += self.move_speed
        elif self._default_move_key == ord("f"):
            self.base_eye_pos[2] -= self.move_speed
            self.cam_target[2] -= self.move_speed
        self.drone_height_m = float(self.base_eye_pos[2])

    def handle_save(self, keys, debug_left, img_right, depth_vis_u8, disp_vis_u8):
        active_window = self.left_window_name if self.verbose else self.default_window_name
        if ord("t") not in keys or not (keys[ord("t")] & p.KEY_WAS_TRIGGERED):
            cv2.setWindowTitle(active_window, active_window)
            return

        l_filename = os.path.join(self.paths.captured_images_dir, f"rect_left_{self.img_counter}.png")
        r_filename = os.path.join(self.paths.captured_images_dir, f"rect_right_{self.img_counter}.png")
        d_filename = os.path.join(self.paths.captured_images_dir, f"depth_vis_{self.img_counter}.png")
        s_filename = os.path.join(self.paths.captured_images_dir, f"disp_vis_{self.img_counter}.png")

        cv2.imwrite(l_filename, debug_left)
        cv2.imwrite(r_filename, img_right)
        cv2.imwrite(d_filename, depth_vis_u8)
        cv2.imwrite(s_filename, disp_vis_u8)

        print(f"!!! SUCCESS !!! Saved set {self.img_counter}")
        print("Left :", l_filename)
        print("Right:", r_filename)
        print("Depth:", d_filename)
        print("Disp :", s_filename)
        cv2.setWindowTitle(active_window, "SAVED! - SAVED! - SAVED!")
        self.img_counter += 1

    def run(self):
        print("[startup] Starting simulation...", flush=True)
        self.setup_simulation()
        print("[startup] World ready.", flush=True)
        self.print_controls()
        self._start_detector_loading()

        while True:
            key_pressed = cv2.waitKey(1) & 0xFF
            del key_pressed
            p.stepSimulation()
            keys = p.getKeyboardEvents()

            self.handle_movement(keys)

            if ord("g") in keys and keys[ord("g")] & p.KEY_WAS_TRIGGERED:
                self.show_stereo = not self.show_stereo

            left_eye, right_eye, left_target, right_target = get_parallel_stereo_views(
                self.base_eye_pos, self.cam_target, self.cam_up, self.baseline_m
            )
            self.current_left_eye = left_eye
            self.current_left_target = left_target
            img_left_full, left_depth_full, left_seg_full = get_camera_image(
                left_eye,
                left_target,
                self.cam_up,
                self.fov,
                self.render_width,
                self.render_height,
                self.near_val,
                self.far_val,
                return_depth_and_seg=True,
            )
            img_right = get_camera_image(
                right_eye,
                right_target,
                self.cam_up,
                self.fov,
                self.render_width,
                self.render_height,
                self.near_val,
                self.far_val,
            )

            disp_full, depth_m_full, _, disp_vis_u8_full, depth_vis_u8_full = self.stereo_processor.compute_depth_and_visuals(img_left_full, img_right)

            crop = self.num_disp
            img_left = img_left_full[:, crop:]
            left_depth_m = left_depth_full[:, crop:]
            left_seg_mask = left_seg_full[:, crop:]
            disp = disp_full[:, crop:]
            depth_m = depth_m_full[:, crop:]
            disp_vis_u8 = disp_vis_u8_full[:, crop:]
            depth_vis_u8 = depth_vis_u8_full[:, crop:]
            valid_ratio = float(np.isfinite(depth_m).mean())
            self.current_left_depth_m = left_depth_m
            self.current_left_seg_mask = left_seg_mask

            if self.frame_i % 30 == 0:
                z_center = self.stereo_processor.patch_median_depth(depth_m, int(self.cx), int(self.cy), half_size=8)
                z_test1 = self.stereo_processor.patch_median_depth(depth_m, 320, 240, half_size=8)
                z_test2 = self.stereo_processor.patch_median_depth(depth_m, 380, 280, half_size=8)
                print(
                    "Depth center:",
                    z_center,
                    "| test1:",
                    z_test1,
                    "| test2:",
                    z_test2,
                    "| valid ratio:",
                    round(valid_ratio, 3),
                    "| eye:",
                    np.round(self.base_eye_pos, 2),
                    "| target:",
                    np.round(self.cam_target, 2),
                )

            self.frame_i += 1
            debug_left = img_left.copy()
            self.process_click(debug_left, depth_m)

            if self._detector_ready:
                detections = self.detector.detect(img_left, self.width, self.height, conf_thres=self.conf_threshold)
            else:
                detections = []
                if self._detector_error is not None and not self._detector_wait_printed:
                    print("[startup] Continuing without YOLO detections.", flush=True)
                    self._detector_wait_printed = True
                elif self._detector_error is None and not self._detector_wait_printed:
                    print("[startup] Waiting for YOLO to finish loading...", flush=True)
                    self._detector_wait_printed = True
            yolo_ground_polygons = []
            fire_polygons = []

            for det_i, det in enumerate(detections):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                bx, by = det["cx"], det["cy"]
                conf = det["conf"]
                class_name = det["class_name"]

                ground_poly = self.project_bbox_corners_to_ground(x1, y1, x2, y2)
                yolo_ground_polygons.append(ground_poly)

                fire_range_m = self.estimate_range_from_bbox_edges(depth_m, x1, y1, x2, y2)
                if np.isfinite(fire_range_m):
                    label = f"{class_name} {conf:.2f} | {fire_range_m:.2f} m"
                else:
                    label = f"{class_name} {conf:.2f} | range=nan"

                cv2.rectangle(debug_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(debug_left, (bx, by), 5, (0, 165, 255), -1)
                cv2.putText(
                    debug_left,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if self.frame_i % 30 == 0:
                    print(
                        f"[YOLO] #{det_i} {class_name} conf={conf:.2f} "
                        f"bbox=({x1},{y1},{x2},{y2}) center=({bx},{by}) "
                        f"range={fire_range_m:.2f} m"
                        if np.isfinite(fire_range_m)
                        else f"[YOLO] #{det_i} {class_name} conf={conf:.2f} "
                        f"bbox=({x1},{y1},{x2},{y2}) center=({bx},{by}) range=nan"
                    )

                if self.is_fire_detection(class_name):
                    poly_xy = [(pt[0], pt[1]) for pt in ground_poly if pt is not None]
                    if len(poly_xy) >= 3:
                        fire_polygons.append(
                            {
                                "polygon_xy": poly_xy,
                                "confidence": float(conf),
                            }
                        )

            if ord("c") in keys and keys[ord("c")] & p.KEY_WAS_TRIGGERED:
                if self.clicked_ground_point is not None:
                    self.saved_ground_points.append(
                        {
                            "world": self.clicked_ground_point.copy(),
                            "depth": self.clicked_depth_value if self.clicked_depth_value is not None else np.nan,
                            "drone_eye": self.base_eye_pos.copy(),
                            "drone_target": self.cam_target.copy(),
                        }
                    )
                    print(
                        f"Saved point #{len(self.saved_ground_points)-1}: "
                        f"world=({self.clicked_ground_point[0]:.2f}, {self.clicked_ground_point[1]:.2f}, {self.clicked_ground_point[2]:.2f}), "
                        f"depth={self.clicked_depth_value:.2f} m"
                        if self.clicked_depth_value is not None and np.isfinite(self.clicked_depth_value)
                        else f"Saved point #{len(self.saved_ground_points)-1}: world=({self.clicked_ground_point[0]:.2f}, {self.clicked_ground_point[1]:.2f}, {self.clicked_ground_point[2]:.2f}), depth=nan"
                    )
                else:
                    print("No clicked ground point to save.")

            cv2.circle(debug_left, (int(self.cx), int(self.cy)), 5, (0, 255, 255), -1)
            cv2.circle(debug_left, (320, 240), 5, (255, 0, 255), -1)
            cv2.circle(debug_left, (380, 280), 5, (255, 255, 0), -1)

            topdown = draw_topdown_map(
                drone_pos=self.base_eye_pos,
                target_pos=self.cam_target,
                saved_points=self.saved_ground_points,
                clicked_ground_point=self.clicked_ground_point,
                yolo_ground_polygons=yolo_ground_polygons,
                map_size_px=800,
                world_half_extent=80.0,
            )

            self.update_fire_map(fire_polygons, time.time())

            if self.verbose:
                self._show_clean_window(self.left_window_name, debug_left)
                cv2.setMouseCallback(self.left_window_name, self.on_mouse)
                self._show_clean_window("Right Eye (Shifted)", img_right)
                self._show_clean_window("Top-Down Map", topdown)
                self._show_clean_window("Depth Comparison", self.make_depth_comparison_panel())
            else:
                default_panel = self.make_default_view_panel(debug_left, topdown, self._fire_snapshot)
                self._show_default_window(default_panel)
                cv2.setMouseCallback(self.default_window_name, self.on_mouse_default_view)

            if self.verbose and self.show_stereo:
                self._show_clean_window("Disparity", disp_vis_u8)
                self._show_clean_window("Depth (0-80m, invalid=white)", depth_vis_u8)

            self.handle_save(keys, debug_left, img_right, depth_vis_u8, disp_vis_u8)

            if ord("q") in keys:
                break

            time.sleep(1 / 240)

        p.disconnect()
        cv2.destroyAllWindows()
