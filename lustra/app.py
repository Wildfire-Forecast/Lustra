import os
import threading
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from .config import get_project_paths
from .visualization import draw_topdown_map
from .vision.camera import get_camera_image, get_stereo_eyes
from .vision.geometry import camera_ray_to_world, get_camera_basis, intersect_ray_with_ground, pixel_to_camera_ray
from .vision.stereo import StereoProcessor
from .world import WorldBuilder

os.environ["KMP_WARNINGS"] = "0"


class LustraApp:
    def __init__(self):
        print("[startup] Initializing Lustra app...", flush=True)
        self.paths = get_project_paths()
        os.makedirs(self.paths.captured_images_dir, exist_ok=True)

        self.clicked_point = None
        self.clicked_ground_point = None
        self.clicked_depth_value = None
        self.saved_ground_points = []

        self.width, self.height = 640, 640
        self.fov = 60
        self.near_val, self.far_val = 0.1, 120.0

        cam_height = 18.0
        self.cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.cam_target = np.array([8.0, 8.0, 0.0], dtype=np.float32)
        self.base_eye_pos = np.array([20.0, 20.0, cam_height], dtype=np.float32)
        self.baseline_m = 0.30

        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        fov_rad = np.deg2rad(self.fov)
        self.fy = (self.height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = self.fy * (self.width / self.height)

        self.detector = None
        self._detector_thread = None
        self._detector_started = False
        self._detector_ready = False
        self._detector_error = None
        self._detector_wait_printed = False

        self.stereo_processor = StereoProcessor(self.fx, self.baseline_m)
        self.conf_threshold = 0.40

        self.move_speed = 0.5
        self.img_counter = 0
        self.frame_i = 0
        self.show_stereo = True

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
        wb.create_fire_sphere(position=[5, 5, 1])

        p.resetDebugVisualizerCamera(
            cameraDistance=40,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=self.cam_target.tolist(),
        )

    def print_controls(self):
        print("Intrinsics:", "fx=", self.fx, "fy=", self.fy, "cx=", self.cx, "cy=", self.cy)
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

    def on_mouse(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked pixel: ({x}, {y})")

    def project_bbox_corners_to_ground(self, x1, y1, x2, y2):
        corners_img = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        ground_pts = []
        for (u, v) in corners_img:
            ray_cam = pixel_to_camera_ray(u, v, self.fx, self.fy, self.cx, self.cy)
            ray_world = camera_ray_to_world(ray_cam, self.base_eye_pos, self.cam_target, self.cam_up)
            ground_pt = intersect_ray_with_ground(self.base_eye_pos, ray_world, ground_z=0.0)
            ground_pts.append(ground_pt)
        return ground_pts

    def process_click(self, debug_left, depth_m):
        if self.clicked_point is None:
            return

        px, py = self.clicked_point
        z_click = self.stereo_processor.patch_median_depth(depth_m, px, py, half_size=8)
        self.clicked_depth_value = z_click

        ray_cam = pixel_to_camera_ray(px, py, self.fx, self.fy, self.cx, self.cy)
        ray_world = camera_ray_to_world(ray_cam, self.base_eye_pos, self.cam_target, self.cam_up)
        ground_pt = intersect_ray_with_ground(self.base_eye_pos, ray_world, ground_z=0.0)
        self.clicked_ground_point = ground_pt

        if self.frame_i % 5 == 0 and self.clicked_ground_point is not None:
            print(
                f"Clicked depth: {z_click:.2f} m" if np.isfinite(z_click) else "Clicked depth: nan",
                f"| Ground point: ({self.clicked_ground_point[0]:.2f}, {ground_pt[1]:.2f}, {ground_pt[2]:.2f})",
            )

        cv2.circle(debug_left, (px, py), 7, (0, 0, 255), -1)
        depth_label = f"{z_click:.2f} m" if np.isfinite(z_click) else "nan"
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

    def handle_save(self, keys, debug_left, img_right, depth_vis_u8, disp_vis_u8):
        if ord("t") not in keys or not (keys[ord("t")] & p.KEY_WAS_TRIGGERED):
            cv2.setWindowTitle("Left Eye (Reference)", "Left Eye (Reference)")
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
        cv2.setWindowTitle("Left Eye (Reference)", "SAVED! - SAVED! - SAVED!")
        self.img_counter += 1

    def run(self):
        print("[startup] Starting simulation...", flush=True)
        self.setup_simulation()
        print("[startup] World ready.", flush=True)
        self.print_controls()
        self._start_detector_loading()

        while True:
            p.stepSimulation()
            keys = p.getKeyboardEvents()

            self.handle_movement(keys)

            if ord("g") in keys and keys[ord("g")] & p.KEY_WAS_TRIGGERED:
                self.show_stereo = not self.show_stereo

            left_eye, right_eye = get_stereo_eyes(self.base_eye_pos, self.cam_target, self.cam_up, self.baseline_m)
            img_left = get_camera_image(
                left_eye,
                self.cam_target,
                self.cam_up,
                self.fov,
                self.width,
                self.height,
                self.near_val,
                self.far_val,
            )
            img_right = get_camera_image(
                right_eye,
                self.cam_target,
                self.cam_up,
                self.fov,
                self.width,
                self.height,
                self.near_val,
                self.far_val,
            )

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
            disp, depth_m, valid_ratio, disp_vis_u8, depth_vis_u8 = self.stereo_processor.compute_depth_and_visuals(img_left, img_right)

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
            yolo_ground_polygons = []

            for det_i, det in enumerate(detections):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                bx, by = det["cx"], det["cy"]
                conf = det["conf"]
                class_name = det["class_name"]

                ground_poly = self.project_bbox_corners_to_ground(x1, y1, x2, y2)
                yolo_ground_polygons.append(ground_poly)

                fire_depth = self.stereo_processor.patch_median_depth(depth_m, bx, by, half_size=10)
                if np.isfinite(fire_depth):
                    label = f"{class_name} {conf:.2f} | {fire_depth:.2f} m"
                else:
                    label = f"{class_name} {conf:.2f} | depth=nan"

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
                        f"depth={fire_depth:.2f} m"
                        if np.isfinite(fire_depth)
                        else f"[YOLO] #{det_i} {class_name} conf={conf:.2f} "
                        f"bbox=({x1},{y1},{x2},{y2}) center=({bx},{by}) depth=nan"
                    )

            self.process_click(debug_left, depth_m)

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

            cv2.imshow("Left Eye (Reference)", debug_left)
            cv2.setMouseCallback("Left Eye (Reference)", self.on_mouse)
            cv2.imshow("Right Eye (Shifted)", img_right)

            topdown = draw_topdown_map(
                drone_pos=self.base_eye_pos,
                target_pos=self.cam_target,
                saved_points=self.saved_ground_points,
                clicked_ground_point=self.clicked_ground_point,
                yolo_ground_polygons=yolo_ground_polygons,
                map_size_px=800,
                world_half_extent=80.0,
            )
            cv2.imshow("Top-Down Map", topdown)

            if self.show_stereo:
                cv2.imshow("Disparity", disp_vis_u8)
                cv2.imshow("Depth (0-80m, invalid=white)", depth_vis_u8)

            self.handle_save(keys, debug_left, img_right, depth_vis_u8, disp_vis_u8)

            if (ord("q") in keys) or (cv2.waitKey(1) & 0xFF == ord("q")):
                break

            time.sleep(1 / 240)

        p.disconnect()
        cv2.destroyAllWindows()
