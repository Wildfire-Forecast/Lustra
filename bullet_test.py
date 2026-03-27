import pybullet as p
import pybullet_data
import time
import random
import os
import cv2
import inspect
import numpy as np
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
assets_dir = os.path.join(current_dir, "assets_new")

clicked_point = None
clicked_ground_point = None
clicked_depth_value = None
saved_ground_points = []

# image folder
save_path = os.path.join(current_dir, "captured_images")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- PyBullet setup ---
cid = p.connect(p.GUI, options="--disable-example-browser")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

p.setGravity(0, 0, -9.8)
data_path = pybullet_data.getDataPath()
p.setAdditionalSearchPath(assets_dir)

plane_id = p.loadURDF(os.path.join(data_path, "plane.urdf"))
p.changeVisualShape(plane_id, -1, rgbaColor=[1, 1, 1, 1])

# =========================
# Camera / Stereo Settings
# =========================
width, height = 640, 640
fov = 60
nearVal, farVal = 0.1, 120.0

# Start a bit lower and closer than before so stereo is more meaningful
cam_height = 18.0
cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
cam_target = np.array([8.0, 8.0, 0.0], dtype=np.float32)
base_eye_pos = np.array([20.0, 20.0, cam_height], dtype=np.float32)

# Slightly larger baseline than before to make disparity more visible
baseline_m = 0.30  # assuming 1 pybullet unit = 1 meter

# Intrinsics
cx = width / 2.0
cy = height / 2.0
fov_rad = np.deg2rad(fov)
fy = (height / 2.0) / np.tan(fov_rad / 2.0)
fx = fy * (width / height)

print("Intrinsics:", "fx=", fx, "fy=", fy, "cx=", cx, "cy=", cy)

# =========================
# YOLO model
# =========================
model_path = os.path.join(current_dir, "last.pt")   # change if needed
model = YOLO(model_path)

# optional: if your class name is fire, nice for display
CLASS_NAMES = model.names if hasattr(model, "names") else {0: "fire"}

CONF_THRES = 0.40


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def draw_topdown_map(
    drone_pos,
    target_pos,
    saved_points,
    clicked_ground_point=None,
    yolo_ground_polygons=None,
    map_size_px=800,
    world_half_extent=80.0
):
    """
    Draw a simple top-down local map using PyBullet world coordinates.
    x -> horizontal
    y -> vertical
    """
    canvas = np.ones((map_size_px, map_size_px, 3), dtype=np.uint8) * 245

    def world_to_pixel(x, y):
        px = int((x + world_half_extent) / (2 * world_half_extent) * map_size_px)
        py = int(map_size_px - ((y + world_half_extent) / (2 * world_half_extent) * map_size_px))
        return px, py

    # grid
    for g in range(-80, 81, 20):
        x1, y1 = world_to_pixel(g, -world_half_extent)
        x2, y2 = world_to_pixel(g, world_half_extent)
        cv2.line(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

        x1, y1 = world_to_pixel(-world_half_extent, g)
        x2, y2 = world_to_pixel(world_half_extent, g)
        cv2.line(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

    # axes
    x1, y1 = world_to_pixel(0, -world_half_extent)
    x2, y2 = world_to_pixel(0, world_half_extent)
    cv2.line(canvas, (x1, y1), (x2, y2), (150, 150, 150), 2)

    x1, y1 = world_to_pixel(-world_half_extent, 0)
    x2, y2 = world_to_pixel(world_half_extent, 0)
    cv2.line(canvas, (x1, y1), (x2, y2), (150, 150, 150), 2)

    # drone
    dx, dy = world_to_pixel(drone_pos[0], drone_pos[1])
    cv2.circle(canvas, (dx, dy), 8, (255, 0, 0), -1)
    cv2.putText(canvas, "Drone", (dx + 10, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # target/look point
    tx, ty = world_to_pixel(target_pos[0], target_pos[1])
    cv2.circle(canvas, (tx, ty), 6, (0, 180, 0), -1)
    cv2.putText(canvas, "Target", (tx + 10, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

    # line from drone to target
    cv2.line(canvas, (dx, dy), (tx, ty), (0, 180, 0), 2)

    # saved points
    for i, pt in enumerate(saved_points):
        px, py = world_to_pixel(pt["world"][0], pt["world"][1])
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(canvas, f"P{i}", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # current clicked point
    if clicked_ground_point is not None:
        cxp, cyp = world_to_pixel(clicked_ground_point[0], clicked_ground_point[1])
        cv2.circle(canvas, (cxp, cyp), 8, (0, 140, 255), 2)
        cv2.putText(canvas, "Current", (cxp + 10, cyp + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        
        # YOLO projected bbox corners on ground
    if yolo_ground_polygons is not None:
        for det_i, poly in enumerate(yolo_ground_polygons):
            pts_px = []

            for pt in poly:
                if pt is None:
                    continue
                px, py = world_to_pixel(pt[0], pt[1])
                pts_px.append((px, py))

                # draw each corner point
                cv2.circle(canvas, (px, py), 4, (255, 0, 255), -1)

            # if we have enough valid points, connect them
            if len(pts_px) >= 2:
                for i in range(len(pts_px)):
                    p1 = pts_px[i]
                    p2 = pts_px[(i + 1) % len(pts_px)]
                    cv2.line(canvas, p1, p2, (255, 0, 255), 2)

            # label polygon
            if len(pts_px) > 0:
                cv2.putText(
                    canvas,
                    f"Y{det_i}",
                    (pts_px[0][0] + 6, pts_px[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1
                )

    return canvas

def project_bbox_corners_to_ground(x1, y1, x2, y2, eye_pos, target_pos, up_vec):
    """
    Project the 4 image bbox corners onto the ground plane z=0.
    Returns a list of 4 points in world coordinates (or None where invalid).
    Order:
      top-left, top-right, bottom-right, bottom-left
    """
    corners_img = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x2, y2),  # bottom-right
        (x1, y2),  # bottom-left
    ]

    ground_pts = []

    for (u, v) in corners_img:
        ray_cam = pixel_to_camera_ray(u, v, fx, fy, cx, cy)
        ray_world = camera_ray_to_world(ray_cam, eye_pos, target_pos, up_vec)
        ground_pt = intersect_ray_with_ground(eye_pos, ray_world, ground_z=0.0)
        ground_pts.append(ground_pt)

    return ground_pts

def get_camera_basis():
    """
    Returns forward, right, up vectors of the camera.
    """
    forward = _normalize(cam_target - base_eye_pos)
    right = _normalize(np.cross(forward, cam_up))
    up = _normalize(np.cross(right, forward))
    return forward, right, up

def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked pixel: ({x}, {y})")

def get_stereo_eyes():
    """
    Stereo eyes offset along the camera right vector.
    """
    _, right, _ = get_camera_basis()
    left_eye = base_eye_pos - right * (baseline_m / 2.0)
    right_eye = base_eye_pos + right * (baseline_m / 2.0)
    return left_eye, right_eye


def get_camera_image(eye_pos):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye_pos.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=cam_up.tolist()
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=float(width) / float(height),
        nearVal=nearVal,
        farVal=farVal
    )

    (_, _, px, _, _) = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb_array = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))
    return cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)

def run_yolo_on_frame(frame_bgr, conf_thres=0.40):
    """
    Run YOLO on a BGR OpenCV image and return detections as dicts.
    """
    results = model.predict(
        source=frame_bgr,
        conf=conf_thres,
        verbose=False
    )

    detections = []

    if not results:
        return detections

    r = results[0]
    if r.boxes is None:
        return detections

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = box.astype(int)

        # clamp to image
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cx_box = int((x1 + x2) / 2)
        cy_box = int((y1 + y2) / 2)

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "conf": float(conf),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx_box,
            "cy": cy_box
        })

    return detections


# =========================
# Stereo matcher
# =========================
num_disp = 16 * 12   # must be divisible by 16
block_size = 7       # odd number

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size * block_size,
    P2=32 * block_size * block_size,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=150,
    speckleRange=1,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


def compute_depth_and_visuals(img_left, img_right):
    """
    Computes disparity + depth map and returns visualization images.
    """
    grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Tiny noise helps matching in textureless simulated regions
    noise = np.random.normal(0, 2, grayL.shape).astype(np.int16)
    grayL = np.clip(grayL.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    grayR = np.clip(grayR.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0.5] = np.nan

    depth_m = (fx * baseline_m) / disp
    valid_ratio = np.isfinite(depth_m).mean()

    # Disparity visualization
    disp_vis = np.nan_to_num(disp, nan=0.0)
    disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis_u8 = disp_vis.astype(np.uint8)

    # Depth visualization
    depth_vis = np.copy(depth_m)
    depth_vis[np.isnan(depth_vis)] = 80.0  # invalid as white/far
    max_show = 80.0
    depth_vis = np.clip(depth_vis, 0, max_show)
    depth_vis_u8 = (depth_vis / max_show * 255.0).astype(np.uint8)

    return disp, depth_m, valid_ratio, disp_vis_u8, depth_vis_u8


def patch_median_depth(depth_map, x, y, half_size=6):
    """
    More reliable than reading a single pixel.
    """
    x1 = max(0, x - half_size)
    x2 = min(depth_map.shape[1], x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(depth_map.shape[0], y + half_size + 1)

    patch = depth_map[y1:y2, x1:x2]
    vals = patch[np.isfinite(patch)]

    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))

def pixel_to_camera_ray(u, v, fx, fy, cx, cy):
    """
    Convert image pixel to a ray direction in camera coordinates.
    OpenCV-style camera frame:
      x = right
      y = down
      z = forward
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0], dtype=np.float32)
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    return ray_cam


def camera_ray_to_world(ray_cam, eye_pos, target_pos, up_vec):
    """
    Rotate a camera-frame ray into world coordinates.
    """
    forward = _normalize(target_pos - eye_pos)
    right = _normalize(np.cross(forward, up_vec))
    true_up = _normalize(np.cross(right, forward))

    # Camera-to-world rotation basis
    # ray_world = x*right + y*true_up? careful:
    # camera y points DOWN in image coords, but world up is +true_up
    # therefore use -true_up for camera y axis
    ray_world = (
        ray_cam[0] * right +
        ray_cam[1] * (-true_up) +
        ray_cam[2] * forward
    )
    ray_world = _normalize(ray_world)
    return ray_world







def intersect_ray_with_ground(ray_origin, ray_dir, ground_z=0.0):
    """
    Intersect world ray with horizontal ground plane z = ground_z.
    Returns None if no valid intersection.
    """
    if abs(ray_dir[2]) < 1e-6:
        return None

    t = (ground_z - ray_origin[2]) / ray_dir[2]

    # Must be in front of the camera
    if t <= 0:
        return None

    point = ray_origin + t * ray_dir
    return point

# =========================
# Object creation
# =========================
def create_fire_sphere(position=[0, 0, 5], radius=1.5):
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1, 0, 0, 1]
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=position
    )


def create_sphere_grid(start_pos, rows, cols, radius=1.5):
    spacing = radius * 2
    for i in range(rows):
        for j in range(cols):
            create_fire_sphere(
                position=[start_pos[0] + i * spacing, start_pos[1] + j * spacing, start_pos[2]],
                radius=radius
            )


def load_custom_object(urdf_filename, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    try:
        obj_id = p.loadURDF(
            urdf_filename,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=True
        )

        obj_folder = os.path.dirname(urdf_filename)
        tex_path = os.path.join(assets_dir, obj_folder, "leaf_pattern.png")

        if os.path.exists(tex_path):
            tex_id = p.loadTexture(tex_path)
            p.changeVisualShape(obj_id, 0, textureUniqueId=tex_id, rgbaColor=[1, 1, 1, 1])

        return obj_id

    except Exception as e:
        print(f"Error loading {urdf_filename}: {e}")
        return None


# --- random positioning ---
area_size = 80
placed_positions = []


def place_randomly(urdf_list, count, min_dist):
    placed = 0
    attempts = 0
    while placed < count and attempts < 1000:
        pos = [random.uniform(-area_size, area_size), random.uniform(-area_size, area_size), 0]
        if all(np.linalg.norm(np.array(pos) - np.array(p_pos)) > min_dist for p_pos in placed_positions):
            path = random.choice(urdf_list)
            rot = p.getQuaternionFromEuler([0, 0, random.uniform(0, 6.28)])
            load_custom_object(path, position=pos, orientation=rot)
            placed_positions.append(pos)
            placed += 1
        attempts += 1


# Scene
place_randomly(["pinusbruita/pinusbruita.urdf", "oak/oak.urdf", "tree/tree.urdf"], 80, 5.0)
create_sphere_grid(start_pos=[-10, -10, 2], rows=4, cols=3)
create_sphere_grid(start_pos=[10, 10, 2], rows=2, cols=3)
place_randomly(["smallrock/smallrock.urdf", "mediumrock/mediumrock.urdf", "bigrock/bigrock.urdf"], 80, 3.0)
place_randomly(["bush/bush.urdf"], 90, 2.5)

# Debug camera
p.resetDebugVisualizerCamera(
    cameraDistance=40,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=cam_target.tolist()
)

move_speed = 0.5
img_counter = 0
frame_i = 0
show_stereo = True

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

# =========================
# Main Loop
# =========================
while True:
    p.stepSimulation()
    keys = p.getKeyboardEvents()

    forward, right, _ = get_camera_basis()

    # movement
    if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
        base_eye_pos += forward * move_speed
        cam_target += forward * move_speed
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        base_eye_pos -= forward * move_speed
        cam_target -= forward * move_speed
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        base_eye_pos -= right * move_speed
        cam_target -= right * move_speed
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        base_eye_pos += right * move_speed
        cam_target += right * move_speed
    if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
        base_eye_pos[2] += move_speed
        cam_target[2] += move_speed
    if ord('f') in keys and keys[ord('f')] & p.KEY_IS_DOWN:
        base_eye_pos[2] -= move_speed
        cam_target[2] -= move_speed

    # toggle stereo windows
    if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
        show_stereo = not show_stereo

    # capture stereo pair
    left_eye, right_eye = get_stereo_eyes()
    img_left = get_camera_image(left_eye)
    img_right = get_camera_image(right_eye)

    # YOLO detections on left image
    detections = run_yolo_on_frame(img_left, conf_thres=CONF_THRES)
    
    # compute disparity / depth
    disp, depth_m, valid_ratio, disp_vis_u8, depth_vis_u8 = compute_depth_and_visuals(img_left, img_right)

    # debug depth from a few patches instead of one pixel
    if frame_i % 30 == 0:
        z_center = patch_median_depth(depth_m, int(cx), int(cy), half_size=8)
        z_test1 = patch_median_depth(depth_m, 320, 240, half_size=8)
        z_test2 = patch_median_depth(depth_m, 380, 280, half_size=8)

        print(
            "Depth center:", z_center,
            "| test1:", z_test1,
            "| test2:", z_test2,
            "| valid ratio:", round(valid_ratio, 3),
            "| eye:", np.round(base_eye_pos, 2),
            "| target:", np.round(cam_target, 2)
        )
        

    frame_i += 1

    # draw helper points on left image
    debug_left = img_left.copy()
    
    yolo_ground_polygons = []
    
    # =========================
    # YOLO draw + depth estimate
    # =========================
    for det_i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bx, by = det["cx"], det["cy"]
        conf = det["conf"]
        class_name = det["class_name"]
        
        # project bbox corners to ground for top-down map
        ground_poly = project_bbox_corners_to_ground(
            x1, y1, x2, y2,
            base_eye_pos,
            cam_target,
            cam_up
        )
        yolo_ground_polygons.append(ground_poly)

        # patch-based stereo depth at bbox center
        fire_depth = patch_median_depth(depth_m, bx, by, half_size=10)

        # distance label
        if np.isfinite(fire_depth):
            label = f"{class_name} {conf:.2f} | {fire_depth:.2f} m"
        else:
            label = f"{class_name} {conf:.2f} | depth=nan"

        # draw bbox
        cv2.rectangle(debug_left, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw center point
        cv2.circle(debug_left, (bx, by), 5, (0, 165, 255), -1)

        # text
        cv2.putText(
            debug_left,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # optional print every 30 frames
        if frame_i % 30 == 0:
            print(
                f"[YOLO] #{det_i} {class_name} conf={conf:.2f} "
                f"bbox=({x1},{y1},{x2},{y2}) center=({bx},{by}) "
                f"depth={fire_depth:.2f} m" if np.isfinite(fire_depth)
                else f"[YOLO] #{det_i} {class_name} conf={conf:.2f} "
                     f"bbox=({x1},{y1},{x2},{y2}) center=({bx},{by}) depth=nan"
            )
    
    if clicked_point is not None:
        px, py = clicked_point
        z_click = patch_median_depth(depth_m, px, py, half_size=8)
        clicked_depth_value = z_click

        ray_cam = pixel_to_camera_ray(px, py, fx, fy, cx, cy)
        ray_world = camera_ray_to_world(ray_cam, base_eye_pos, cam_target, cam_up)
        ground_pt = intersect_ray_with_ground(base_eye_pos, ray_world, ground_z=0.0)
        clicked_ground_point = ground_pt

        if frame_i % 5 == 0 and clicked_ground_point is not None:
            print(
                f"Clicked depth: {z_click:.2f} m" if np.isfinite(z_click) else "Clicked depth: nan",
                f"| Ground point: ({clicked_ground_point[0]:.2f}, {ground_pt[1]:.2f}, {ground_pt[2]:.2f})"
            )

        cv2.circle(debug_left, (px, py), 7, (0, 0, 255), -1)

        depth_label = f"{z_click:.2f} m" if np.isfinite(z_click) else "nan"
        cv2.putText(
            debug_left,
            depth_label,
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    if clicked_ground_point is not None and clicked_point is not None:
        px, py = clicked_point
        ground_label = f"({clicked_ground_point[0]:.2f}, {clicked_ground_point[1]:.2f})"
        cv2.putText(
            debug_left,
            ground_label,
            (px + 10, py + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
    )
            
    if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
        if clicked_ground_point is not None:
            saved_ground_points.append({
                "world": clicked_ground_point.copy(),
                "depth": clicked_depth_value if clicked_depth_value is not None else np.nan,
                "drone_eye": base_eye_pos.copy(),
                "drone_target": cam_target.copy()
            })
            print(
                f"Saved point #{len(saved_ground_points)-1}: "
                f"world=({clicked_ground_point[0]:.2f}, {clicked_ground_point[1]:.2f}, {clicked_ground_point[2]:.2f}), "
                f"depth={clicked_depth_value:.2f} m" if clicked_depth_value is not None and np.isfinite(clicked_depth_value)
                else f"Saved point #{len(saved_ground_points)-1}: world=({clicked_ground_point[0]:.2f}, {clicked_ground_point[1]:.2f}, {clicked_ground_point[2]:.2f}), depth=nan"
            )
        else:
            print("No clicked ground point to save.")
        
        
    cv2.circle(debug_left, (int(cx), int(cy)), 5, (0, 255, 255), -1)
    cv2.circle(debug_left, (320, 240), 5, (255, 0, 255), -1)
    cv2.circle(debug_left, (380, 280), 5, (255, 255, 0), -1)

    cv2.imshow("Left Eye (Reference)", debug_left)
    cv2.setMouseCallback("Left Eye (Reference)", on_mouse)
    cv2.imshow("Right Eye (Shifted)", img_right)
    
    topdown = draw_topdown_map(
        drone_pos=base_eye_pos,
        target_pos=cam_target,
        saved_points=saved_ground_points,
        clicked_ground_point=clicked_ground_point,
        yolo_ground_polygons=yolo_ground_polygons,
        map_size_px=800,
        world_half_extent=80.0
    )

    cv2.imshow("Top-Down Map", topdown)

    if show_stereo:
        cv2.imshow("Disparity", disp_vis_u8)
        cv2.imshow("Depth (0-80m, invalid=white)", depth_vis_u8)

    # save images
    if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
        l_filename = os.path.join(save_path, f"rect_left_{img_counter}.png")
        r_filename = os.path.join(save_path, f"rect_right_{img_counter}.png")
        d_filename = os.path.join(save_path, f"depth_vis_{img_counter}.png")
        s_filename = os.path.join(save_path, f"disp_vis_{img_counter}.png")

        cv2.imwrite(l_filename, debug_left)
        cv2.imwrite(r_filename, img_right)
        cv2.imwrite(d_filename, depth_vis_u8)
        cv2.imwrite(s_filename, disp_vis_u8)

        print(f"!!! SUCCESS !!! Saved set {img_counter}")
        print("Left :", l_filename)
        print("Right:", r_filename)
        print("Depth:", d_filename)
        print("Disp :", s_filename)

        cv2.setWindowTitle("Left Eye (Reference)", "SAVED! - SAVED! - SAVED!")
        img_counter += 1
    else:
        cv2.setWindowTitle("Left Eye (Reference)", "Left Eye (Reference)")

    # quit
    if (ord('q') in keys) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    time.sleep(1 / 240)

p.disconnect()
cv2.destroyAllWindows()