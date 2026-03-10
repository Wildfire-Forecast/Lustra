import pybullet as p
import pybullet_data
import time
import random
import os
import cv2
import inspect
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
assets_dir = os.path.join(current_dir, "assets_new")

clicked_point = None

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
p.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.5, 0.2, 1])

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


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


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
    
    if clicked_point is not None:
        px, py = clicked_point
        z_click = patch_median_depth(depth_m, px, py, half_size=8)

        cv2.circle(debug_left, (px, py), 7, (0, 0, 255), -1)
        cv2.putText(
            debug_left,
            f"{z_click:.2f} m" if np.isfinite(z_click) else "nan",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        
        
    cv2.circle(debug_left, (int(cx), int(cy)), 5, (0, 255, 255), -1)
    cv2.circle(debug_left, (320, 240), 5, (255, 0, 255), -1)
    cv2.circle(debug_left, (380, 280), 5, (255, 255, 0), -1)

    cv2.imshow("Left Eye (Reference)", debug_left)
    cv2.setMouseCallback("Left Eye (Reference)", on_mouse)
    cv2.imshow("Right Eye (Shifted)", img_right)

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