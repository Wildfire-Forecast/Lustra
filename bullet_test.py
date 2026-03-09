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

#image folder
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

# camera variables
cam_height = 30.0
width, height = 640, 640
fov = 60
nearVal, farVal = 0.1, 100.0

cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
base_eye_pos = np.array([20.0, 20.0, cam_height], dtype=np.float32)

baseline_m = 0.06  # meters, assuming 1 bullet unit = 1 meter

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def get_stereo_eyes():
    forward = _normalize(cam_target - base_eye_pos)   # camera forward direction
    right = _normalize(np.cross(forward, cam_up))     # camera right direction
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
        fov=fov, aspect=float(width)/float(height), nearVal=nearVal, farVal=farVal
    )

    (_, _, px, _, _) = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb_array = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))
    return cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)

# Object creating
def create_fire_sphere(position=[0, 0, 5], radius=1.5):
    visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=position)

def create_sphere_grid(start_pos, rows, cols, radius=1.5):
    spacing = radius * 2 
    for i in range(rows):
        for j in range(cols):
            create_fire_sphere(position=[start_pos[0] + i*spacing, start_pos[1] + j*spacing, start_pos[2]], radius=radius)

def load_custom_object(urdf_filename, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    try:
        obj_id = p.loadURDF(urdf_filename, basePosition=position, 
                            baseOrientation=orientation, useFixedBase=True)
        
        #custom texture for leaves
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

#putting trees,rocks,bushes and fire
place_randomly(["pinusbruita/pinusbruita.urdf", "oak/oak.urdf", "tree/tree.urdf"], 80, 5.0)
create_sphere_grid(start_pos=[-10, -10, 2], rows=4, cols=3)
create_sphere_grid(start_pos=[10, 10, 2], rows=2, cols=3)
place_randomly(["smallrock/smallrock.urdf", "mediumrock/mediumrock.urdf", "bigrock/bigrock.urdf"], 80, 3.0)
place_randomly(["bush/bush.urdf"], 90, 2.5)


# camera settings
p.resetDebugVisualizerCamera(
    cameraDistance=40, cameraYaw=45, cameraPitch=-35, cameraTargetPosition=cam_target.tolist()
)

# --- Main Loop ---
img_counter = 0
print("------ Controls -----")
print("Press 's' to toggle shaders.")
print("Press 't' to capture images.")
print("Press 'q' to quit.")

while True:
    p.stepSimulation()
    
    #for saving images or closing
    keys = p.getKeyboardEvents()
    
    #both camera captures images
    left_eye, right_eye = get_stereo_eyes()
    img_left  = get_camera_image(left_eye)
    img_right = get_camera_image(right_eye)
    
    cv2.imshow("Left Eye (Reference)", img_left)
    cv2.imshow("Right Eye (Shifted)", img_right)

    # input check for 's' key 
    if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
        l_filename = os.path.join(save_path, f"rect_l_{img_counter}.png")
        r_filename = os.path.join(save_path, f"rect_r_{img_counter}.png")
        
        cv2.imwrite(l_filename, img_left)
        cv2.imwrite(r_filename, img_right)
        
        print(f"!!! SUCCESS !!! Saved pair {img_counter}")
        print(f"Path: {l_filename}")
        
        # feedback about image
        cv2.setWindowTitle("Left Eye (Reference)", "SAVED! - SAVED! - SAVED!")
        img_counter += 1
    else:
        cv2.setWindowTitle("Left Eye (Reference)", "Left Eye (Reference)")

    # 4. Check for 'Q' or OpenCV quit
    if (ord('q') in keys) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break
        
    time.sleep(1/240)

p.disconnect()
cv2.destroyAllWindows()