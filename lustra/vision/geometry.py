import numpy as np


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def get_camera_basis(base_eye_pos, cam_target, cam_up):
    forward = normalize(cam_target - base_eye_pos)
    right = normalize(np.cross(forward, cam_up))
    up = normalize(np.cross(right, forward))
    return forward, right, up


def pixel_to_camera_ray(u, v, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0], dtype=np.float32)
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    return ray_cam


def camera_ray_to_world(ray_cam, eye_pos, target_pos, up_vec):
    forward = normalize(target_pos - eye_pos)
    right = normalize(np.cross(forward, up_vec))
    true_up = normalize(np.cross(right, forward))
    ray_world = (
        ray_cam[0] * right
        + ray_cam[1] * (-true_up)
        + ray_cam[2] * forward
    )
    return normalize(ray_world)


def intersect_ray_with_ground(ray_origin, ray_dir, ground_z=0.0):
    if abs(ray_dir[2]) < 1e-6:
        return None

    t = (ground_z - ray_origin[2]) / ray_dir[2]
    if t <= 0:
        return None

    return ray_origin + t * ray_dir

