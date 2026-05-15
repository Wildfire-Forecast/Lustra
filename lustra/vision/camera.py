import cv2
import numpy as np
import pybullet as p

from .geometry import get_camera_basis


def get_parallel_stereo_views(base_eye_pos, cam_target, cam_up, baseline_m):
    forward, right, _ = get_camera_basis(base_eye_pos, cam_target, cam_up)
    left_eye = base_eye_pos - right * (baseline_m / 2.0)
    right_eye = base_eye_pos + right * (baseline_m / 2.0)
    view_dist = np.linalg.norm(cam_target - base_eye_pos)
    left_target = left_eye + forward * view_dist
    right_target = right_eye + forward * view_dist
    return left_eye, right_eye, left_target, right_target


def get_camera_image(eye_pos, cam_target, cam_up, fov, width, height, near_val, far_val, return_depth_and_seg=False):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye_pos.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=cam_up.tolist(),
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=float(width) / float(height),
        nearVal=near_val,
        farVal=far_val,
    )

    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if return_depth_and_seg else p.ER_NO_SEGMENTATION_MASK
    (_, _, px, depth_buffer, seg_mask) = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=flags,
    )

    rgb_array = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))
    bgr = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)
    if not return_depth_and_seg:
        return bgr

    depth_buffer = np.reshape(np.array(depth_buffer, dtype=np.float32), (height, width))
    depth_m = (far_val * near_val) / (far_val - (far_val - near_val) * depth_buffer)
    seg_mask = np.reshape(np.array(seg_mask, dtype=np.int32), (height, width))
    return bgr, depth_m.astype(np.float32), seg_mask
