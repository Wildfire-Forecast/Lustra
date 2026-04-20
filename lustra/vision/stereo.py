import cv2
import numpy as np


class StereoProcessor:
    def __init__(self, fx, baseline_m):
        self.fx = fx
        self.baseline_m = baseline_m
        self.num_disp = 16 * 12
        self.block_size = 7
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * self.block_size * self.block_size,
            P2=32 * self.block_size * self.block_size,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=150,
            speckleRange=1,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_depth_and_visuals(self, img_left, img_right):
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        noise = np.random.normal(0, 2, gray_left.shape).astype(np.int16)
        gray_left = np.clip(gray_left.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        gray_right = np.clip(gray_right.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        disp = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disp[disp <= 0.5] = np.nan

        depth_m = (self.fx * self.baseline_m) / disp
        valid_ratio = np.isfinite(depth_m).mean()

        disp_vis = np.nan_to_num(disp, nan=0.0)
        disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis_u8 = disp_vis.astype(np.uint8)

        depth_vis = np.copy(depth_m)
        depth_vis[np.isnan(depth_vis)] = 80.0
        max_show = 80.0
        depth_vis = np.clip(depth_vis, 0, max_show)
        depth_vis_u8 = (depth_vis / max_show * 255.0).astype(np.uint8)

        return disp, depth_m, valid_ratio, disp_vis_u8, depth_vis_u8

    @staticmethod
    def patch_median_depth(depth_map, x, y, half_size=6):
        x1 = max(0, x - half_size)
        x2 = min(depth_map.shape[1], x + half_size + 1)
        y1 = max(0, y - half_size)
        y2 = min(depth_map.shape[0], y + half_size + 1)

        patch = depth_map[y1:y2, x1:x2]
        vals = patch[np.isfinite(patch)]

        if len(vals) == 0:
            return np.nan
        return float(np.median(vals))

