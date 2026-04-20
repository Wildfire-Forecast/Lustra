import cv2
import numpy as np


def draw_topdown_map(
    drone_pos,
    target_pos,
    saved_points,
    clicked_ground_point=None,
    yolo_ground_polygons=None,
    map_size_px=800,
    world_half_extent=80.0,
):
    canvas = np.ones((map_size_px, map_size_px, 3), dtype=np.uint8) * 245

    def world_to_pixel(x, y):
        px = int((x + world_half_extent) / (2 * world_half_extent) * map_size_px)
        py = int(map_size_px - ((y + world_half_extent) / (2 * world_half_extent) * map_size_px))
        return px, py

    for g in range(-80, 81, 20):
        x1, y1 = world_to_pixel(g, -world_half_extent)
        x2, y2 = world_to_pixel(g, world_half_extent)
        cv2.line(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

        x1, y1 = world_to_pixel(-world_half_extent, g)
        x2, y2 = world_to_pixel(world_half_extent, g)
        cv2.line(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

    x1, y1 = world_to_pixel(0, -world_half_extent)
    x2, y2 = world_to_pixel(0, world_half_extent)
    cv2.line(canvas, (x1, y1), (x2, y2), (150, 150, 150), 2)

    x1, y1 = world_to_pixel(-world_half_extent, 0)
    x2, y2 = world_to_pixel(world_half_extent, 0)
    cv2.line(canvas, (x1, y1), (x2, y2), (150, 150, 150), 2)

    dx, dy = world_to_pixel(drone_pos[0], drone_pos[1])
    cv2.circle(canvas, (dx, dy), 8, (255, 0, 0), -1)
    cv2.putText(canvas, "Drone", (dx + 10, dy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    tx, ty = world_to_pixel(target_pos[0], target_pos[1])
    cv2.circle(canvas, (tx, ty), 6, (0, 180, 0), -1)
    cv2.putText(canvas, "Target", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

    cv2.line(canvas, (dx, dy), (tx, ty), (0, 180, 0), 2)

    for i, pt in enumerate(saved_points):
        px, py = world_to_pixel(pt["world"][0], pt["world"][1])
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(canvas, f"P{i}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    if clicked_ground_point is not None:
        cxp, cyp = world_to_pixel(clicked_ground_point[0], clicked_ground_point[1])
        cv2.circle(canvas, (cxp, cyp), 8, (0, 140, 255), 2)
        cv2.putText(canvas, "Current", (cxp + 10, cyp + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    if yolo_ground_polygons is not None:
        for det_i, poly in enumerate(yolo_ground_polygons):
            pts_px = []

            for pt in poly:
                if pt is None:
                    continue
                px, py = world_to_pixel(pt[0], pt[1])
                pts_px.append((px, py))
                cv2.circle(canvas, (px, py), 4, (255, 0, 255), -1)

            if len(pts_px) >= 2:
                for i in range(len(pts_px)):
                    p1 = pts_px[i]
                    p2 = pts_px[(i + 1) % len(pts_px)]
                    cv2.line(canvas, p1, p2, (255, 0, 255), 2)

            if len(pts_px) > 0:
                cv2.putText(
                    canvas,
                    f"Y{det_i}",
                    (pts_px[0][0] + 6, pts_px[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )

    return canvas

