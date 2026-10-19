"""Point support of ground-truth boxes for a given set of LiDAR nodes."""

import math
import numpy as np


def box_point_counts(points, boxes, bounds, cell=2.0):
    """Number of ``points`` inside each rotated 3D box in ``boxes``.

    A coarse grid of the boxes' bounding circles discards most of the scene
    first, so only points near a box take the exact rotated test.
    """
    counts = np.zeros(len(boxes), dtype=np.int64)
    if not len(boxes) or not len(points):
        return counts
    values = np.array(
        [[b.x, b.y, b.z, b.length, b.width, b.height, b.yaw] for b in boxes],
        dtype=float,
    )
    radius = 0.5 * np.hypot(values[:, 3], values[:, 4])
    width = max(1, math.ceil((bounds[3] - bounds[0]) / cell))
    height = max(1, math.ceil((bounds[4] - bounds[1]) / cell))
    near = np.zeros((height, width), dtype=bool)
    low_x = np.floor((values[:, 0] - radius - bounds[0]) / cell).astype(int)
    high_x = np.floor((values[:, 0] + radius - bounds[0]) / cell).astype(int)
    low_y = np.floor((values[:, 1] - radius - bounds[1]) / cell).astype(int)
    high_y = np.floor((values[:, 1] + radius - bounds[1]) / cell).astype(int)
    for x0, x1, y0, y1 in zip(low_x, high_x, low_y, high_y):
        if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
            continue
        near[
            max(y0, 0) : min(y1, height - 1) + 1, max(x0, 0) : min(x1, width - 1) + 1
        ] = True
    px = np.floor((points[:, 0] - bounds[0]) / cell).astype(int)
    py = np.floor((points[:, 1] - bounds[1]) / cell).astype(int)
    inside_grid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    candidates = np.zeros(len(points), dtype=bool)
    candidates[inside_grid] = near[py[inside_grid], px[inside_grid]]
    selected = points[candidates, :3].astype(np.float64)
    if not len(selected):
        return counts
    for index, (x, y, z, length, box_width, box_height, yaw) in enumerate(values):
        dx, dy = selected[:, 0] - x, selected[:, 1] - y
        cosine, sine = math.cos(yaw), math.sin(yaw)
        along = dx * cosine + dy * sine
        across = -dx * sine + dy * cosine
        counts[index] = int(
            np.count_nonzero(
                (np.abs(along) <= length / 2)
                & (np.abs(across) <= box_width / 2)
                & (np.abs(selected[:, 2] - z) <= box_height / 2)
            )
        )
    return counts


def node_origin(node):
    """Sensor position in the shared frame, when the node has a calibration."""
    if "calibration" not in node:
        return None
    from src.data.coinfra import calibration

    return calibration(node["calibration"])[:3, 3]
