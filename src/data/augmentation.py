"""Scene-consistent training augmentation for globally aligned multi-node LiDAR.

Every node's cloud and every ground-truth box receive the same flip, rotation,
and scale about the centre of the configured region, so cooperative geometry
between nodes is preserved. Parameters are drawn on the training thread from
the checkpointed random generator; workers only apply them.
"""

import math
import numpy as np
from src.data.coinfra import in_bounds
from src.data.loader import BoundingBox3D


def sample_parameters(rng, settings):
    if not settings["enabled"]:
        return None
    degrees = settings["rotation_degrees"]
    low, high = settings["scale_range"]
    return {
        "flip_x": settings["flip_x"] and rng.random() < 0.5,
        "flip_y": settings["flip_y"] and rng.random() < 0.5,
        "rotation": math.radians(rng.uniform(-degrees, degrees)),
        "scale": rng.uniform(low, high),
    }


def augment_scene(clouds, boxes, parameters, cfg):
    """Return transformed copies of ``clouds`` and ``boxes`` cropped to bounds."""
    bounds = np.asarray(cfg["bounds"], dtype=float)
    centre_x, centre_y = (bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2
    cosine, sine = math.cos(parameters["rotation"]), math.sin(parameters["rotation"])
    scale = parameters["scale"]

    def plane(x, y):
        x, y = x - centre_x, y - centre_y
        if parameters["flip_x"]:
            x = -x
        if parameters["flip_y"]:
            y = -y
        x, y = cosine * x - sine * y, sine * x + cosine * y
        return x * scale + centre_x, y * scale + centre_y

    transformed = []
    for cloud in clouds:
        points = cloud.copy()
        points[:, 0], points[:, 1] = plane(
            cloud[:, 0].astype(np.float64), cloud[:, 1].astype(np.float64)
        )
        points[:, 2] *= scale
        inside = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(
            axis=1
        )
        transformed.append(points[inside])
    moved = []
    for box in boxes:
        x, y = plane(box.x, box.y)
        z = box.z * scale
        if not in_bounds(x, y, z, cfg["bounds"]):
            continue
        yaw = box.yaw
        if parameters["flip_x"]:
            yaw = math.pi - yaw
        if parameters["flip_y"]:
            yaw = -yaw
        yaw += parameters["rotation"]
        moved.append(
            BoundingBox3D(
                x,
                y,
                z,
                box.length * scale,
                box.width * scale,
                box.height * scale,
                math.atan2(math.sin(yaw), math.cos(yaw)),
                box.label,
                box.score,
            )
        )
    return transformed, moved
