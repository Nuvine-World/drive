"""Shared 3D bounding-box record."""
from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    label: str
    score: float = 1.0

    def to_array(self):
        return np.array([self.x, self.y, self.z, self.length, self.width, self.height, self.yaw], dtype=np.float32)
