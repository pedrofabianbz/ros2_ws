import numpy as np
from .sdf_loader import load_static_rects_from_sdf

class World2D:
    def __init__(self, sdf_path: str | None = None):
        if sdf_path:
            self.static_rects = load_static_rects_from_sdf(sdf_path).static_rects
        else:
            self.static_rects = []

        # dinámicos (los sigues añadiendo aparte)
        self.dynamic_circles = [
            # [x, y, r, vx, vy]
        ]

    def step(self, dt: float):
        for c in self.dynamic_circles:
            c[0] += c[3] * dt
            c[1] += c[4] * dt

    def nearest_dynamic_distance_to_robot(self, x, y, robot_radius):
        dmin = np.inf
        for (cx, cy, r, *_ ) in self.dynamic_circles:
            d = np.hypot(x - cx, y - cy) - (robot_radius + r)
            dmin = min(dmin, d)
        if dmin == np.inf:
            return np.inf
        return max(dmin, 0.0)
