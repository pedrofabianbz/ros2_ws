#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

# Mismo tamaño que tu /map: 5m x 5m, 0.05 m/celda -> 100x100
RES = 0.05
H = 100
W = 100

# Estilo OccupancyGrid
OCC_FREE = 0
OCC_OCCUPIED = 100
OCC_UNKNOWN = -1

# Etiquetas para entrenamiento
LABEL_STATIC = 0
LABEL_DYNAMIC = 1
LABEL_IGNORE = -1


@dataclass
class MovingObject:
    x: float      # posición (m)
    y: float
    vx: float     # velocidad (m/s)
    vy: float
    radius: float # radio (m)

    def step(self, dt: float, xmin: float, ymin: float, xmax: float, ymax: float):
        """Actualiza posición y rebota en los bordes."""
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x - self.radius < xmin or self.x + self.radius > xmax:
            self.vx *= -1.0
            self.x = np.clip(self.x, xmin + self.radius, xmax - self.radius)

        if self.y - self.radius < ymin or self.y + self.radius > ymax:
            self.vy *= -1.0
            self.y = np.clip(self.y, ymin + self.radius, ymax - self.radius)


class World2D:
    def __init__(self):
        # Mundo 5x5 m aprox, centrado en (0,0)
        self.width = W
        self.height = H
        self.resolution = RES

        self.origin_x = - (W * RES) / 2.0  # esquina inferior izq en mundo
        self.origin_y = - (H * RES) / 2.0

        self.world_min_x = self.origin_x
        self.world_max_x = self.origin_x + W * RES
        self.world_min_y = self.origin_y
        self.world_max_y = self.origin_y + H * RES

        # Mapa estático: paredes y algunos obstáculos fijos
        self.static_grid = np.full((H, W), OCC_FREE, dtype=np.int8)
        self._create_static_map()

        # Objetos dinámicos
        self.moving_objects = []
        self._create_moving_objects()

    def _create_static_map(self):
        """Crea paredes tipo arena + unos bloques estáticos."""
        # Bordes
        self.static_grid[0, :] = OCC_OCCUPIED
        self.static_grid[-1, :] = OCC_OCCUPIED
        self.static_grid[:, 0] = OCC_OCCUPIED
        self.static_grid[:, -1] = OCC_OCCUPIED

        # Obstáculo vertical en el centro
        mid_x = W // 2
        self.static_grid[20:80, mid_x] = OCC_OCCUPIED

        # Obstáculo horizontal en el centro
        mid_y = H // 2
        self.static_grid[mid_y, 20:80] = OCC_OCCUPIED

    def _create_moving_objects(self):
        """Crea 2 objetos móviles que rebotan dentro del mundo."""
        self.moving_objects.append(
            MovingObject(
                x=0.5, y=0.0,
                vx=0.3, vy=0.0,
                radius=0.20
            )
        )
        self.moving_objects.append(
            MovingObject(
                x=-0.5, y=0.5,
                vx=0.0, vy=-0.25,
                radius=0.15
            )
        )

    def step(self, dt: float):
        """Avanza el mundo dt segundos y devuelve:
           - occupancy (H, W): 0 libre, 100 ocupado
           - labels   (H, W): 0 estático, 1 dinámico, -1 ignorar
        """
        # 1) actualizar objetos dinámicos
        for obj in self.moving_objects:
            obj.step(dt, self.world_min_x, self.world_min_y,
                     self.world_max_x, self.world_max_y)

        # 2) grid de dinámicos
        dyn_grid = np.zeros((H, W), dtype=np.int8)

        for obj in self.moving_objects:
            # bounding box del disco
            min_x = obj.x - obj.radius
            max_x = obj.x + obj.radius
            min_y = obj.y - obj.radius
            max_y = obj.y + obj.radius

            for i in range(H):
                cy = self.origin_y + (i + 0.5) * self.resolution
                if cy < min_y or cy > max_y:
                    continue
                for j in range(W):
                    cx = self.origin_x + (j + 0.5) * self.resolution
                    if cx < min_x or cx > max_x:
                        continue
                    if (cx - obj.x) ** 2 + (cy - obj.y) ** 2 <= obj.radius ** 2:
                        dyn_grid[i, j] = 1

        # 3) occupancy = estático + dinámico
        occupancy = np.array(self.static_grid, copy=True)
        occupancy[dyn_grid == 1] = OCC_OCCUPIED

        # 4) etiquetas
        labels = np.full((H, W), LABEL_IGNORE, dtype=np.int8)
        # dinámico
        labels[dyn_grid == 1] = LABEL_DYNAMIC
        # estático: ocupado en static_grid y no dinámico
        static_mask = (self.static_grid == OCC_OCCUPIED) & (dyn_grid == 0)
        labels[static_mask] = LABEL_STATIC

        return occupancy, labels


if __name__ == "__main__":
    world = World2D()
    dt = 0.1  # 10 Hz
    for t in range(50):
        occ, lab = world.step(dt)
        print(f"t={t}, ocupadas={np.sum(occ == OCC_OCCUPIED)}, dinamicas={np.sum(lab == LABEL_DYNAMIC)}")
