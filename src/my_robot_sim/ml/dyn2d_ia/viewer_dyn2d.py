#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --------- Cargar dataset ---------
DATA_PATH = "dataset_dyn2d_corridor.npz"

data = np.load(DATA_PATH)
grids = data["grids"]
states = data["states"]    # [xr, yr, xg, yg, xo, yo, vox, voy]
actions = data["actions"]  # [vx, vy]

GRID_SIZE = grids.shape[1]
WORLD_SIZE = 5.0    # porque 100 * 0.05
CELL = WORLD_SIZE / GRID_SIZE

# índice global accesible
idx = 0


# --------- Utilidades ---------
def world_to_pix(x, y):
    j = x / CELL
    i = y / CELL
    return i, j


# --------- Dibujar frame ---------
def draw_example(idx):
    plt.clf()

    grid = grids[idx]
    xr, yr, xg, yg, xo, yo, vox, voy = states[idx]
    vx, vy = actions[idx]

    plt.imshow(grid, cmap="gray", origin="lower")

    # Robot circular
    ri, rj = world_to_pix(xr, yr)
    plt.scatter(rj, ri, c="blue", s=80, label="robot")

    # Goal circular
    gi, gj = world_to_pix(xg, yg)
    plt.scatter(gj, gi, c="green", s=80, label="goal")

    # Objeto dinámico cuadrado (0.5 × 0.5 m)
    oi, oj = world_to_pix(xo, yo)
    half_side_pix = 0.25 / CELL
    square = patches.Rectangle(
        (oj - half_side_pix, oi - half_side_pix),
        2 * half_side_pix, 2 * half_side_pix,
        linewidth=1, edgecolor='red', facecolor='none'
    )
    plt.gca().add_patch(square)

    # Acción recomendada (flecha amarilla)
    plt.arrow(rj, ri, vx * 3, vy * 3, color="yellow", width=0.1)

    plt.title(f"Example {idx}/{len(grids)-1}")
    plt.legend(loc="upper right")
    plt.pause(0.001)


# --------- Evento del teclado ---------
def on_key(event):
    global idx

    if event.key == "right":
        idx = (idx + 1) % len(grids)
    elif event.key == "left":
        idx = (idx - 1) % len(grids)

    draw_example(idx)


# --------- Main ---------
def main():
    print("Controles:")
    print("  →  Siguiente ejemplo")
    print("  ←  Ejemplo anterior")

    fig = plt.figure(figsize=(6,6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    draw_example(idx)
    plt.show()


if __name__ == "__main__":
    main()
