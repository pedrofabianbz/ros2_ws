# sim/core/render_utils.py
import numpy as np
import matplotlib.pyplot as plt


def _collect_bounds(world, robot_pose, robot_radius, goal=None):
    xs = []
    ys = []

    # static rects
    for (xmin, ymin, xmax, ymax) in getattr(world, "static_rects", []):
        xs += [xmin, xmax]
        ys += [ymin, ymax]

    # dynamic circles
    for c in getattr(world, "dynamic_circles", []):
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        xs += [cx - r, cx + r]
        ys += [cy - r, cy + r]

    # robot
    x, y, _ = robot_pose
    xs += [x - robot_radius, x + robot_radius]
    ys += [y - robot_radius, y + robot_radius]

    # goal
    if goal is not None:
        gx, gy = float(goal[0]), float(goal[1])
        xs += [gx, gx]
        ys += [gy, gy]

    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)

    return (min(xs), max(xs), min(ys), max(ys))


def render_world(ax, world, robot_pose, robot_radius, goal=None, auto_view=True, pad=0.8):
    """
    Dibuja mundo + robot + goal.
    Si auto_view=True, ajusta xlim/ylim automáticamente al contenido (mejor para maze/pasillos).
    """
    ax.clear()

    # Obstáculos estáticos
    for (xmin, ymin, xmax, ymax) in getattr(world, "static_rects", []):
        w = xmax - xmin
        h = ymax - ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h, fill=False))

    # Obstáculos dinámicos
    for c in getattr(world, "dynamic_circles", []):
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        ax.add_patch(plt.Circle((cx, cy), r, fill=False))

    # Robot
    x, y, theta = robot_pose
    ax.add_patch(plt.Circle((x, y), robot_radius, fill=False))
    ax.plot(
        [x, x + robot_radius * np.cos(theta)],
        [y, y + robot_radius * np.sin(theta)]
    )

    # Goal
    if goal is not None:
        ax.plot(float(goal[0]), float(goal[1]), marker="x")

    ax.set_aspect("equal", "box")
    ax.grid(True)

    if auto_view:
        xmin, xmax, ymin, ymax = _collect_bounds(world, robot_pose, robot_radius, goal=goal)
        # padding
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad

        # evita ventana degenerada
        if (xmax - xmin) < 1.0:
            cx = 0.5 * (xmin + xmax)
            xmin, xmax = cx - 0.5, cx + 0.5
        if (ymax - ymin) < 1.0:
            cy = 0.5 * (ymin + ymax)
            ymin, ymax = cy - 0.5, cy + 0.5

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
