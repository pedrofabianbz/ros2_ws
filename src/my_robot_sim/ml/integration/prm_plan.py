# ml/integration/prm_plan.py
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx

from sim.core.world import World2D

Point2 = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass
class PRMConfig:
    n_samples: int = 800
    k_neighbors: int = 12
    max_sample_tries: int = 50_000
    edge_step: float = 0.05          # discretización aristas
    clearance: float = 0.20          # margen robot vs obstáculos (m)
    world_limit: float = 10.0        # si tu World2D no lo expone, usamos esto
    seed: int = 123


# ----------- Geometría / colisiones -----------
def _dist_point_to_segment(px, py, ax, ay, bx, by) -> float:
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _iter_static_rects(world: World2D) -> Iterable[Rect]:
    """Tu World2D guarda obstáculos como rectángulos AABB en world.static_rects."""
    rects = getattr(world, "static_rects", None)
    if rects is None:
        return
        yield  # type: ignore[misc]
    for r in rects:
        xmin, ymin, xmax, ymax = r
        yield float(xmin), float(ymin), float(xmax), float(ymax)


def _iter_static_segments(world: World2D) -> Iterable[Tuple[float, float, float, float]]:
    """
    Extrae segmentos estáticos (si existen) y además deriva segmentos desde static_rects.
    """
    # 1) Si el world ya tiene segmentos explícitos
    for name in ("static_segments", "segments", "walls"):
        segs = getattr(world, name, None)
        if segs is not None:
            for s in segs:
                if len(s) == 4 and all(isinstance(v, (int, float)) for v in s):
                    yield float(s[0]), float(s[1]), float(s[2]), float(s[3])
                elif len(s) == 2:
                    (x1, y1), (x2, y2) = s
                    yield float(x1), float(y1), float(x2), float(y2)
            break

    # 2) Derivar bordes desde rectángulos AABB
    for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
        # abajo
        yield xmin, ymin, xmax, ymin
        # derecha
        yield xmax, ymin, xmax, ymax
        # arriba
        yield xmax, ymax, xmin, ymax
        # izquierda
        yield xmin, ymax, xmin, ymin


def _point_in_bounds(x: float, y: float, world: World2D, cfg: PRMConfig) -> bool:
    L = float(getattr(world, "world_limit", cfg.world_limit))
    return (-L <= x <= L) and (-L <= y <= L)


def is_free(world: World2D, x: float, y: float, cfg: PRMConfig) -> bool:
    if not _point_in_bounds(x, y, world, cfg):
        return False

    # --- clearance contra rectángulos (AABB) ---
    # Inflamos el rectángulo por 'clearance' y si el punto cae dentro -> colisión
    c = cfg.clearance
    for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
        if (xmin - c) <= x <= (xmax + c) and (ymin - c) <= y <= (ymax + c):
            return False

    # --- clearance contra segmentos (si los usas en el futuro) ---
    for (x1, y1, x2, y2) in _iter_static_segments(world):
        if _dist_point_to_segment(x, y, x1, y1, x2, y2) <= cfg.clearance:
            return False

    return True


def edge_free(world: World2D, p1: Point2, p2: Point2, cfg: PRMConfig) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist <= 1e-9:
        return is_free(world, x1, y1, cfg)

    steps = max(2, int(dist / max(cfg.edge_step, 1e-6)))
    for i in range(steps + 1):
        t = i / float(steps)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if not is_free(world, x, y, cfg):
            return False
    return True


# ----------- PRM -----------
def sample_free(world: World2D, cfg: PRMConfig) -> List[Point2]:
    rng = np.random.default_rng(cfg.seed)
    L = float(getattr(world, "world_limit", cfg.world_limit))

    pts: List[Point2] = []
    tries = 0
    while len(pts) < cfg.n_samples and tries < cfg.max_sample_tries:
        tries += 1
        x = float(rng.uniform(-L, L))
        y = float(rng.uniform(-L, L))
        if is_free(world, x, y, cfg):
            pts.append((x, y))

    if len(pts) < max(50, int(0.2 * cfg.n_samples)):
        raise RuntimeError(
            f"Muy pocas muestras libres: {len(pts)}/{cfg.n_samples}. "
            "Baja clearance, sube max_sample_tries, o revisa el SDF."
        )

    return pts


def build_prm_graph(world: World2D, points: List[Point2], cfg: PRMConfig) -> nx.Graph:
    G = nx.Graph()
    for i, p in enumerate(points):
        G.add_node(i, pos=p)

    P = np.array(points, dtype=np.float64)
    N = len(points)

    for i in range(N):
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        nn = np.argsort(d)
        neighbors = [int(j) for j in nn[1 : 1 + cfg.k_neighbors]]

        for j in neighbors:
            if G.has_edge(i, j):
                continue
            p1 = points[i]
            p2 = points[j]
            if edge_free(world, p1, p2, cfg):
                G.add_edge(i, j, weight=float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])))

    return G


def connect_node(world: World2D, G: nx.Graph, pos: Point2, cfg: PRMConfig) -> int:
    if not is_free(world, pos[0], pos[1], cfg):
        raise ValueError(f"Start/Goal en colisión o fuera de límites: {pos}")

    new_id = (max(G.nodes) + 1) if len(G.nodes) else 0
    G.add_node(new_id, pos=pos)

    nodes = [n for n in G.nodes if n != new_id]
    pts = np.array([G.nodes[n]["pos"] for n in nodes], dtype=np.float64)
    d = np.hypot(pts[:, 0] - pos[0], pts[:, 1] - pos[1])
    order = np.argsort(d)

    added = 0
    for idx in order[: max(20, cfg.k_neighbors * 2)]:
        nid = nodes[int(idx)]
        p2 = G.nodes[nid]["pos"]
        if edge_free(world, pos, p2, cfg):
            G.add_edge(new_id, nid, weight=float(math.hypot(p2[0] - pos[0], p2[1] - pos[1])))
            added += 1
            if added >= cfg.k_neighbors:
                break

    return new_id


def shortest_path(world: World2D, G: nx.Graph, start: Point2, goal: Point2, cfg: PRMConfig) -> List[Point2]:
    s_id = connect_node(world, G, start, cfg)
    g_id = connect_node(world, G, goal, cfg)

    def h(a: int, b: int) -> float:
        ax, ay = G.nodes[a]["pos"]
        bx, by = G.nodes[b]["pos"]
        return float(math.hypot(ax - bx, ay - by))

    path_ids = nx.astar_path(G, s_id, g_id, heuristic=h, weight="weight")
    return [tuple(map(float, G.nodes[i]["pos"])) for i in path_ids]


def shortcut_smooth(world: World2D, path: List[Point2], cfg: PRMConfig, iters: int = 200) -> List[Point2]:
    if len(path) <= 2:
        return path

    rng = np.random.default_rng(cfg.seed + 999)
    pts = path[:]
    for _ in range(iters):
        if len(pts) <= 2:
            break
        i = int(rng.integers(0, len(pts) - 1))
        j = int(rng.integers(i + 1, len(pts)))
        if j <= i + 1:
            continue
        if edge_free(world, pts[i], pts[j], cfg):
            pts = pts[: i + 1] + pts[j:]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=str, required=True, help="Ruta al .sdf (por ejemplo worlds/maze.sdf)")
    ap.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--out", type=str, default="integration/path.json")

    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--clearance", type=float, default=0.20)
    ap.add_argument("--edge-step", type=float, default=0.05)
    ap.add_argument("--world-limit", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth-iters", type=int, default=200)

    ap.add_argument("--plot", action="store_true", help="Muestra path (requiere matplotlib)")
    ap.add_argument("--plot-prm", action="store_true", help="Dibuja nodos y aristas PRM (puede ser pesado)")
    ap.add_argument("--plot-edges-max", type=int, default=20000, help="Cap de aristas a dibujar")

    args = ap.parse_args()

    cfg = PRMConfig(
        n_samples=args.n_samples,
        k_neighbors=args.k,
        clearance=args.clearance,
        edge_step=args.edge_step,
        world_limit=args.world_limit,
        seed=args.seed,
    )

    world = World2D(sdf_path=args.world)

    start = (float(args.start[0]), float(args.start[1]))
    goal = (float(args.goal[0]), float(args.goal[1]))

    points = sample_free(world, cfg)
    G = build_prm_graph(world, points, cfg)
    path = shortest_path(world, G, start, goal, cfg)

    if args.smooth:
        path = shortcut_smooth(world, path, cfg, iters=args.smooth_iters)

    payload = {
        "world": args.world,
        "start": {"x": start[0], "y": start[1]},
        "goal": {"x": goal[0], "y": goal[1]},
        "clearance": cfg.clearance,
        "edge_step": cfg.edge_step,
        "n_samples": cfg.n_samples,
        "k_neighbors": cfg.k_neighbors,
        "seed": cfg.seed,
        "path": [{"x": float(x), "y": float(y)} for (x, y) in path],
    }

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[PRM] saved path -> {out_path}  (N={len(path)} waypoints)")

    if args.plot or args.plot_prm:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.figure()

        # --- DIBUJAR MAPA (rectángulos) ---
        for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
            w = xmax - xmin
            h = ymax - ymin
            plt.gca().add_patch(
                patches.Rectangle(
                    (xmin, ymin), w, h,
                    fill=False, linewidth=1.5
                )
            )

        # --- PRM (nodos + aristas) ---
        if args.plot_prm:
            edges = list(G.edges())
            if len(edges) > args.plot_edges_max:
                edges = edges[: args.plot_edges_max]

            for (u, v) in edges:
                (x1, y1) = G.nodes[u]["pos"]
                (x2, y2) = G.nodes[v]["pos"]
                plt.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.2)

            xs = [G.nodes[n]["pos"][0] for n in G.nodes()]
            ys = [G.nodes[n]["pos"][1] for n in G.nodes()]
            plt.scatter(xs, ys, s=6, alpha=0.4)

        # --- ruta ---
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, marker="o", linewidth=2.5)

        # start/goal
        #plt.scatter([start[0]], [start[1]], marker="s")
        plt.scatter([goal[0]], [goal[1]], marker="*")

        plt.axis("equal")
        plt.grid(True)
        plt.title("PRM + Path + Map" if args.plot_prm else "Path + Map")
        plt.show()


if __name__ == "__main__":
    main()
