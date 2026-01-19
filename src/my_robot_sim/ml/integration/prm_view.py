# ml/integration/prm_view.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

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
    edge_step: float = 0.05
    clearance: float = 0.20
    world_limit: float = 10.0
    seed: int = 123


# ---------------- geometry / collision ----------------
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
    rects = getattr(world, "static_rects", None)
    if rects is None:
        return
        yield  # type: ignore[misc]
    for r in rects:
        xmin, ymin, xmax, ymax = r
        yield float(xmin), float(ymin), float(xmax), float(ymax)


def _iter_static_segments(world: World2D) -> Iterable[Tuple[float, float, float, float]]:
    # derive segments from rects (good enough for clearance checks)
    for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
        yield xmin, ymin, xmax, ymin
        yield xmax, ymin, xmax, ymax
        yield xmax, ymax, xmin, ymax
        yield xmin, ymax, xmin, ymin


def _point_in_bounds(x: float, y: float, world: World2D, cfg: PRMConfig) -> bool:
    L = float(getattr(world, "world_limit", cfg.world_limit))
    return (-L <= x <= L) and (-L <= y <= L)


def is_free(world: World2D, x: float, y: float, cfg: PRMConfig) -> bool:
    if not _point_in_bounds(x, y, world, cfg):
        return False

    c = cfg.clearance
    for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
        if (xmin - c) <= x <= (xmax + c) and (ymin - c) <= y <= (ymax + c):
            return False

    # optional: segment distance too (a bit redundant with AABB inflate)
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


# ---------------- PRM ----------------
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
            "Baja clearance o revisa el SDF."
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
            p1, p2 = points[i], points[j]
            if edge_free(world, p1, p2, cfg):
                G.add_edge(i, j, weight=float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])))
    return G


def connect_node(world: World2D, G: nx.Graph, pos: Point2, cfg: PRMConfig) -> int:
    if not is_free(world, pos[0], pos[1], cfg):
        raise ValueError(f"Start/Goal en colisión o fuera: {pos}")

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


def compute_path(world: World2D, G_base: nx.Graph, start: Point2, goal: Point2, cfg: PRMConfig) -> List[Point2]:
    # copiamos base para no ir “ensuciando” con nodos start/goal
    G = G_base.copy()

    s_id = connect_node(world, G, start, cfg)
    g_id = connect_node(world, G, goal, cfg)

    def h(a: int, b: int) -> float:
        ax, ay = G.nodes[a]["pos"]
        bx, by = G.nodes[b]["pos"]
        return float(math.hypot(ax - bx, ay - by))

    ids = nx.astar_path(G, s_id, g_id, heuristic=h, weight="weight")
    return [tuple(map(float, G.nodes[i]["pos"])) for i in ids]


# ---------------- interactive viewer ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--clearance", type=float, default=0.20)
    ap.add_argument("--edge-step", type=float, default=0.05)
    ap.add_argument("--world-limit", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--plot-prm", action="store_true", help="Dibuja nodos/aristas (puede ser pesado)")
    ap.add_argument("--plot-edges-max", type=int, default=20000)
    ap.add_argument("--out", type=str, default="integration/path_click.json")
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

    print("[PRM] building roadmap...")
    points = sample_free(world, cfg)
    G = build_prm_graph(world, points, cfg)
    print(f"[PRM] roadmap ready: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.grid(True)

    # cache: artists
    start_pt: Optional[Point2] = None
    goal_pt: Optional[Point2] = None
    path_pts: Optional[List[Point2]] = None

    start_artist = None
    goal_artist = None
    path_artist = None
    msg_artist = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def redraw_map_and_prm():
        ax.clear()
        ax.set_aspect("equal", "box")
        ax.grid(True)

        # map rects
        for (xmin, ymin, xmax, ymax) in _iter_static_rects(world):
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1.5))

        # prm
        if args.plot_prm:
            edges = list(G.edges())
            if len(edges) > args.plot_edges_max:
                edges = edges[: args.plot_edges_max]
            for (u, v) in edges:
                (x1, y1) = G.nodes[u]["pos"]
                (x2, y2) = G.nodes[v]["pos"]
                ax.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.2)
            xs = [G.nodes[n]["pos"][0] for n in G.nodes()]
            ys = [G.nodes[n]["pos"][1] for n in G.nodes()]
            ax.scatter(xs, ys, s=6, alpha=0.35)

        # limits
        L = float(getattr(world, "world_limit", cfg.world_limit))
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)

    def redraw_overlays():
        nonlocal start_artist, goal_artist, path_artist, msg_artist
        if start_pt is not None:
            start_artist = ax.scatter([start_pt[0]], [start_pt[1]], marker="s")
        if goal_pt is not None:
            goal_artist = ax.scatter([goal_pt[0]], [goal_pt[1]], marker="*")
        if path_pts is not None and len(path_pts) >= 2:
            xs = [p[0] for p in path_pts]
            ys = [p[1] for p in path_pts]
            path_artist, = ax.plot(xs, ys, marker="o", linewidth=2.5)
        msg_artist = ax.text(0.02, 0.98, help_text(), transform=ax.transAxes, va="top")

    def help_text() -> str:
        lines = [
            "Click izquierdo: START (si no hay), luego GOAL",
            "Teclas: [r]=reset  [s]=save JSON  [q]=quit",
        ]
        if start_pt is None:
            lines.append("Estado: elige START")
        elif goal_pt is None:
            lines.append("Estado: elige GOAL")
        else:
            lines.append("Estado: start+goal listos (clic para cambiar GOAL)")
        return "\n".join(lines)

    def compute_and_draw():
        nonlocal path_pts
        if start_pt is None or goal_pt is None:
            return
        try:
            path_pts = compute_path(world, G, start_pt, goal_pt, cfg)
            print(f"[PRM] path ok: {len(path_pts)} waypoints")
        except Exception as e:
            path_pts = None
            print(f"[PRM] no path: {e}")

    def on_click(event):
        nonlocal start_pt, goal_pt
        if event.inaxes != ax:
            return
        x, y = float(event.xdata), float(event.ydata)

        if start_pt is None:
            if not is_free(world, x, y, cfg):
                print("[PRM] START inválido (colisión).")
                return
            start_pt = (x, y)
        else:
            if not is_free(world, x, y, cfg):
                print("[PRM] GOAL inválido (colisión).")
                return
            goal_pt = (x, y)

        compute_and_draw()
        redraw_map_and_prm()
        redraw_overlays()
        fig.canvas.draw_idle()

    def save_json():
        if start_pt is None or goal_pt is None or not path_pts:
            print("[PRM] Nada que guardar (falta start/goal/path).")
            return
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "world": args.world,
            "start": {"x": start_pt[0], "y": start_pt[1]},
            "goal": {"x": goal_pt[0], "y": goal_pt[1]},
            "clearance": cfg.clearance,
            "edge_step": cfg.edge_step,
            "n_samples": cfg.n_samples,
            "k_neighbors": cfg.k_neighbors,
            "seed": cfg.seed,
            "path": [{"x": float(x), "y": float(y)} for (x, y) in path_pts],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[PRM] saved -> {out}")

    def on_key(event):
        nonlocal start_pt, goal_pt, path_pts
        if event.key == "q":
            plt.close(fig)
        elif event.key == "r":
            start_pt, goal_pt, path_pts = None, None, None
            redraw_map_and_prm()
            redraw_overlays()
            fig.canvas.draw_idle()
        elif event.key == "s":
            save_json()

    redraw_map_and_prm()
    redraw_overlays()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.title("PRM interactive: click start, click goal")
    plt.show()


if __name__ == "__main__":
    main()
