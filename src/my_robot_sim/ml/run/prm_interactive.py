# ml/run/prm_interactive.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx

from sim.core.world import World2D

Point2 = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


# ---------------- PRM core (AABB rects) ----------------
@dataclass
class PRMConfig:
    n_samples: int = 800
    k_neighbors: int = 12
    max_sample_tries: int = 50_000
    edge_step: float = 0.05
    clearance: float = 0.20
    world_limit: float = 12.0
    seed: int = 123


def _inflated_rects(static_rects: List[Rect], clearance: float) -> List[Rect]:
    c = float(clearance)
    return [(xmin - c, ymin - c, xmax + c, ymax + c) for (xmin, ymin, xmax, ymax) in static_rects]


def _in_bounds(x: float, y: float, cfg: PRMConfig) -> bool:
    L = cfg.world_limit
    return (-L <= x <= L) and (-L <= y <= L)


def is_free_rects(x: float, y: float, inflated_rects: List[Rect], cfg: PRMConfig) -> bool:
    if not _in_bounds(x, y, cfg):
        return False
    for (xmin, ymin, xmax, ymax) in inflated_rects:
        if (xmin <= x <= xmax) and (ymin <= y <= ymax):
            return False
    return True


def edge_free_rects(p1: Point2, p2: Point2, inflated_rects: List[Rect], cfg: PRMConfig) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    dist = float(math.hypot(x2 - x1, y2 - y1))
    if dist <= 1e-9:
        return is_free_rects(x1, y1, inflated_rects, cfg)

    steps = max(2, int(dist / max(cfg.edge_step, 1e-6)))
    for i in range(steps + 1):
        t = i / float(steps)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if not is_free_rects(x, y, inflated_rects, cfg):
            return False
    return True


def sample_free_points(inflated_rects: List[Rect], cfg: PRMConfig) -> List[Point2]:
    rng = np.random.default_rng(cfg.seed)
    L = cfg.world_limit
    pts: List[Point2] = []
    tries = 0

    while len(pts) < cfg.n_samples and tries < cfg.max_sample_tries:
        tries += 1
        x = float(rng.uniform(-L, L))
        y = float(rng.uniform(-L, L))
        if is_free_rects(x, y, inflated_rects, cfg):
            pts.append((x, y))

    if len(pts) < max(50, int(0.2 * cfg.n_samples)):
        raise RuntimeError(
            f"Muy pocas muestras libres: {len(pts)}/{cfg.n_samples}. "
            "Baja clearance o revisa world_limit / mapa."
        )
    return pts


def build_prm_graph(points: List[Point2], inflated_rects: List[Rect], cfg: PRMConfig) -> nx.Graph:
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
            if edge_free_rects(p1, p2, inflated_rects, cfg):
                G.add_edge(i, j, weight=float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])))

    return G


def connect_node(G: nx.Graph, pos: Point2, inflated_rects: List[Rect], cfg: PRMConfig) -> int:
    if not is_free_rects(pos[0], pos[1], inflated_rects, cfg):
        raise ValueError(f"Nodo en colisión: {pos}")

    new_id = (max(G.nodes) + 1) if len(G.nodes) else 0
    G.add_node(new_id, pos=pos)

    nodes = [n for n in G.nodes if n != new_id]
    pts = np.array([G.nodes[n]["pos"] for n in nodes], dtype=np.float64)
    d = np.hypot(pts[:, 0] - pos[0], pts[:, 1] - pos[1])
    order = np.argsort(d)

    added = 0
    for idx in order[: max(30, cfg.k_neighbors * 3)]:
        nid = nodes[int(idx)]
        p2 = G.nodes[nid]["pos"]
        if edge_free_rects(pos, p2, inflated_rects, cfg):
            G.add_edge(new_id, nid, weight=float(math.hypot(p2[0] - pos[0], p2[1] - pos[1])))
            added += 1
            if added >= cfg.k_neighbors:
                break

    if added == 0:
        raise RuntimeError("No pude conectar nodo al roadmap (clearance muy grande o muestreo pobre).")
    return new_id


def astar_path(G: nx.Graph, start_id: int, goal_id: int) -> List[Point2]:
    def h(a: int, b: int) -> float:
        ax, ay = G.nodes[a]["pos"]
        bx, by = G.nodes[b]["pos"]
        return float(math.hypot(ax - bx, ay - by))

    ids = nx.astar_path(G, start_id, goal_id, heuristic=h, weight="weight")
    return [tuple(map(float, G.nodes[i]["pos"])) for i in ids]


# ---------------- RL policy wrapper ----------------
class RLAvoidPolicy:
    """
    Wrapper para PPO discreto (acciones 0..6).
    - Autodetecta obs_dim del modelo (7/9/10).
    - Aplica el mismo mapeo de control: w_cmd = w_nom + k_rl*wcorr(a), v_cmd brake/boost/nom.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        k_rl: float = 4.0,
        v_nom: float = 0.4,
        v_brake: float = 0.15,  # alinear con entrenamiento típico
        v_boost: float = 0.80,
        v_max: float = 0.8,
        w_max: float = 5.0,
    ):
        self.model_path = model_path
        self.device = device

        self.k_rl = float(k_rl)
        self.v_nom = float(v_nom)
        self.v_brake = float(np.clip(v_brake, 0.0, v_max))
        self.v_boost = float(np.clip(v_boost, 0.0, v_max))
        self.v_max = float(v_max)
        self.w_max = float(w_max)

        self.model = None
        self.obs_dim: int = 7
        self._load()

    def _load(self):
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.model_path, device=self.device)
            try:
                self.obs_dim = int(self.model.observation_space.shape[0])
            except Exception:
                self.obs_dim = 7
            print(f"[RL] loaded -> {self.model_path} (obs_dim={self.obs_dim})")
        except Exception as e:
            self.model = None
            self.obs_dim = 7
            print(f"[RL] ❌ no pude cargar modelo: {self.model_path}\n     error: {e}")

    @staticmethod
    def _action_to_wcorr(a: int) -> float:
        if a == 0:
            return 0.0
        if a == 1:
            return +0.4
        if a == 2:
            return -0.4
        if a == 3:
            return +1.0
        if a == 4:
            return -1.0
        return 0.0

    def _action_to_vcmd(self, a: int) -> float:
        if a == 5:
            return self.v_brake
        if a == 6:
            return self.v_boost
        return float(np.clip(self.v_nom, 0.0, self.v_max))

    def act(self, obs: np.ndarray, v_nom_cmd: float, w_nom: float) -> Tuple[float, float, int]:
        """
        obs: shape (obs_dim,) float32
        returns: (v_cmd, w_cmd, action)
        """
        if self.model is None:
            v_cmd = float(np.clip(v_nom_cmd, 0.0, self.v_max))
            w_cmd = float(np.clip(w_nom, -self.w_max, self.w_max))
            return v_cmd, w_cmd, 0

        if obs is None:
            obs = np.zeros((self.obs_dim,), dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            z = np.zeros((self.obs_dim,), dtype=np.float32)
            m = min(self.obs_dim, obs.shape[0])
            z[:m] = obs[:m]
            obs = z

        action, _ = self.model.predict(obs, deterministic=True)
        a = int(action)

        w_cmd = float(w_nom + self.k_rl * self._action_to_wcorr(a))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

        if a in (5, 6):
            v_cmd = float(np.clip(self._action_to_vcmd(a), 0.0, self.v_max))
        else:
            v_cmd = float(np.clip(v_nom_cmd, 0.0, self.v_max))

        return v_cmd, w_cmd, a


# ---------------- Waypoint follower (polyline + lookahead) + RL switch ----------------
class WaypointFollower:
    """
    - Sigue la POLILÍNEA del path (segmentos).
    - Target = punto a lookahead sobre la polilínea.
    - Control nominal: Pure Pursuit.
    - RL: ACTIVACIÓN PRINCIPAL por distancia centro-a-centro (como d_dyn del entrenamiento).
      (Opcional) TTC como refuerzo.
    - Dinámica: rate limit en v y w.
    """

    def __init__(
        self,
        world: World2D,
        path: List[Point2],
        rl: Optional[RLAvoidPolicy] = None,
        rl_activate_dist: float = 1.2,   # distancia centro-a-centro
        rl_deactivate_dist: float = 1.4, # histéresis centro-a-centro
        lookahead_dist: float = 1.3,
        lookahead_rl: float = 0.9,
        ttc_activate: float = 1.2,       # refuerzo opcional
        v_eps_closing: float = 0.05,
        use_ttc_gate: bool = True,       # puedes apagar TTC si quieres "tal cual entreno"
    ):
        self.world = world
        self.path = [(float(x), float(y)) for (x, y) in path]

        # robot state
        self.x, self.y = self.path[0]
        if len(self.path) > 1:
            dx, dy = self.path[1][0] - self.x, self.path[1][1] - self.y
            self.theta = float(math.atan2(dy, dx))
        else:
            self.theta = 0.0

        # params
        self.dt = 0.05
        self.v_nom = 0.4
        self.v_max = 0.8
        self.w_max = 5.0
        self.robot_radius = 0.10
        self.wp_tol = 0.20

        # actuator-like limits
        self.a_lin = 0.8   # m/s^2
        self.a_ang = 5.0   # rad/s^2

        # pure pursuit
        self.lookahead_dist = float(max(0.5, lookahead_dist))
        self.lookahead_rl = float(max(0.5, lookahead_rl))

        self._s_cum = self._build_arc_lengths()
        self._s_progress = 0.0
        self._seg_hint = 0

        # runtime
        self.v = 0.0
        self.w = 0.0
        self.t_sim = 0.0
        self.last_action: int = 0

        # RL integration
        self.rl = rl
        self.rl_active = False
        self.rl_activate_dist = float(rl_activate_dist)
        self.rl_deactivate_dist = float(rl_deactivate_dist)

        # TTC gate
        self.use_ttc_gate = bool(use_ttc_gate)
        self.ttc_activate = float(ttc_activate)
        self.v_eps_closing = float(v_eps_closing)

        # debug: detect switch
        self._prev_rl_active: Optional[bool] = None

        # Trajectory with mode flag (0=nominal, 1=rl)
        self.traj: List[Point2] = [(self.x, self.y)]
        self.traj_mode: List[int] = [0]

    @staticmethod
    def _wrap(a: float) -> float:
        return float(math.atan2(math.sin(a), math.cos(a)))

    def _build_arc_lengths(self) -> np.ndarray:
        s = [0.0]
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            s.append(s[-1] + float(math.hypot(x2 - x1, y2 - y1)))
        return np.array(s, dtype=np.float64)

    @staticmethod
    def _proj_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        den = vx * vx + vy * vy
        if den <= 1e-12:
            return 0.0, ax, ay
        t = (wx * vx + wy * vy) / den
        t = float(max(0.0, min(1.0, t)))
        qx = ax + t * vx
        qy = ay + t * vy
        return t, qx, qy

    def _closest_point_on_polyline_ahead(self):
        nseg = max(0, len(self.path) - 1)
        if nseg == 0:
            return 0.0, 0

        i0 = int(max(0, min(nseg - 1, self._seg_hint)))
        i1 = int(min(nseg - 1, i0 + 40))

        best_d2 = 1e18
        best_s = float(self._s_progress)
        best_i = i0

        px, py = self.x, self.y
        for i in range(i0, i1 + 1):
            ax, ay = self.path[i]
            bx, by = self.path[i + 1]
            t, qx, qy = self._proj_point_to_segment(px, py, ax, ay, bx, by)
            ex, ey = px - qx, py - qy
            d2 = ex * ex + ey * ey

            dx, dy = bx - ax, by - ay
            seg_len = float(math.hypot(dx, dy))
            s_on_seg = float(self._s_cum[i] + t * seg_len)

            if d2 < best_d2:
                best_d2 = d2
                best_s = s_on_seg
                best_i = i

        return best_s, best_i

    def _point_at_s(self, s_target: float) -> Point2:
        s_target = float(max(0.0, min(s_target, float(self._s_cum[-1]))))
        idx = int(np.searchsorted(self._s_cum, s_target, side="right") - 1)
        idx = max(0, min(idx, len(self.path) - 2))

        s0 = float(self._s_cum[idx])
        s1 = float(self._s_cum[idx + 1])
        if s1 <= s0 + 1e-12:
            return self.path[idx + 1]

        t = (s_target - s0) / (s1 - s0)
        ax, ay = self.path[idx]
        bx, by = self.path[idx + 1]
        return (float(ax + t * (bx - ax)), float(ay + t * (by - ay)))

    # ✅ FIX 1: aceptar dynamic_circles con len>=3 (vx/vy opcionales)
    def _nearest_dynamic(self) -> Optional[Tuple[float, float, float, float, float]]:
        best = None
        best_d = float("inf")
        for c in getattr(self.world, "dynamic_circles", []):
            if c is None or len(c) < 3:
                continue
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            vx = float(c[3]) if len(c) >= 4 else 0.0
            vy = float(c[4]) if len(c) >= 5 else 0.0
            d = float(math.hypot(cx - self.x, cy - self.y))
            if d < best_d:
                best_d = d
                best = (cx, cy, r, vx, vy)
        return best

    def _rel_kinematics(self, dyn: Tuple[float, float, float, float, float]):
        cx, cy, _r, vx, vy = dyn
        dx = float(cx - self.x)
        dy = float(cy - self.y)
        d_dyn = float(math.hypot(dx, dy))
        bearing_world = float(math.atan2(dy, dx))
        alpha = self._wrap(bearing_world - self.theta)

        v_rx = self.v * math.cos(self.theta)
        v_ry = self.v * math.sin(self.theta)
        rvx = float(vx - v_rx)
        rvy = float(vy - v_ry)

        if d_dyn < 1e-6:
            v_closing = 0.0
        else:
            v_closing = float((rvx * dx + rvy * dy) / d_dyn)

        lx = -math.sin(self.theta)
        ly = math.cos(self.theta)
        v_lat_rel = float(rvx * lx + rvy * ly)

        return dx, dy, d_dyn, alpha, v_closing, v_lat_rel

    def _cross_track_signed(self) -> float:
        nseg = max(0, len(self.path) - 1)
        if nseg == 0:
            return 0.0

        i = int(max(0, min(nseg - 1, self._seg_hint)))
        ax, ay = self.path[i]
        bx, by = self.path[i + 1]
        vx, vy = (bx - ax), (by - ay)
        seg_len = float(math.hypot(vx, vy))
        if seg_len < 1e-9:
            return 0.0

        _t, qx, qy = self._proj_point_to_segment(self.x, self.y, ax, ay, bx, by)
        ex, ey = (self.x - qx), (self.y - qy)

        nx_, ny_ = (-vy / seg_len), (vx / seg_len)
        ct = float(ex * nx_ + ey * ny_)
        return ct

    def _compute_obs(self, target: Point2, dyn: Tuple[float, float, float, float, float]) -> np.ndarray:
        gx, gy = float(target[0]), float(target[1])
        dgx = gx - self.x
        dgy = gy - self.y
        dist_goal = float(math.hypot(dgx, dgy))
        ang_goal_world = float(math.atan2(dgy, dgx))
        ang_goal = self._wrap(ang_goal_world - self.theta)

        _dx, _dy, d_dyn, alpha, v_closing, v_lat_rel = self._rel_kinematics(dyn)

        base7 = [
            dist_goal,
            float(math.cos(ang_goal)),
            float(math.sin(ang_goal)),
            float(d_dyn),
            float(math.cos(alpha)),
            float(math.sin(alpha)),
            float(v_closing),
        ]

        obs_dim = 7
        if self.rl is not None:
            obs_dim = int(getattr(self.rl, "obs_dim", 7))

        if obs_dim == 7:
            return np.array(base7, dtype=np.float32)

        ct = self._cross_track_signed()
        ct_scale = 1.0
        ct_norm = float(np.clip(ct / max(ct_scale, 1e-6), -1.0, 1.0))

        wall_dist_norm = 1.0  # fallback

        if obs_dim == 9:
            return np.array(base7 + [float(v_lat_rel), float(ct_norm)], dtype=np.float32)

        if obs_dim == 10:
            return np.array(base7 + [float(v_lat_rel), float(ct_norm), float(wall_dist_norm)], dtype=np.float32)

        z = np.zeros((obs_dim,), dtype=np.float32)
        m = min(obs_dim, len(base7))
        z[:m] = np.array(base7[:m], dtype=np.float32)
        return z

    # ✅ FIX 2: activar por distancia centro-a-centro (d_dyn) como el entrenamiento.
    # TTC queda como refuerzo opcional (use_ttc_gate).
    def _update_rl_active(self, dyn: Optional[Tuple[float, float, float, float, float]]) -> Tuple[float, float]:
        """
        Returns: (d_center, ttc)
        """
        if dyn is None:
            self.rl_active = False
            return float("inf"), float("inf")

        cx, cy, r, vx, vy = dyn
        d_center = float(math.hypot(cx - self.x, cy - self.y))

        # TTC usando d_clear (solo para refuerzo), pero la activación principal es d_center.
        d_clear = float(d_center - (self.robot_radius + float(r)))
        _dx, _dy, _d_dyn, _alpha, v_closing, _vlat = self._rel_kinematics(dyn)
        closing = float(-v_closing)  # positivo si se acerca
        if closing > self.v_eps_closing:
            ttc = float(max(0.0, d_clear / closing))
        else:
            ttc = float("inf")

        want_on = (d_center <= self.rl_activate_dist) or (self.use_ttc_gate and (ttc <= self.ttc_activate))

        if self.rl_active:
            if (d_center >= self.rl_deactivate_dist) and (not self.use_ttc_gate or (ttc > self.ttc_activate * 1.5)):
                self.rl_active = False
        else:
            if want_on:
                self.rl_active = True

        return d_center, ttc

    def _traj_append(self, mode: int):
        self.traj.append((self.x, self.y))
        self.traj_mode.append(int(mode))

    def get_traj_split(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        tx_nom: List[float] = []
        ty_nom: List[float] = []
        tx_rl: List[float] = []
        ty_rl: List[float] = []

        prev_mode = self.traj_mode[0]
        for (p, m) in zip(self.traj, self.traj_mode):
            x, y = p
            if m == 0:
                if prev_mode == 1:
                    tx_nom.append(float("nan"))
                    ty_nom.append(float("nan"))
                tx_nom.append(x)
                ty_nom.append(y)
            else:
                if prev_mode == 0:
                    tx_rl.append(float("nan"))
                    ty_rl.append(float("nan"))
                tx_rl.append(x)
                ty_rl.append(y)
            prev_mode = m

        return tx_nom, ty_nom, tx_rl, ty_rl

    def _rate_limit(self, current: float, cmd: float, amax: float) -> float:
        dv = float(cmd - current)
        dv_max = float(amax * self.dt)
        dv = float(np.clip(dv, -dv_max, dv_max))
        return float(current + dv)

    def step(self) -> bool:
        gx_final, gy_final = self.path[-1]
        if float(math.hypot(gx_final - self.x, gy_final - self.y)) < self.wp_tol:
            return True

        # 1) progress
        s_closest, seg_idx = self._closest_point_on_polyline_ahead()
        self._seg_hint = int(seg_idx)
        self._s_progress = float(max(self._s_progress, s_closest))

        # 2) RL gating (d_center + opcional TTC)
        dyn = self._nearest_dynamic()
        d_center, ttc = self._update_rl_active(dyn)

        # debug switch (una línea cuando cambia)
        if self._prev_rl_active is None:
            self._prev_rl_active = self.rl_active
        if self.rl_active != self._prev_rl_active:
            print(f"[RL_SWITCH] rl_active={self.rl_active} d_center={d_center:.2f} ttc={ttc:.2f}")
            self._prev_rl_active = self.rl_active

        # lookahead adaptativo
        L_use = self.lookahead_rl if (self.rl_active) else self.lookahead_dist

        # 3) target
        s_target = float(min(self._s_progress + L_use, float(self._s_cum[-1])))
        tx, ty = self._point_at_s(s_target)
        target = (tx, ty)

        # 4) nominal pure pursuit
        dx = tx - self.x
        dy = ty - self.y
        ang_target = float(math.atan2(dy, dx))
        alpha_pp = self._wrap(ang_target - self.theta)

        v_nom_cmd = float(np.clip(self.v_nom, 0.0, self.v_max))

        Ld = float(max(L_use, 1e-3))
        w_nom = float(2.0 * v_nom_cmd * math.sin(alpha_pp) / Ld)
        w_nom = float(np.clip(w_nom, -self.w_max, self.w_max))

        # 5) RL override
        mode = 0
        v_cmd = v_nom_cmd
        w_cmd = w_nom

        if self.rl_active and (self.rl is not None) and (dyn is not None):
            obs = self._compute_obs(target, dyn)
            v_cmd, w_cmd, a = self.rl.act(obs, v_nom_cmd=v_nom_cmd, w_nom=w_nom)
            self.last_action = int(a)
            mode = 1
        else:
            self.last_action = 0
            mode = 0

        # 6) rate limit + saturación
        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))
        self.v = float(np.clip(self._rate_limit(self.v, v_cmd, self.a_lin), 0.0, self.v_max))
        self.w = float(np.clip(self._rate_limit(self.w, w_cmd, self.a_ang), -self.w_max, self.w_max))

        # integrate
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta = self._wrap(self.theta + self.w * self.dt)

        # world step
        self.world.step(self.dt)
        self.t_sim += self.dt

        self._traj_append(mode)
        _ = (d_center, ttc)
        return False


# ---------------- CLI selection ----------------
def _resolve_worlds_dir(worlds_dir: str) -> Path:
    p = Path(worlds_dir)
    if p.exists():
        return p.resolve()

    here = Path(__file__).resolve()
    base_ml = here.parents[1]  # .../ml
    p2 = (base_ml / worlds_dir).resolve()
    if p2.exists():
        return p2
    return p.resolve()


def list_worlds(worlds_dir: Path) -> List[Path]:
    w = sorted(worlds_dir.glob("*.sdf"))
    if not w:
        raise RuntimeError(f"No encuentro .sdf en: {worlds_dir}")
    return w


def prompt_world(worlds: List[Path]) -> int:
    print("\n=== Selecciona WORLD ===")
    for i, p in enumerate(worlds, start=1):
        print(f"{i:2d}) {p.name}")
    while True:
        s = input("World #> ").strip()
        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(worlds):
                return idx
        print("❌ inválido. Escribe un número de la lista.")


# ---------------- Plot UI ----------------
class PlotRunner:
    def __init__(
        self,
        world_path: Path,
        out_json: Path,
        cfg: PRMConfig,
        plot_edges_max: int,
        rl_model: Optional[str],
        rl_device: str,
        rl_activate: float,
        rl_deactivate: float,
        dyn_specs: List[Tuple[float, float, float, float, float]],
        lookahead: float,
        lookahead_rl: float,
        ttc_activate: float,
        use_ttc_gate: bool,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.animation import FuncAnimation

        self.cfg = cfg
        self.out_json = out_json
        self.plot_edges_max = int(plot_edges_max)

        self.world_path = world_path
        self.world = World2D(str(world_path))
        self.static_rects: List[Rect] = list(getattr(self.world, "static_rects", []))
        self.inflated_rects = _inflated_rects(self.static_rects, self.cfg.clearance)

        self._dyn_specs = dyn_specs[:]
        self._apply_dyn_specs()

        self.start: Point2 = (0.0, 0.0)
        if not is_free_rects(self.start[0], self.start[1], self.inflated_rects, self.cfg):
            raise RuntimeError("Start fijo (0,0) cae en obstáculo. Baja clearance o cambia mapa.")

        self.goal: Optional[Point2] = None

        self.rl_policy: Optional[RLAvoidPolicy] = None
        self.rl_activate = float(rl_activate)
        self.rl_deactivate = float(rl_deactivate)
        self.lookahead = float(lookahead)
        self.lookahead_rl = float(lookahead_rl)
        self.ttc_activate = float(ttc_activate)
        self.use_ttc_gate = bool(use_ttc_gate)

        if rl_model:
            self.rl_policy = RLAvoidPolicy(
                model_path=rl_model,
                device=rl_device,
                k_rl=4.0,
                v_nom=0.4,
                v_brake=0.15,
                v_boost=0.80,
                v_max=0.8,
                w_max=5.0,
            )

        self._prm_built = False
        self.G_base: Optional[nx.Graph] = None

        self.path: Optional[List[Point2]] = None
        self.follower: Optional[WaypointFollower] = None

        self.show_prm: bool = True

        self.plt = plt
        self.FuncAnimation = FuncAnimation
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        L = self.cfg.world_limit
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_aspect("equal", "box")
        self.ax.grid(True)

        for (xmin, ymin, xmax, ymax) in self.static_rects:
            self.ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1.0))

        self.Circle = Circle
        self.dyn_patches: List[Circle] = []
        self._init_dyn_patches()

        self.robot_radius = 0.10
        self.robot_circle = Circle(self.start, self.robot_radius, fill=False, linewidth=2.0)
        self.ax.add_patch(self.robot_circle)
        self.robot_heading, = self.ax.plot(
            [self.start[0], self.start[0] + self.robot_radius],
            [self.start[1], self.start[1]],
            linewidth=2.0,
        )

        self.goal_artist, = self.ax.plot([], [], marker="x", markersize=8)
        self.path_line, = self.ax.plot([], [], marker="o", markersize=3, linewidth=1.2, alpha=0.95)

        self.traj_nom_line, = self.ax.plot([], [], linewidth=1.2, alpha=0.95)
        self.traj_rl_line, = self.ax.plot([], [], linewidth=1.8, alpha=0.95)

        self.prm_edges_lines: List = []
        self.prm_nodes_scatter = None

        self.title = self.ax.set_title(f"{self.world_path.name} | start=(0,0) | clearance={self.cfg.clearance:.2f}")
        self.help_text = self.ax.text(
            0.01,
            0.01,
            "Click: GOAL | B build PRM | P plan | F follow | T toggle PRM | S save | R reset | ESC quit",
            transform=self.ax.transAxes,
            fontsize=9,
        )
        self.time_text = self.ax.text(
            0.5,
            1.01,
            "t=0.00s",
            transform=self.ax.transAxes,
            ha="center",
            fontsize=10,
        )

        self._anim = None
        self._running = False

        self._update_artists()
        self.fig.canvas.draw()

        print("\n[UI] Listo.")
        print("START fijo en (0,0). Click para GOAL.")
        if self.rl_policy is None:
            print("RL: OFF (no model). Usa --rl-model para activar.")
        else:
            gate = "dist(d_center)+TTC" if self.use_ttc_gate else "dist(d_center) ONLY"
            print(
                f"RL: ON  gate={gate} | activate<= {self.rl_activate:.2f}m | deactivate>= {self.rl_deactivate:.2f}m | "
                f"ttc<={self.ttc_activate:.2f}s | lookahead={self.lookahead:.2f} (rl={self.lookahead_rl:.2f})"
            )
        print("Teclas: B build PRM | P plan | F follow | T toggle PRM | S save | R reset | ESC quit")

    # ---------- dinámicos ----------
    def _apply_dyn_specs(self):
        dyn = []
        for (x, y, r, vx, vy) in self._dyn_specs:
            dyn.append([float(x), float(y), float(r), float(vx), float(vy)])
        self.world.dynamic_circles = dyn

    def _init_dyn_patches(self):
        for p in self.dyn_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.dyn_patches = []

        for c in getattr(self.world, "dynamic_circles", []):
            if len(c) < 3:
                continue
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            patch = self.Circle((cx, cy), r, fill=False, linewidth=1.5)
            self.ax.add_patch(patch)
            self.dyn_patches.append(patch)

    def _update_dyn_patches(self):
        dyn = getattr(self.world, "dynamic_circles", [])
        if len(dyn) != len(self.dyn_patches):
            self._init_dyn_patches()
            return
        for patch, c in zip(self.dyn_patches, dyn):
            if len(c) < 3:
                continue
            patch.center = (float(c[0]), float(c[1]))
            patch.radius = float(c[2])

    # ---------- PRM draw control ----------
    def _remove_prm_artists(self):
        for ln in self.prm_edges_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.prm_edges_lines = []
        if self.prm_nodes_scatter is not None:
            try:
                self.prm_nodes_scatter.remove()
            except Exception:
                pass
            self.prm_nodes_scatter = None

    def _draw_prm_once(self):
        if not self.show_prm:
            return
        self._remove_prm_artists()
        if self.G_base is None:
            return

        edges = list(self.G_base.edges())
        if len(edges) > self.plot_edges_max:
            edges = edges[: self.plot_edges_max]

        for (u, v) in edges:
            (x1, y1) = self.G_base.nodes[u]["pos"]
            (x2, y2) = self.G_base.nodes[v]["pos"]
            ln, = self.ax.plot([x1, x2], [y1, y2], linewidth=0.30, alpha=0.06, zorder=2)
            self.prm_edges_lines.append(ln)

        xs = [self.G_base.nodes[n]["pos"][0] for n in self.G_base.nodes()]
        ys = [self.G_base.nodes[n]["pos"][1] for n in self.G_base.nodes()]
        self.prm_nodes_scatter = self.ax.scatter(xs, ys, s=3, alpha=0.12, zorder=3)

    def toggle_prm(self):
        self.show_prm = not self.show_prm
        if not self.show_prm:
            self._remove_prm_artists()
            print("[UI] PRM oculto.")
        else:
            if self._prm_built:
                self._draw_prm_once()
            print("[UI] PRM visible.")
        self.fig.canvas.draw_idle()

    # ---------- fast updates ----------
    def _update_artists(self):
        self._update_dyn_patches()

        if self.follower is None:
            x, y, th = self.start[0], self.start[1], 0.0
            t = 0.0
            mode = "NOM"
            a = 0
        else:
            x, y, th = self.follower.x, self.follower.y, self.follower.theta
            t = self.follower.t_sim
            mode = "RL" if self.follower.rl_active else "NOM"
            a = getattr(self.follower, "last_action", 0)

        self.robot_circle.center = (x, y)
        self.robot_heading.set_data(
            [x, x + self.robot_radius * math.cos(th)],
            [y, y + self.robot_radius * math.sin(th)],
        )

        if self.goal is None:
            self.goal_artist.set_data([], [])
        else:
            self.goal_artist.set_data([self.goal[0]], [self.goal[1]])

        if self.path:
            xs = [p[0] for p in self.path]
            ys = [p[1] for p in self.path]
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        if self.follower and self.follower.traj:
            tx_nom, ty_nom, tx_rl, ty_rl = self.follower.get_traj_split()
            self.traj_nom_line.set_data(tx_nom, ty_nom)
            self.traj_rl_line.set_data(tx_rl, ty_rl)
        else:
            self.traj_nom_line.set_data([], [])
            self.traj_rl_line.set_data([], [])

        self.time_text.set_text(f"t={t:.2f}s")

        if self.follower is not None:
            self.title.set_text(
                f"{self.world_path.name} | t={t:.2f}s | mode={mode} | a={a} | clearance={self.cfg.clearance:.2f}"
            )
        else:
            self.title.set_text(f"{self.world_path.name} | start=(0,0) | clearance={self.cfg.clearance:.2f}")

    def _artists_for_blit(self):
        arts = [
            self.robot_circle,
            self.robot_heading,
            self.goal_artist,
            self.path_line,
            self.traj_nom_line,
            self.traj_rl_line,
            self.help_text,
            self.time_text,
            self.title,
        ]
        arts.extend(self.dyn_patches)

        if self.show_prm:
            if self.prm_nodes_scatter is not None:
                arts.append(self.prm_nodes_scatter)
            arts.extend(self.prm_edges_lines)
        return arts

    # ---------- events ----------
    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if not is_free_rects(x, y, self.inflated_rects, self.cfg):
            print("❌ GOAL en obstáculo/clearance. Elige otro.")
            return
        self.goal = (x, y)
        self.path = None
        self.follower = None
        self._update_artists()
        self.fig.canvas.draw_idle()
        print(f"✅ goal={self.goal}")

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "escape":
            self.plt.close(self.fig)
            return
        if k == "r":
            self.plt.close(self.fig)
            raise KeyboardInterrupt
        if k == "t":
            self.toggle_prm()
            return
        if k == "b":
            self.build_prm_cached()
            self._update_artists()
            self.fig.canvas.draw_idle()
            return
        if k == "p":
            self.plan()
            self._update_artists()
            self.fig.canvas.draw_idle()
            return
        if k == "f":
            self.follow_real_time()
            return
        if k == "s":
            self.save_json()
            return

    # ---------- PRM cache ----------
    def build_prm_cached(self):
        if self._prm_built:
            print("[PRM] ya estaba construido (cache).")
            return
        print(f"[PRM] BUILD (cache) n={self.cfg.n_samples} k={self.cfg.k_neighbors} clearance={self.cfg.clearance:.2f}")
        pts = sample_free_points(self.inflated_rects, self.cfg)
        G = build_prm_graph(pts, self.inflated_rects, self.cfg)
        self.G_base = G
        self._prm_built = True
        print(f"[PRM] cache listo: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        if self.show_prm:
            self._draw_prm_once()

    def plan(self):
        if self.goal is None:
            print("❌ Falta GOAL (click).")
            return
        if not self._prm_built:
            self.build_prm_cached()
        assert self.G_base is not None

        G = self.G_base.copy()
        try:
            s_id = connect_node(G, self.start, self.inflated_rects, self.cfg)
            g_id = connect_node(G, self.goal, self.inflated_rects, self.cfg)
            path = astar_path(G, s_id, g_id)
        except Exception as e:
            print(f"[PRM] no path: {e}")
            self.path = None
            return

        self.path = path
        self.follower = None
        print(f"[PRM] path ok -> {len(path)} waypoints")

    # ---------- FOLLOW ----------
    def follow_real_time(self):
        if not self.path:
            print("[UI] no hay path. Presiona P primero.")
            return
        if self._running:
            print("[UI] ya está corriendo.")
            return

        if self.show_prm:
            self.show_prm = False
            self._remove_prm_artists()
            print("[UI] FOLLOW: PRM oculto (solo path + traj).")

        self.world = World2D(str(self.world_path))
        self._apply_dyn_specs()
        self._init_dyn_patches()

        self.follower = WaypointFollower(
            self.world,
            self.path,
            rl=self.rl_policy,
            rl_activate_dist=self.rl_activate,
            rl_deactivate_dist=self.rl_deactivate,
            lookahead_dist=self.lookahead,
            lookahead_rl=self.lookahead_rl,
            ttc_activate=self.ttc_activate,
            use_ttc_gate=self.use_ttc_gate,
        )
        self.follower.t_sim = 0.0

        self._running = True

        steps_per_frame = 1
        frame_interval_ms = int(round(self.follower.dt * 1000.0))

        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass
            self._anim = None

        def _init():
            self._update_artists()
            return self._artists_for_blit()

        def _update(_frame_idx):
            assert self.follower is not None
            done = False
            for _ in range(steps_per_frame):
                done = self.follower.step()
                if done:
                    break

            self._update_artists()

            if done:
                print("✅ Goal reached (follow + RL override)")
                self._running = False
                if self._anim is not None:
                    self._anim.event_source.stop()

            return self._artists_for_blit()

        self._anim = self.FuncAnimation(
            self.fig,
            _update,
            init_func=_init,
            interval=frame_interval_ms,
            blit=True,
            cache_frame_data=False,
        )
        self.fig.canvas.draw_idle()

    def save_json(self):
        if not self.path or self.goal is None:
            print("[UI] no hay path para guardar. Presiona P primero.")
            return

        payload = {
            "world": str(self.world_path),
            "start": {"x": 0.0, "y": 0.0},
            "goal": {"x": float(self.goal[0]), "y": float(self.goal[1])},
            "clearance": float(self.cfg.clearance),
            "edge_step": float(self.cfg.edge_step),
            "n_samples": int(self.cfg.n_samples),
            "k_neighbors": int(self.cfg.k_neighbors),
            "seed": int(self.cfg.seed),
            "path": [{"x": float(x), "y": float(y)} for (x, y) in self.path],
        }

        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[UI] saved -> {self.out_json.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", type=str, default="worlds")
    ap.add_argument("--out", type=str, default="integration/path_click.json")

    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--clearance", type=float, default=0.20)
    ap.add_argument("--edge-step", type=float, default=0.05)
    ap.add_argument("--world-limit", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--plot-edges-max", type=int, default=20000)

    ap.add_argument("--rl-model", type=str, default=None, help="Ruta al .zip de PPO")
    ap.add_argument("--rl-device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    # 👇 IMPORTANTE: ahora es d_center (centro-a-centro), como entrenamiento
    ap.add_argument("--rl-activate", type=float, default=1.2, help="Activa RL si d_center <= esto (m)")
    ap.add_argument("--rl-deactivate", type=float, default=1.4, help="Desactiva RL si d_center >= esto (m)")

    ap.add_argument("--lookahead", type=float, default=1.3, help="Lookahead nominal (m)")
    ap.add_argument("--lookahead-rl", type=float, default=0.9, help="Lookahead cuando RL activo (m)")

    # TTC gate opcional (puedes apagarlo para quedarte solo con distancia)
    ap.add_argument("--ttc-activate", type=float, default=1.2, help="Activa RL si TTC <= esto (s)")
    ap.add_argument("--no-ttc-gate", action="store_true", help="Desactiva activación por TTC (solo distancia).")

    ap.add_argument(
        "--dyn",
        type=float,
        nargs=5,
        action="append",
        metavar=("X", "Y", "R", "VX", "VY"),
        help="Agrega un obstáculo dinámico: x y r vx vy (puedes usarlo varias veces)",
    )

    args = ap.parse_args()

    worlds_dir = _resolve_worlds_dir(args.worlds_dir)
    worlds = list_worlds(worlds_dir)
    out = Path(args.out)

    cfg = PRMConfig(
        n_samples=args.n_samples,
        k_neighbors=args.k,
        clearance=args.clearance,
        edge_step=args.edge_step,
        world_limit=args.world_limit,
        seed=args.seed,
    )

    dyn_specs: List[Tuple[float, float, float, float, float]] = []
    if args.dyn:
        for d in args.dyn:
            dyn_specs.append((float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])))

    while True:
        idx = prompt_world(worlds)
        world_path = worlds[idx]
        try:
            _runner = PlotRunner(
                world_path,
                out,
                cfg,
                plot_edges_max=args.plot_edges_max,
                rl_model=args.rl_model,
                rl_device=args.rl_device,
                rl_activate=float(args.rl_activate),
                rl_deactivate=float(args.rl_deactivate),
                dyn_specs=dyn_specs,
                lookahead=float(args.lookahead),
                lookahead_rl=float(args.lookahead_rl),
                ttc_activate=float(args.ttc_activate),
                use_ttc_gate=(not bool(args.no_ttc_gate)),
            )
            import matplotlib.pyplot as plt

            plt.show()
            break
        except KeyboardInterrupt:
            print("\n[UI] reset -> volviendo a selección por terminal...\n")
            continue


if __name__ == "__main__":
    main()
