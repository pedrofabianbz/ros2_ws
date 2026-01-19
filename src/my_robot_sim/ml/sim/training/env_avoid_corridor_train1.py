from __future__ import annotations

import warnings
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..core.world import World2D
from ..core.render_utils import render_world


class AvoidCorridorTrainEnv(gym.Env):
    """
    Entrenamiento navegación + evitación (discreto) EN PASILLO RÍGIDO.

    Observación:
      - use_corridor_obs=True  -> 10D
          [0] dist_goal
          [1] cos(ang_goal)
          [2] sin(ang_goal)
          [3] d_dyn
          [4] cos(alpha)
          [5] sin(alpha)
          [6] v_closing
          [7] v_lat_rel
          [8] ct_signed_norm
          [9] wall_dist_norm
      - use_corridor_obs=False -> 8D

    Extras de seguridad:
      - CPA override: preferir steer (pasar por detrás) y frenar solo cuando toca
      - Wall guard + wall touch shaping
      - TRAP-WALL:
          * Caso A (había hueco pero te volvías lento): ESCAPE MODE => piso de velocidad si giras al centro.
          * Caso B (NO hay hueco): STOP MODE => freno total (v_cmd=0).

    NUEVO (para tu problema actual):
      - Activación temprana “rear-end” + TTC: obstáculo lento al frente => RL se activa antes.
      - Anti-conservador:
          * stall_penalty: castiga quedarte casi parado cuando no hace falta.
          * brake_penalty: castiga frenar lejos del riesgo (salvo STOP MODE).
      - pass_close_bonus: bonus suave por pasar “cerca razonable” (~0.5m centro-centro),
        solo cuando es seguro (sin cierre fuerte y no en STOP MODE).
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        sdf_world: str | None = None,
        # distribución
        p_conflict: float = 0.8,
        v_dyn_min: float = 0.15,
        v_dyn_max: float = 0.8,
        v_nom: float = 0.4,
        # (selector/compat)
        risk_radius: float = 1.0,
        world_limit: float = 10.0,
        max_steps: int = 250,
        render_mode: str | None = None,
        seed: int | None = None,
        # match con runtime
        k_rl: float = 4.0,
        # selector + histéresis
        activate_radius: float = 0.8,
        deactivate_radius: float = 1.1,
        # activar por peligro (v_closing)
        use_closing_activation: bool = True,
        closing_th: float = -0.20,
        closing_dist: float = 2.0,
        # reward base
        k_progress: float = 1.0,
        k_risk: float = 2.5,
        goal_bonus: float = 20.0,
        time_cost: float = 0.02,
        finish_bonus_fast: float = 5.0,
        # RL decide cada N steps
        rl_decimation: int = 1,
        # render zoom
        render_half_span: float = 2.5,
        # ---- control discreto de velocidad ----
        v_brake: float = 0.0,
        v_boost: float = 0.80,
        # ---- spawn rejection ----
        min_spawn_sep: float = 0.35,
        min_goal_sep: float = 0.35,
        spawn_tries: int = 50,
        # ---- mirror augmentation ----
        p_mirror: float = 0.35,
        # ---- shaping "no abrirse" ----
        k_close: float = 0.9,
        prog_clip_lo: float = -0.05,
        prog_clip_hi: float = 0.25,
        vclose_clip: float = 1.5,
        # ---- boost penalty cerca ----
        k_boost_near: float = 0.35,
        # ---- curriculum "parallel-ahead" (solo spawn / activación) ----
        p_parallel: float = 0.35,
        parallel_ahead_min: float = 0.7,
        parallel_ahead_max: float = 1.6,
        parallel_lat_min: float = -0.35,
        parallel_lat_max: float = 0.35,
        parallel_dir_noise: float = 0.25,
        parallel_speed_scale_min: float = 0.7,
        parallel_speed_scale_max: float = 1.2,
        # ---- activación RL por patrón paralelo ----
        parallel_front_th: float = 0.7,     # rad
        parallel_vclose_th: float = 0.10,   # |v_closing|
        parallel_extra_margin: float = 0.35,
        # ---- pass shaping (solo logging) ----
        k_pass: float = 2.0,
        pass_margin: float = 0.12,
        pass_gate_dist: float = 1.8,
        # ---- PASILLO ----
        corridor_half_width: float = 0.9,
        randomize_corridor: bool = True,
        corridor_hw_min: float = 0.55,
        corridor_hw_max: float = 1.20,
        # ---- OBS corridor extra ----
        use_corridor_obs: bool = True,
        # ---- shaping “pasar cerca” + pared ----
        obstacle_clearance: float = 0.20,
        wall_safe: float = 0.20,
        k_wall: float = 2.0,
        k_wall_v: float = 2.5,
        k_brake_wall: float = 0.0,
        wall_activate_dist: float = 0.35,
        hard_brake_at_wall_dist: float | None = None,
        # ---- v_lat_rel clip ----
        vlat_clip: float = 1.5,
        # ---- "lado malo" + CPA override (crossing) ----
        k_bad_side: float = 0.35,
        bad_side_dist: float = 1.6,
        bad_side_alpha_min: float = 0.35,
        bad_side_alpha_max: float = 1.25,
        bad_side_vlat_th: float = 0.15,
        bad_side_vclose_abs_th: float = 0.35,
        use_cpa_brake: bool = True,
        cpa_t_th: float = 0.85,             # s
        cpa_dist_margin: float = 0.02,      # m
        cpa_alpha_min: float = 0.35,
        cpa_alpha_max: float = 1.30,
        cpa_vlat_th: float = 0.15,
        cpa_vclose_abs_th: float = 0.40,
        # ===== FIX CPA =====
        cpa_approach_vclosing_th: float = 0.15,
        cpa_zone_dist: float = 1.35,
        cpa_prefer_steer: bool = True,
        cpa_steer_vlat_min: float = 0.12,
        cpa_steer_wall_dist_min: float = 0.18,
        cpa_trust_policy_if_steering_away: bool = True,
        # ---- SAFETY FIRST ----
        wall_guard_dist: float = 0.35,
        k_wall_guard_w: float = 3.0,
        wall_touch_th: float = 0.06,
        k_wall_touch: float = 2.0,
        wall_collision_penalty: float = 200.0,
        dyn_collision_penalty: float = 100.0,
        wall_guard_vcap_scale: float = 1.0,
        # ---- TRAP-WALL ----
        trap_wall_dist: float = 1.20,
        trap_wall_alpha_min: float = 0.35,
        trap_wall_alpha_max: float = 1.45,
        trap_wall_vlat_min: float = 0.10,
        trap_wall_wall_dist_th: float = 0.18,
        trap_wall_vcap_scale: float = 0.90,   # menos agresivo: evita volverte lento por cap
        trap_vn_th: float = 0.12,
        trap_prefer_steer: bool = True,
        trap_brake_if_no_room: bool = False,
        trap_wall_robot_dist_th: float = 0.12,
        trap_disable_boost: bool = True,
        # ---- NUEVO: Caso A (escape) ----
        escape_v_floor: float = 0.30,   # piso de v cuando estás cerca pared y giras al centro
        escape_d_gate: float = 1.10,    # si d_dyn0 < esto, activa escape
        # ---- NUEVO: Caso B (stop si no hay hueco) ----
        trap_stop_no_gap: bool = True,
        trap_stop_gap_margin: float = 0.02,

        # ==========================
        # NUEVO: activación temprana
        # ==========================
        rear_end_enable: bool = True,
        rear_end_front_th: float = 0.35,      # rad
        rear_end_dist: float = 1.8,           # m
        rear_end_vclosing_th: float = 0.05,   # m/s (si -v_closing > th => lo alcanzas)

        use_ttc_activation: bool = True,
        ttc_th: float = 4.5,                  # s
        ttc_dist_max: float = 2.5,            # m
        ttc_vclosing_min: float = 0.05,       # m/s

        # ==========================
        # NUEVO: anti-conservador
        # ==========================
        brake_penalty: float = 0.010,         # castiga frenar lejos del riesgo (no STOP MODE)
        stall_penalty: float = 0.015,         # castiga quedarte casi parado sin necesidad
        stall_v_th: float = 0.08,             # m/s
        stall_d_th: float = 1.0,              # m

        # ==========================
        # NUEVO: pasar cerca ~0.5
        # ==========================
        k_pass_close: float = 0.10,
        pass_close_target: float = 0.25,      # clearance objetivo (m). d_target = d_coll + target => ~0.50
        pass_close_width: float = 0.15,

        # ---- catch-all ----
        **_ignored_kwargs,
    ):
        super().__init__()

        if _ignored_kwargs:
            warnings.warn(
                f"AvoidCorridorTrainEnv: ignoring unknown kwargs: {sorted(_ignored_kwargs.keys())}",
                RuntimeWarning,
            )

        self.rng = np.random.default_rng(seed)

        # Robot params
        self.dt = 0.05
        self.v_max = 0.8
        self.w_max = 5.0
        self.a_lin = 0.8
        self.a_ang = 5.0
        self.robot_radius = 0.10

        # Dinámico
        self.r_dyn = 0.15
        self.v_dyn_min = float(v_dyn_min)
        self.v_dyn_max = float(v_dyn_max)

        self.p_conflict = float(p_conflict)
        self.v_nom = float(v_nom)
        self.risk_radius = float(risk_radius)
        self.world_limit = float(world_limit)
        self.max_steps = int(max_steps)

        self.k_rl = float(k_rl)

        # selector
        self.activate_radius = float(activate_radius)
        self.deactivate_radius = float(deactivate_radius)
        if self.deactivate_radius <= self.activate_radius:
            self.deactivate_radius = self.activate_radius + 0.5

        # closing activation
        self.use_closing_activation = bool(use_closing_activation)
        self.closing_th = float(closing_th)
        self.closing_dist = float(closing_dist)

        # reward base
        self.k_progress = float(k_progress)
        self.k_risk = float(k_risk)
        self.k_close = float(k_close)
        self.prog_clip_lo = float(prog_clip_lo)
        self.prog_clip_hi = float(prog_clip_hi)
        self.vclose_clip = float(max(1e-6, vclose_clip))
        self.k_boost_near = float(max(0.0, k_boost_near))

        self.time_cost = float(time_cost)
        self.goal_bonus = float(goal_bonus)
        self.finish_bonus_fast = float(finish_bonus_fast)

        # clearance + wall shaping
        self.obstacle_clearance = float(max(0.0, obstacle_clearance))
        self.wall_safe = float(max(1e-6, wall_safe))
        self.k_wall = float(max(0.0, k_wall))
        self.k_wall_v = float(max(0.0, k_wall_v))
        self.k_brake_wall = float(max(0.0, k_brake_wall))
        self.wall_activate_dist = float(max(0.0, wall_activate_dist))
        self.hard_brake_at_wall_dist = None if hard_brake_at_wall_dist is None else float(max(0.0, hard_brake_at_wall_dist))

        # v_lat_rel clip
        self.vlat_clip = float(max(1e-6, vlat_clip))

        # crossing extras
        self.k_bad_side = float(max(0.0, k_bad_side))
        self.bad_side_dist = float(max(0.05, bad_side_dist))
        self.bad_side_alpha_min = float(max(0.0, bad_side_alpha_min))
        self.bad_side_alpha_max = float(max(self.bad_side_alpha_min + 1e-3, bad_side_alpha_max))
        self.bad_side_vlat_th = float(max(0.0, bad_side_vlat_th))
        self.bad_side_vclose_abs_th = float(max(0.0, bad_side_vclose_abs_th))

        # CPA
        self.use_cpa_brake = bool(use_cpa_brake)
        self.cpa_t_th = float(max(0.05, cpa_t_th))
        self.cpa_dist_margin = float(max(0.0, cpa_dist_margin))
        self.cpa_alpha_min = float(max(0.0, cpa_alpha_min))
        self.cpa_alpha_max = float(max(self.cpa_alpha_min + 1e-3, cpa_alpha_max))
        self.cpa_vlat_th = float(max(0.0, cpa_vlat_th))
        self.cpa_vclose_abs_th = float(max(0.0, cpa_vclose_abs_th))

        # FIX CPA
        self.cpa_approach_vclosing_th = float(max(0.0, cpa_approach_vclosing_th))
        self.cpa_zone_dist = float(max(0.1, cpa_zone_dist))
        self.cpa_prefer_steer = bool(cpa_prefer_steer)
        self.cpa_steer_vlat_min = float(max(0.0, cpa_steer_vlat_min))
        self.cpa_steer_wall_dist_min = float(max(0.0, cpa_steer_wall_dist_min))
        self.cpa_trust_policy_if_steering_away = bool(cpa_trust_policy_if_steering_away)

        # SAFETY FIRST
        self.wall_guard_dist = float(max(0.0, wall_guard_dist))
        self.k_wall_guard_w = float(max(0.0, k_wall_guard_w))
        self.wall_touch_th = float(max(1e-6, wall_touch_th))
        self.k_wall_touch = float(max(0.0, k_wall_touch))
        self.wall_collision_penalty = float(max(0.0, wall_collision_penalty))
        self.dyn_collision_penalty = float(max(0.0, dyn_collision_penalty))
        self.wall_guard_vcap_scale = float(max(0.0, wall_guard_vcap_scale))

        # TRAP-WALL
        self.trap_wall_dist = float(max(0.0, trap_wall_dist))
        self.trap_wall_alpha_min = float(max(0.0, trap_wall_alpha_min))
        self.trap_wall_alpha_max = float(max(self.trap_wall_alpha_min + 1e-3, trap_wall_alpha_max))
        self.trap_wall_vlat_min = float(max(0.0, trap_wall_vlat_min))
        self.trap_wall_wall_dist_th = float(max(0.0, trap_wall_wall_dist_th))
        self.trap_wall_vcap_scale = float(max(0.0, trap_wall_vcap_scale))
        self.trap_vn_th = float(max(0.0, trap_vn_th))
        self.trap_prefer_steer = bool(trap_prefer_steer)
        self.trap_brake_if_no_room = bool(trap_brake_if_no_room)
        self.trap_wall_robot_dist_th = float(max(0.0, trap_wall_robot_dist_th))
        self.trap_disable_boost = bool(trap_disable_boost)

        # NUEVO: escape / stop
        self.escape_v_floor = float(np.clip(escape_v_floor, 0.0, self.v_max))
        self.escape_d_gate = float(max(0.1, escape_d_gate))
        self.trap_stop_no_gap = bool(trap_stop_no_gap)
        self.trap_stop_gap_margin = float(max(0.0, trap_stop_gap_margin))

        # NUEVO: rear-end + TTC activation
        self.rear_end_enable = bool(rear_end_enable)
        self.rear_end_front_th = float(max(0.0, rear_end_front_th))
        self.rear_end_dist = float(max(0.1, rear_end_dist))
        self.rear_end_vclosing_th = float(max(0.0, rear_end_vclosing_th))

        self.use_ttc_activation = bool(use_ttc_activation)
        self.ttc_th = float(max(0.1, ttc_th))
        self.ttc_dist_max = float(max(0.1, ttc_dist_max))
        self.ttc_vclosing_min = float(max(0.0, ttc_vclosing_min))

        # NUEVO: anti-conservador + pass-close
        self.brake_penalty = float(max(0.0, brake_penalty))
        self.stall_penalty = float(max(0.0, stall_penalty))
        self.stall_v_th = float(max(0.0, stall_v_th))
        self.stall_d_th = float(max(0.1, stall_d_th))

        self.k_pass_close = float(max(0.0, k_pass_close))
        self.pass_close_target = float(max(0.0, pass_close_target))
        self.pass_close_width = float(max(1e-6, pass_close_width))

        # RL decimation
        self.rl_decimation = int(max(1, rl_decimation))

        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        self.render_half_span = float(render_half_span)

        # velocidad discreta
        self.v_brake = float(np.clip(v_brake, 0.0, self.v_max))
        self.v_boost = float(np.clip(v_boost, 0.0, self.v_max))
        if self.v_boost < self.v_brake:
            self.v_boost, self.v_brake = self.v_brake, self.v_boost

        # spawn rejection
        self.min_spawn_sep = float(min_spawn_sep)
        self.min_goal_sep = float(min_goal_sep)
        self.spawn_tries = int(max(1, spawn_tries))

        # mirror
        self.p_mirror = float(np.clip(p_mirror, 0.0, 1.0))
        self._mirrored = False

        # curriculum parallel-ahead
        self.p_parallel = float(np.clip(p_parallel, 0.0, 1.0))
        self.parallel_ahead_min = float(parallel_ahead_min)
        self.parallel_ahead_max = float(parallel_ahead_max)
        self.parallel_lat_min = float(parallel_lat_min)
        self.parallel_lat_max = float(parallel_lat_max)
        self.parallel_dir_noise = float(max(0.0, parallel_dir_noise))
        self.parallel_speed_scale_min = float(parallel_speed_scale_min)
        self.parallel_speed_scale_max = float(parallel_speed_scale_max)

        # activación RL por patrón paralelo
        self.parallel_front_th = float(max(0.0, parallel_front_th))
        self.parallel_vclose_th = float(max(0.0, parallel_vclose_th))
        self.parallel_extra_margin = float(max(0.0, parallel_extra_margin))

        # pass detection (logging)
        self.k_pass = float(max(0.0, k_pass))
        self.pass_margin = float(max(0.0, pass_margin))
        self.pass_gate_dist = float(max(0.1, pass_gate_dist))

        self._s_forward_prev = 0.0

        # PASILLO
        self.corridor_half_width = float(max(0.2, corridor_half_width))
        self.randomize_corridor = bool(randomize_corridor)
        self.corridor_hw_min = float(max(0.25, corridor_hw_min))
        self.corridor_hw_max = float(max(self.corridor_hw_min + 1e-3, corridor_hw_max))

        # OBS corridor extra
        self.use_corridor_obs = bool(use_corridor_obs)

        # basis del pasillo
        self._corr_tx = 1.0
        self._corr_ty = 0.0
        self._corr_nx = 0.0
        self._corr_ny = 1.0
        self._corr_ox = 0.0
        self._corr_oy = 0.0

        # Mundo estático desde SDF
        self.world = World2D(sdf_path=sdf_world)

        # 7 acciones
        self.action_space = spaces.Discrete(7)

        # Observation space (8D o 10D)
        if self.use_corridor_obs:
            low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -5.0, -5.0, -1.0, 0.0], dtype=np.float32)
            high = np.array([50.0, 1.0, 1.0, 50.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -5.0, -5.0], dtype=np.float32)
            high = np.array([50.0, 1.0, 1.0, 50.0, 1.0, 1.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    # ---------------- utils ----------------
    def _wrap(self, ang: float) -> float:
        return float(np.arctan2(np.sin(ang), np.cos(ang)))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _mirror_action(self, a: int) -> int:
        a = int(a)
        if a == 1:
            return 2
        if a == 2:
            return 1
        if a == 3:
            return 4
        if a == 4:
            return 3
        return a

    def _canonicalize_obs(self, obs_world: np.ndarray) -> np.ndarray:
        if not self._mirrored:
            return obs_world
        obs = obs_world.copy()
        obs[2] = -obs[2]  # sin(ang_goal)
        obs[5] = -obs[5]  # sin(alpha)
        if obs.shape[0] >= 8:
            obs[7] = -obs[7]  # v_lat_rel
        if obs.shape[0] >= 10:
            obs[8] = -obs[8]  # ct_signed_norm
        return obs

    def _build_corridor_basis(self):
        dx = float(self.gx - self.x0)
        dy = float(self.gy - self.y0)
        nrm = float(np.hypot(dx, dy))
        if nrm < 1e-6:
            self._corr_tx, self._corr_ty = 1.0, 0.0
        else:
            self._corr_tx, self._corr_ty = dx / nrm, dy / nrm
        self._corr_nx, self._corr_ny = -self._corr_ty, self._corr_tx
        self._corr_ox, self._corr_oy = float(self.x0), float(self.y0)

    def _corridor_signed_dist(self, px: float, py: float) -> float:
        dx = float(px - self._corr_ox)
        dy = float(py - self._corr_oy)
        return dx * self._corr_nx + dy * self._corr_ny

    def _corridor_push_and_bounce_circle(self, px: float, py: float, vx: float, vy: float, radius: float):
        half = self.corridor_half_width
        d = self._corridor_signed_dist(px, py)
        limit = half - radius
        if limit <= 1e-6:
            return px, py, vx, vy
        if abs(d) <= limit:
            return px, py, vx, vy

        s = 1.0 if d > 0.0 else -1.0
        excess = abs(d) - limit

        px = float(px - s * excess * self._corr_nx)
        py = float(py - s * excess * self._corr_ny)

        vn = float(vx * self._corr_nx + vy * self._corr_ny)
        vx = float(vx - 2.0 * vn * self._corr_nx)
        vy = float(vy - 2.0 * vn * self._corr_ny)
        return px, py, vx, vy

    def _corridor_features_world(self) -> tuple[float, float, float, float]:
        signed_ct = float(self._corridor_signed_dist(self.x, self.y))
        wall_limit = float(max(self.corridor_half_width - self.robot_radius, 1e-6))
        ct_signed_norm = float(np.clip(signed_ct / wall_limit, -1.0, 1.0))
        wall_dist = float(np.clip(wall_limit - abs(signed_ct), 0.0, wall_limit))
        wall_dist_norm = float(np.clip(wall_dist / wall_limit, 0.0, 1.0))
        return signed_ct, ct_signed_norm, wall_dist, wall_dist_norm

    def _s_forward(self) -> float:
        fx = float(np.cos(self.theta))
        fy = float(np.sin(self.theta))
        dx = float(self.cx - self.x)
        dy = float(self.cy - self.y)
        return dx * fx + dy * fy

    def _action_to_wcorr(self, a: int) -> float:
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

    def _action_to_vcmd(self, a: int, v_nominal: float) -> float:
        if a == 5:
            return float(self.v_brake)
        if a == 6:
            return float(self.v_boost)
        return float(np.clip(v_nominal, 0.0, self.v_max))

    def _nominal_controller(self) -> tuple[float, float, float]:
        dx = self.gx - self.x
        dy = self.gy - self.y
        ang_goal_world = float(np.arctan2(dy, dx))
        ang_goal = self._wrap(ang_goal_world - self.theta)

        k_w_nom = 1.8
        w_nom = self._clip(k_w_nom * ang_goal, -self.w_max, self.w_max)
        v_cmd = self._clip(self.v_nom, 0.0, self.v_max)
        return v_cmd, w_nom, ang_goal

    # ---------------- scenario spawning ----------------
    def _spawn_clock_scenario(self):
        cx0, cy0 = 0.0, 0.0
        R = 1.0

        phi = float(self.rng.uniform(-np.pi, np.pi))
        self.x = cx0 + R * float(np.cos(phi))
        self.y = cy0 + R * float(np.sin(phi))
        self.gx = cx0 - R * float(np.cos(phi))
        self.gy = cy0 - R * float(np.sin(phi))

        self.theta = float(np.arctan2(self.gy - self.y, self.gx - self.x))
        self.v = 0.0
        self.w = 0.0

        t_r = 1.0 / max(self.v_nom, 1e-3)

        best = None
        best_score = -1.0

        for _ in range(self.spawn_tries):
            psi = float(self.rng.uniform(-np.pi, np.pi))
            vdyn = float(self.rng.uniform(self.v_dyn_min, self.v_dyn_max))
            vx = vdyn * float(np.cos(psi))
            vy = vdyn * float(np.sin(psi))

            dirx = float(np.cos(psi))
            diry = float(np.sin(psi))

            conflict = (float(self.rng.uniform(0.0, 1.0)) < self.p_conflict)

            if conflict:
                d0 = vdyn * t_r * float(self.rng.uniform(0.9, 1.1))
                cx = cx0 - dirx * d0
                cy = cy0 - diry * d0
            else:
                dt_offset = float(self.rng.uniform(-1.5, 1.5))
                d0 = vdyn * max(t_r + dt_offset, 0.1)
                cx = cx0 - dirx * d0
                cy = cy0 - diry * d0

            d_robot = float(np.hypot(cx - self.x, cy - self.y))
            d_goal = float(np.hypot(cx - self.gx, cy - self.gy))

            ok = (d_robot >= self.min_spawn_sep) and (d_goal >= self.min_goal_sep)
            score = min(d_robot, d_goal)

            if ok:
                self.cx, self.cy = float(cx), float(cy)
                self.vx, self.vy = float(vx), float(vy)
                self._conflict = bool(conflict)
                self._scenario = "clock"
                return

            if score > best_score:
                best_score = score
                best = (cx, cy, vx, vy, conflict)

        cx, cy, vx, vy, conflict = best
        self.cx, self.cy = float(cx), float(cy)
        self.vx, self.vy = float(vx), float(vy)
        self._conflict = bool(conflict)
        self._scenario = "clock"

    def _spawn_parallel_scenario(self) -> bool:
        cx0, cy0 = 0.0, 0.0
        R = 1.0

        phi = float(self.rng.uniform(-np.pi, np.pi))
        self.x = cx0 + R * float(np.cos(phi))
        self.y = cy0 + R * float(np.sin(phi))
        self.gx = cx0 - R * float(np.cos(phi))
        self.gy = cy0 - R * float(np.sin(phi))

        self.theta = float(np.arctan2(self.gy - self.y, self.gx - self.x))
        self.v = 0.0
        self.w = 0.0

        ahead = float(self.rng.uniform(self.parallel_ahead_min, self.parallel_ahead_max))
        lat = float(self.rng.uniform(self.parallel_lat_min, self.parallel_lat_max))

        fx = float(np.cos(self.theta))
        fy = float(np.sin(self.theta))
        lx = -fy
        ly = fx

        cx = self.x + ahead * fx + lat * lx
        cy = self.y + ahead * fy + lat * ly

        dir_noise = float(self.rng.uniform(-self.parallel_dir_noise, self.parallel_dir_noise))
        psi = float(self.theta + dir_noise)

        v_scale = float(self.rng.uniform(self.parallel_speed_scale_min, self.parallel_speed_scale_max))
        vdyn = float(np.clip(v_scale * self.v_nom, self.v_dyn_min, self.v_dyn_max))

        vx = vdyn * float(np.cos(psi))
        vy = vdyn * float(np.sin(psi))

        d_robot = float(np.hypot(cx - self.x, cy - self.y))
        d_goal = float(np.hypot(cx - self.gx, cy - self.gy))
        ok = (d_robot >= self.min_spawn_sep) and (d_goal >= self.min_goal_sep)

        if ok:
            self.cx, self.cy = float(cx), float(cy)
            self.vx, self.vy = float(vx), float(vy)
            self._conflict = True
            self._scenario = "parallel"
            return True
        return False

    def _spawn_scenario(self):
        if float(self.rng.uniform(0.0, 1.0)) < self.p_parallel:
            if self._spawn_parallel_scenario():
                return
        self._spawn_clock_scenario()

    def _maybe_mirror(self):
        self._mirrored = (float(self.rng.uniform(0.0, 1.0)) < self.p_mirror)
        if not self._mirrored:
            return

        self.y = -self.y
        self.gy = -self.gy
        self.cy = -self.cy
        self.vy = -self.vy

        self.theta = self._wrap(-self.theta)
        self.w = -self.w

    def _update_dynamic(self):
        self.cx += self.vx * self.dt
        self.cy += self.vy * self.dt

        self.cx, self.cy, self.vx, self.vy = self._corridor_push_and_bounce_circle(
            self.cx, self.cy, self.vx, self.vy, self.r_dyn
        )

        L = self.world_limit
        if self.cx < -L or self.cx > L:
            self.vx *= -1.0
            self.cx = float(np.clip(self.cx, -L, L))
        if self.cy < -L or self.cy > L:
            self.vy *= -1.0
            self.cy = float(np.clip(self.cy, -L, L))

    # ---------------- obs ----------------
    def _get_obs_world(self) -> np.ndarray:
        dgx = self.gx - self.x
        dgy = self.gy - self.y
        dist_goal = float(np.hypot(dgx, dgy))
        ang_goal_world = float(np.arctan2(dgy, dgx))
        ang_goal = self._wrap(ang_goal_world - self.theta)

        dx = self.cx - self.x
        dy = self.cy - self.y
        d_dyn = float(np.hypot(dx, dy))
        bearing_world = float(np.arctan2(dy, dx))
        alpha = self._wrap(bearing_world - self.theta)

        v_rx = self.v * np.cos(self.theta)
        v_ry = self.v * np.sin(self.theta)
        rvx = self.vx - v_rx
        rvy = self.vy - v_ry

        if d_dyn < 1e-6:
            v_closing = 0.0
        else:
            v_closing = float((rvx * dx + rvy * dy) / d_dyn)

        lx = -float(np.sin(self.theta))
        ly = +float(np.cos(self.theta))
        v_lat_rel = float(rvx * lx + rvy * ly)
        v_lat_rel = float(np.clip(v_lat_rel, -self.vlat_clip, self.vlat_clip))

        base = np.array(
            [
                dist_goal, np.cos(ang_goal), np.sin(ang_goal),
                d_dyn, np.cos(alpha), np.sin(alpha),
                v_closing, v_lat_rel
            ],
            dtype=np.float32,
        )

        if not self.use_corridor_obs:
            return base

        _, ct_signed_norm, _, wall_dist_norm = self._corridor_features_world()
        extra = np.array([ct_signed_norm, wall_dist_norm], dtype=np.float32)
        return np.concatenate([base, extra], axis=0)

    def _get_obs(self) -> np.ndarray:
        return self._canonicalize_obs(self._get_obs_world())

    # ---------------- selector (histeresis) ----------------
    def _update_rl_active(self, d_dyn: float, v_closing: float, alpha: float, wall_dist: float):
        # TTC activation (robusta)
        danger_ttc = False
        if self.use_ttc_activation:
            closing_speed = float(max(-v_closing, 0.0))  # >0 si te acercas
            if (closing_speed > self.ttc_vclosing_min) and (d_dyn < self.ttc_dist_max):
                ttc = d_dyn / max(closing_speed, 1e-6)
                danger_ttc = (ttc < self.ttc_th)

        # closing activation original
        danger_closing = self.use_closing_activation and (v_closing < self.closing_th) and (d_dyn < self.closing_dist)

        # rear-end: obstáculo lento al frente que vas a alcanzar
        danger_rear_end = False
        if self.rear_end_enable:
            frontish = (abs(alpha) < self.rear_end_front_th)
            closing = (-v_closing) > self.rear_end_vclosing_th
            danger_rear_end = frontish and closing and (d_dyn < self.rear_end_dist)

        # paralelo (tu lógica)
        front = (abs(alpha) < self.parallel_front_th)
        parallelish = (abs(v_closing) < self.parallel_vclose_th)
        d_parallel_gate = min(self.risk_radius + 0.25, self.activate_radius + self.parallel_extra_margin)
        danger_parallel = (d_dyn < d_parallel_gate) and front and parallelish

        # pared
        danger_wall = (wall_dist < self.wall_activate_dist)

        danger = bool(danger_closing or danger_ttc or danger_rear_end or danger_parallel or danger_wall)

        if self.rl_active:
            if d_dyn > self.deactivate_radius and not danger_wall:
                self.rl_active = False
        else:
            if (d_dyn < self.activate_radius) or danger:
                self.rl_active = True

    # ---------------- CPA ----------------
    def _compute_cpa(self) -> tuple[float, float]:
        dx = float(self.cx - self.x)
        dy = float(self.cy - self.y)

        v_rx = float(self.v * np.cos(self.theta))
        v_ry = float(self.v * np.sin(self.theta))
        rvx = float(self.vx - v_rx)
        rvy = float(self.vy - v_ry)

        vv = rvx * rvx + rvy * rvy
        if vv < 1e-9:
            return 0.0, float(np.hypot(dx, dy))

        t = -(dx * rvx + dy * rvy) / vv
        t = float(np.clip(t, 0.0, self.cpa_t_th))
        px = dx + rvx * t
        py = dy + rvy * t
        return t, float(np.hypot(px, py))

    def _is_crossing_zone(
        self,
        d_dyn: float,
        alpha: float,
        v_closing: float,
        v_lat_rel: float,
        dist_th: float,
        a_min: float,
        a_max: float,
        vlat_th: float,
        vclose_abs_th: float,
    ) -> bool:
        if d_dyn > dist_th:
            return False
        if not (abs(alpha) >= a_min and abs(alpha) <= a_max):
            return False
        if abs(v_lat_rel) < vlat_th:
            return False
        if abs(v_closing) > vclose_abs_th:
            return False
        return True

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        super().reset(seed=seed)
        self.t = 0

        self._spawn_scenario()
        self._maybe_mirror()

        self.x0 = float(self.x)
        self.y0 = float(self.y)

        if self.randomize_corridor:
            hw = float(self.rng.uniform(self.corridor_hw_min, self.corridor_hw_max))
            hw = max(hw, self.robot_radius + 0.12)
            self.corridor_half_width = hw

        self._build_corridor_basis()

        self.cx, self.cy, self.vx, self.vy = self._corridor_push_and_bounce_circle(
            self.cx, self.cy, self.vx, self.vy, self.r_dyn
        )

        self.prev_dist_goal = float(np.hypot(self.gx - self.x, self.gy - self.y))

        self.rl_active = False
        self._hold_action = 0

        self.v = 0.0
        self.w = 0.0
        self.v_set = float(np.clip(self.v_nom, 0.0, self.v_max))
        self.v_set_prev = self.v_set

        self.world.dynamic_circles = [(self.cx, self.cy, self.r_dyn, self.vx, self.vy)]
        self._s_forward_prev = float(self._s_forward())

        return self._get_obs(), {
            "conflict": bool(getattr(self, "_conflict", False)),
            "mirrored": bool(self._mirrored),
            "scenario": str(getattr(self, "_scenario", "unknown")),
            "corridor_half_width": float(self.corridor_half_width),
            "randomize_corridor": bool(self.randomize_corridor),
            "use_corridor_obs": bool(self.use_corridor_obs),
            "reward_profile": "round3_ttc_rearend_anti_brake_stall_passclose_escape_stop",
        }

    def step(self, action: int):
        action_raw = int(action)
        s_prev = float(self._s_forward_prev)

        signed_ct0, ct_signed_norm0, wall_dist0, wall_dist_norm0 = self._corridor_features_world()

        obs0 = self._get_obs()
        d_dyn0 = float(obs0[3])
        v_closing0 = float(obs0[6])
        v_lat_rel0 = float(obs0[7])
        alpha0 = float(np.arctan2(obs0[5], obs0[4]))

        # convenience: distancias base
        d_coll0 = float(self.robot_radius + self.r_dyn)
        d_safe0 = float(d_coll0 + self.obstacle_clearance)

        self._update_rl_active(d_dyn0, v_closing0, alpha0, wall_dist0)

        if self.rl_decimation > 1:
            if (self.t % self.rl_decimation) == 0:
                self._hold_action = action_raw
            action_hold = int(self._hold_action)
        else:
            action_hold = action_raw

        action_exec = self._mirror_action(action_hold) if self._mirrored else action_hold

        # overrides
        override_action = False
        override_brake = False
        override_reason = "none"

        stop_no_gap = False

        # --- hard safety por pared ---
        if self.hard_brake_at_wall_dist is not None and wall_dist0 < self.hard_brake_at_wall_dist:
            action_exec = 5
            override_action = True
            override_brake = True
            override_reason = "wall_hard_brake"

        # ===== CPA FIX =====
        cpa_t, cpa_d = 0.0, float("nan")
        cpa_override = False
        cpa_mode = "none"
        if (not override_action) and self.use_cpa_brake:
            cpa_t, cpa_d = self._compute_cpa()
            cpa_th = float(d_safe0 + self.cpa_dist_margin)

            crossing_zone = self._is_crossing_zone(
                d_dyn=d_dyn0,
                alpha=alpha0,
                v_closing=v_closing0,
                v_lat_rel=v_lat_rel0,
                dist_th=self.cpa_zone_dist,
                a_min=self.cpa_alpha_min,
                a_max=self.cpa_alpha_max,
                vlat_th=self.cpa_vlat_th,
                vclose_abs_th=self.cpa_vclose_abs_th,
            )

            approach = (v_closing0 < -float(self.cpa_approach_vclosing_th))
            hazard = bool(crossing_zone and approach and (cpa_t <= self.cpa_t_th) and (cpa_d <= cpa_th))

            if hazard:
                wc = float(self._action_to_wcorr(action_exec))
                steering_away = False
                if abs(v_lat_rel0) > 1e-6 and wc != 0.0:
                    steering_away = (np.sign(wc) != np.sign(v_lat_rel0)) and (int(action_exec) != 5)

                if self.cpa_trust_policy_if_steering_away and steering_away:
                    pass
                else:
                    can_steer = (
                        self.cpa_prefer_steer
                        and (abs(v_lat_rel0) >= self.cpa_steer_vlat_min)
                        and (wall_dist0 >= self.cpa_steer_wall_dist_min)
                    )

                    if can_steer:
                        steer_action = 4 if v_lat_rel0 > 0.0 else 3
                        action_exec = int(steer_action)
                        override_action = True
                        cpa_override = True
                        cpa_mode = "steer"
                        override_reason = "cpa_steer"
                    else:
                        action_exec = 5
                        override_action = True
                        override_brake = True
                        cpa_override = True
                        cpa_mode = "brake"
                        override_reason = "cpa_brake"

        # ===== TRAP-WALL =====
        trap_active = False
        trap_override = False
        trap_mode = "none"   # steer | brake | stop | none

        signed_ct_obs = float(self._corridor_signed_dist(self.cx, self.cy))

        wall_limit_obs = float(max(self.corridor_half_width - self.r_dyn, 1e-6))
        wall_dist_obs = float(np.clip(wall_limit_obs - abs(signed_ct_obs), 0.0, wall_limit_obs))

        v_n_obs = float(self.vx * self._corr_nx + self.vy * self._corr_ny)
        toward_wall = (np.sign(signed_ct_obs) * v_n_obs) > self.trap_vn_th

        same_side = (abs(np.sign(signed_ct0)) > 0.0) and (np.sign(signed_ct0) == np.sign(signed_ct_obs))

        trap_crossing_zone = self._is_crossing_zone(
            d_dyn=d_dyn0,
            alpha=alpha0,
            v_closing=v_closing0,
            v_lat_rel=v_lat_rel0,
            dist_th=self.trap_wall_dist,
            a_min=self.trap_wall_alpha_min,
            a_max=self.trap_wall_alpha_max,
            vlat_th=self.trap_wall_vlat_min,
            vclose_abs_th=999.0,
        )

        # "gap" disponible entre obstáculo y pared (aprox)
        required_gap = float(self.robot_radius + self.obstacle_clearance + self.trap_stop_gap_margin)
        no_gap = (wall_dist_obs < required_gap)

        hazard_trap = (
            (not override_action)
            and same_side
            and trap_crossing_zone
            and (wall_dist_obs < self.trap_wall_wall_dist_th)
            and toward_wall
            and (wall_dist0 < (self.trap_wall_wall_dist_th + 0.06))
        )

        if hazard_trap:
            trap_active = True

            # Caso B: si NO hay hueco, STOP total (seguridad)
            if self.trap_stop_no_gap and no_gap:
                action_exec = 5
                override_action = True
                override_brake = True
                stop_no_gap = True
                trap_override = True
                trap_mode = "stop"
                override_reason = "trap_wall_stop_no_gap"
            else:
                # Caso A: sí hay hueco -> steer al centro (escape-floor evita volverte lento)
                desired = 4 if signed_ct0 > 0.0 else 3  # hacia el centro
                wc_des = float(self._action_to_wcorr(desired))
                wc_now = float(self._action_to_wcorr(action_exec))

                already_steering_to_center = (wc_now != 0.0) and (np.sign(wc_now) == np.sign(wc_des)) and (int(action_exec) != 5)

                if self.trap_prefer_steer and (not already_steering_to_center) and (int(action_exec) != int(desired)):
                    action_exec = int(desired)
                    trap_override = True
                    trap_mode = "steer"
                    override_action = True
                    override_reason = "trap_wall_steer_to_center"
                else:
                    no_room = (wall_dist0 < self.trap_wall_robot_dist_th)
                    if self.trap_brake_if_no_room and no_room and (int(action_exec) != 5):
                        action_exec = 5
                        trap_override = True
                        trap_mode = "brake"
                        override_action = True
                        override_brake = True
                        override_reason = "trap_wall_brake"

        # --- wall guard (w extra hacia el centro) ---
        w_wall = 0.0
        wall_guard_active = False
        if self.wall_guard_dist > 0.0 and wall_dist0 < self.wall_guard_dist:
            wall_guard_active = True
            s = float(np.sign(signed_ct0))
            if abs(s) < 1e-6:
                s = 0.0
            strength = float(np.clip((self.wall_guard_dist - wall_dist0) / self.wall_guard_dist, 0.0, 1.0))
            w_wall = float(-s * self.k_wall_guard_w * strength)

            v_scale = float(np.clip(self.v / max(self.v_max, 1e-6), 0.0, 1.0))
            w_wall *= (0.25 + 0.75 * v_scale)

        self.t += 1

        v_nom_cmd, w_nom, _ = self._nominal_controller()

        if self.rl_active:
            w_cmd = w_nom + self.k_rl * self._action_to_wcorr(action_exec)
            v_cmd = self._action_to_vcmd(action_exec, v_nom_cmd)
        else:
            w_cmd = w_nom
            v_cmd = v_nom_cmd

        # STOP MODE (Caso B): freno total real
        if stop_no_gap:
            v_cmd = 0.0

        # TRAP: prohibe boost + cap
        trap_vcap_active = False
        if (not stop_no_gap) and trap_active and self.trap_disable_boost and int(action_exec) == 6:
            v_cmd = v_nom_cmd

        if (not stop_no_gap) and trap_active and self.trap_wall_vcap_scale > 0.0:
            v_cap = float(np.clip(self.trap_wall_vcap_scale * self.v_nom, 0.0, self.v_max))
            v_cmd = float(min(v_cmd, v_cap))
            trap_vcap_active = True

        # cap de velocidad con wall_guard (pero luego escape-floor puede “rescatar”)
        if (not stop_no_gap) and wall_guard_active and self.wall_guard_vcap_scale > 0.0:
            v_cap = float(np.clip(self.wall_guard_vcap_scale * self.v_nom, 0.0, self.v_max))
            v_cmd = float(min(v_cmd, v_cap))

        # ===== ESCAPE MODE (Caso A) =====
        escape_active = False
        if (not stop_no_gap) and (wall_guard_active or (wall_dist0 < self.wall_guard_dist)) and (d_dyn0 < self.escape_d_gate):
            desired_center = 4 if signed_ct0 > 0.0 else 3
            wc_des = float(self._action_to_wcorr(desired_center))
            wc_now = float(self._action_to_wcorr(action_exec))
            steering_to_center = (wc_now != 0.0) and (np.sign(wc_now) == np.sign(wc_des)) and (int(action_exec) != 5)

            if steering_to_center and (not override_brake):
                v_cmd = float(max(v_cmd, self.escape_v_floor))
                escape_active = True

        w_cmd = float(np.clip(w_cmd + w_wall, -self.w_max, self.w_max))

        self.v_set_prev = getattr(self, "v_set", 0.0)
        self.v_set = float(v_cmd)

        self.v += np.clip(v_cmd - self.v, -self.a_lin * self.dt, self.a_lin * self.dt)
        self.w += np.clip(w_cmd - self.w, -self.a_ang * self.dt, self.a_ang * self.dt)
        self.v = float(np.clip(self.v, 0.0, self.v_max))
        self.w = float(np.clip(self.w, -self.w_max, self.w_max))

        # kinematics
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta = self._wrap(self.theta + self.w * self.dt)

        # dinámico
        self._update_dynamic()
        self.world.dynamic_circles = [(self.cx, self.cy, self.r_dyn, self.vx, self.vy)]

        # distancias post-step
        d_dyn = float(np.hypot(self.cx - self.x, self.cy - self.y))
        collision_dyn = d_dyn <= (self.robot_radius + self.r_dyn)

        signed_ct, ct_signed_norm, wall_dist, wall_dist_norm = self._corridor_features_world()
        wall_limit = float(max(self.corridor_half_width - self.robot_radius, 1e-6))
        collision_wall = (abs(signed_ct) > wall_limit)

        dist_goal = float(np.hypot(self.gx - self.x, self.gy - self.y))
        reached = dist_goal < 0.25

        # pass detection (logging)
        s_now = float(self._s_forward())
        self._s_forward_prev = s_now
        passed = False
        if self.k_pass > 0.0:
            d_gate = float(min(d_dyn0, d_dyn))
            if d_gate < self.pass_gate_dist:
                m = self.pass_margin
                if (s_prev > +m) and (s_now < -m):
                    passed = True

        # -------- reward --------
        reward = 0.0
        reward -= self.time_cost

        # anti-stall (si estás casi parado, obstáculo lejos, y aún no llegas)
        stall_pen = 0.0
        if self.stall_penalty > 0.0:
            if (self.v < self.stall_v_th) and (d_dyn0 > self.stall_d_th) and (dist_goal > 0.40):
                stall_pen = self.stall_penalty
                reward -= stall_pen

        prog = self.prev_dist_goal - dist_goal
        prog_c = float(np.clip(prog, self.prog_clip_lo, self.prog_clip_hi))
        reward += self.k_progress * prog_c
        self.prev_dist_goal = dist_goal

        # obstáculo: clearance
        d_coll = float(self.robot_radius + self.r_dyn)
        d_safe = float(d_coll + self.obstacle_clearance)

        risk = 0.0
        close_pen = 0.0
        boost_pen = 0.0

        if d_dyn < d_safe:
            risk = float((d_safe - d_dyn) / max(d_safe, 1e-6))
            reward -= self.k_risk * (risk**2)

            close = float(np.clip(-v_closing0, 0.0, self.vclose_clip) / self.vclose_clip)
            close_pen = close * risk
            reward -= self.k_close * close_pen

            if self.rl_active and int(action_hold) == 6 and self.k_boost_near > 0.0:
                boost_pen = self.k_boost_near * risk
                reward -= boost_pen

        # anti-brake: castiga frenar “de gratis” (pero NO cuando es STOP MODE)
        brake_pen = 0.0
        if self.brake_penalty > 0.0:
            if (int(action_exec) == 5) and (not stop_no_gap) and (d_dyn0 > d_safe0 + 0.10):
                brake_pen = self.brake_penalty
                reward -= brake_pen

        # bonus suave por pasar cerca razonable (~0.5m) cuando es seguro
        pass_close_bonus = 0.0
        if self.k_pass_close > 0.0:
            d_target = d_coll + self.pass_close_target  # ~0.25 + 0.25 = 0.50
            # gate: no en STOP MODE, no cerrando fuerte, y dentro de zona “de interacción”
            if (not stop_no_gap) and (d_dyn < 1.2) and (abs(v_closing0) < 0.25):
                x = abs(d_dyn - d_target) / self.pass_close_width
                pass_close_bonus = self.k_pass_close * float(np.clip(1.0 - x, 0.0, 1.0))
                reward += pass_close_bonus

        # pared
        wall_risk = 0.0
        wall_pen = 0.0
        wall_v_pen = 0.0
        brake_wall_bonus = 0.0

        if wall_dist < self.wall_safe:
            wall_risk = float((self.wall_safe - wall_dist) / self.wall_safe)
            wall_pen = self.k_wall * (wall_risk**2)
            reward -= wall_pen

            wall_v_pen = self.k_wall_v * wall_risk * ((self.v / self.v_max) ** 2)
            reward -= wall_v_pen

            if int(action_hold) == 5 and self.k_brake_wall > 0.0:
                brake_wall_bonus = self.k_brake_wall * wall_risk
                reward += brake_wall_bonus

        # wall touch shaping
        wall_touch_pen = 0.0
        if self.k_wall_touch > 0.0 and wall_dist < self.wall_touch_th:
            x = float((self.wall_touch_th - wall_dist) / self.wall_touch_th)
            wall_touch_pen = self.k_wall_touch * (x * x)
            reward -= wall_touch_pen

        # lado malo (crossing)
        bad_side_pen = 0.0
        is_crossing = self._is_crossing_zone(
            d_dyn=d_dyn0,
            alpha=alpha0,
            v_closing=v_closing0,
            v_lat_rel=v_lat_rel0,
            dist_th=self.bad_side_dist,
            a_min=self.bad_side_alpha_min,
            a_max=self.bad_side_alpha_max,
            vlat_th=self.bad_side_vlat_th,
            vclose_abs_th=self.bad_side_vclose_abs_th,
        )
        if self.k_bad_side > 0.0 and is_crossing:
            wc2 = self._action_to_wcorr(action_exec)
            toward_motion = (wc2 != 0.0) and (np.sign(wc2) == np.sign(v_lat_rel0))
            if toward_motion and int(action_exec) != 5:
                w = float(np.clip((self.bad_side_dist - d_dyn0) / max(self.bad_side_dist, 1e-6), 0.0, 1.0))
                bad_side_pen = self.k_bad_side * (w**2)
                reward -= bad_side_pen

        # goal
        if reached:
            fast_bonus = self.finish_bonus_fast * (1.0 - (self.t / max(1, self.max_steps)))
            reward += self.goal_bonus + fast_bonus

        # terminal
        collision = bool(collision_dyn or collision_wall)
        if collision:
            if collision_wall:
                reward = -float(self.wall_collision_penalty)
            else:
                reward = -float(self.dyn_collision_penalty)

        terminated = bool(collision or reached)
        truncated = bool(self.t >= self.max_steps)

        # logging proxies
        step_dist = abs(self.v) * self.dt
        vx_line = float(self.gx - self.x0)
        vy_line = float(self.gy - self.y0)
        den = float(np.hypot(vx_line, vy_line)) + 1e-9
        cross_track = abs(vy_line * (self.x - self.x0) - vx_line * (self.y - self.y0)) / den
        den_r = max(self.deactivate_radius - self.activate_radius, 1e-6)
        alpha_detour = float(np.clip((d_dyn - self.activate_radius) / den_r, 0.0, 1.0))

        info = {
            "scenario": str(getattr(self, "_scenario", "unknown")),
            "d_dyn": float(d_dyn),
            "dist_goal": float(dist_goal),
            "prog": float(prog),
            "prog_c": float(prog_c),
            "collision": bool(collision),
            "collision_dyn": bool(collision_dyn),
            "collision_wall": bool(collision_wall),
            "reached": bool(reached),
            "conflict": bool(getattr(self, "_conflict", False)),
            "mirrored": bool(self._mirrored),

            "rl_active": bool(self.rl_active),
            "activate_radius": float(self.activate_radius),
            "deactivate_radius": float(self.deactivate_radius),
            "use_closing_activation": bool(self.use_closing_activation),
            "v_closing0": float(v_closing0),
            "v_lat_rel0": float(v_lat_rel0),
            "alpha0": float(alpha0),
            "wall_activate_dist": float(self.wall_activate_dist),

            "rl_decimation": int(self.rl_decimation),
            "time_cost": float(self.time_cost),
            "step_dist": float(step_dist),

            "action_raw": int(action_raw),
            "action_hold": int(action_hold),
            "action_exec": int(action_exec),
            "action": int(action_exec),

            "override_action": bool(override_action),
            "override_brake": bool(override_brake),
            "override_reason": str(override_reason),

            "cpa_t": float(cpa_t),
            "cpa_d": float(cpa_d),
            "cpa_override": bool(cpa_override),
            "cpa_mode": str(cpa_mode),

            # TRAP logging
            "trap_active": bool(trap_active),
            "trap_override": bool(trap_override),
            "trap_mode": str(trap_mode),
            "trap_vcap_active": bool(trap_vcap_active),
            "stop_no_gap": bool(stop_no_gap),
            "required_gap": float(required_gap),
            "no_gap": bool(no_gap),

            "signed_ct0": float(signed_ct0),
            "signed_ct_obs": float(signed_ct_obs),
            "wall_dist0": float(wall_dist0),
            "wall_dist_obs": float(wall_dist_obs),
            "v_n_obs": float(v_n_obs),
            "toward_wall": bool(toward_wall),

            # escape logging
            "escape_active": bool(escape_active),
            "escape_v_floor": float(self.escape_v_floor),
            "escape_d_gate": float(self.escape_d_gate),

            # velocity logging
            "v": float(self.v),
            "v_cmd": float(v_cmd),
            "wall_guard_active": bool(wall_guard_active),

            # shaping
            "d_coll": float(d_coll),
            "d_safe": float(d_safe),
            "obstacle_clearance": float(self.obstacle_clearance),
            "risk": float(risk),
            "close_pen": float(close_pen),
            "boost_pen": float(boost_pen),

            "stall_pen": float(stall_pen),
            "brake_pen": float(brake_pen),
            "pass_close_bonus": float(pass_close_bonus),

            "wall_safe": float(self.wall_safe),
            "wall_dist_world": float(wall_dist),
            "wall_dist_norm_world": float(wall_dist_norm),
            "wall_risk": float(wall_risk),
            "wall_pen": float(wall_pen),
            "wall_v_pen": float(wall_v_pen),
            "brake_wall_bonus": float(brake_wall_bonus),
            "wall_touch_th": float(self.wall_touch_th),
            "wall_touch_pen": float(wall_touch_pen),

            "cross_track": float(cross_track),
            "cross_track_signed": float(signed_ct),
            "ct_signed_norm_world": float(ct_signed_norm),
            "corridor_half_width": float(self.corridor_half_width),
            "randomize_corridor": bool(self.randomize_corridor),
            "alpha_detour": float(alpha_detour),

            "s_forward": float(s_now),
            "passed": bool(passed),

            "reward_profile": "round3_ttc_rearend_anti_brake_stall_passclose_escape_stop",
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self._fig is None:
            import matplotlib.pyplot as plt
            self._fig, self._ax = plt.subplots()

        self._ax.clear()

        render_world(
            self._ax,
            self.world,
            (self.x, self.y, self.theta),
            self.robot_radius,
            goal=np.array([self.gx, self.gy], dtype=np.float32),
        )

        span = self.render_half_span
        L = 3.0 * span
        tx, ty = self._corr_tx, self._corr_ty
        nx, ny = self._corr_nx, self._corr_ny
        ox, oy = self._corr_ox, self._corr_oy
        hw = self.corridor_half_width

        for s in (+1.0, -1.0):
            x1 = ox - L * tx + s * hw * nx
            y1 = oy - L * ty + s * hw * ny
            x2 = ox + L * tx + s * hw * nx
            y2 = oy + L * ty + s * hw * ny
            self._ax.plot([x1, x2], [y1, y2])

        self._ax.set_autoscale_on(False)
        self._ax.set_xlim(-span, span)
        self._ax.set_ylim(-span, span)
        self._ax.set_aspect("equal", adjustable="box")

        import matplotlib.pyplot as plt
        plt.pause(0.001)
