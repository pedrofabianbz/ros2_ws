# ml/sim/training/env_avoid_train.py
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..core.world import World2D
from ..core.render_utils import render_world


class AvoidTrainEnv(gym.Env):
    """
    AvoidDynEnv (env_avoid_train.py / AvoidDynEnv)
    =============================================

    Qué hace
    --------
    Entorno Gymnasium para entrenar evitación de colisión con **un (1) obstáculo dinámico**.
    Está pensado como un “sandbox” simple de evasión:

    - Robot diferencial (modelo unicycle) con límites de velocidad y *rate limiting* (aceleraciones máximas).
    - Obstáculo dinámico circular con velocidad constante y rebote dentro de un cuadrado (`world_limit`).
    - NO usa LiDAR: la observación asume que conoces (o estimas) el estado relativo del obstáculo.
    - Mundo estático opcional cargado desde un mapa SDF mediante `World2D` (principalmente para render).

    Cómo se corre / cómo se usa
    ---------------------------
    Este archivo normalmente NO se ejecuta solo: se instancia desde un script de entrenamiento (PPO)
    o desde un script de “play/debug”.

    Ejemplo mínimo en Python (pseudo-uso):

        env = AvoidDynEnv(sdf_world="worlds/diff_drive_empty.sdf", render_mode="human", seed=0)
        obs, _ = env.reset()
        for _ in range(200):
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()

    Inputs principales (constructor)
    --------------------------------
    - sdf_world: ruta al mapa `.sdf` (puede ser None). Se carga con `World2D`.
    - risk_radius: radio de “zona de riesgo” (m). Dentro de este radio se penaliza cercanía.
    - world_limit: límites del cuadrado [-L, L] para rebote del dinámico (solo “empty”).
    - max_steps: límite de pasos por episodio (truncamiento).
    - render_mode: "human" para dibujar con Matplotlib, o None para entrenamiento rápido.
    - seed: semilla para reproducibilidad.

    Acciones (Discrete(4))
    ----------------------
    Acción discreta `a ∈ {0,1,2,3}` que se traduce a comandos (v_cmd, w_cmd):

    - 0: avanzar lento        -> (0.2,  0.0)
    - 1: esquivar izquierda    -> (0.2, +1.0)
    - 2: esquivar derecha      -> (0.2, -1.0)
    - 3: frenar / detener      -> (0.0,  0.0)

    Observación (Box, 6D)
    ---------------------
    Estado observado: `[d, cos(alpha), sin(alpha), v, w, v_closing]`

    - d: distancia robot–dinámico (centro-a-centro).
    - alpha: ángulo del dinámico en el marco del robot.
    - v, w: velocidades actuales del robot.
    - v_closing: velocidad de cierre aproximada (proyección de velocidad relativa sobre la línea de visión).
    Convención en este env: v_closing < 0 significa que se acerca, > 0 que se aleja.

    Rewards y terminales
    --------------------
    - Reward base por “seguir vivo”: +1.0 por paso.
    - Penalización por cercanía dentro de `risk_radius`:
        reward -= (risk_radius - d) * 4.0
    - Penalización por frenar (acción 3): -0.05
    - Colisión (terminal): reward = -100.0, terminated=True

    Condición de colisión:
    - collision = (distancia centro-a-centro <= robot_radius + r_dyn)

    Outputs de step()
    -----------------
    Devuelve tupla estándar Gymnasium:

        obs, reward, terminated, truncated, info

    - terminated: True si hubo colisión.
    - truncated: True si se alcanzó `max_steps`.
    - info: diccionario con:
        - "d_dyn": distancia actual al dinámico
        - "collision": bool

    Render
    ------
    Si `render_mode="human"`, dibuja el mundo/robot/dinámico con `render_world(...)` usando Matplotlib.
    Para entrenamiento masivo, deja `render_mode=None` (acelera bastante).

    Notas rápidas
    -------------
    - El mundo estático cargado por SDF se usa sobre todo para visualización; la dinámica del obstáculo
    se maneja internamente en el env.
    - Si más adelante quieres hacerlo más “realista”, típicamente se randomiza `v_dyn`, se agregan
    varios dinámicos o se cambia la observación para no “regalar” el estado exacto.
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
        # activar por peligro (v_closing) sin subir radio
        use_closing_activation: bool = True,
        closing_th: float = -0.20,
        closing_dist: float = 2.0,
        # reward weights base
        k_progress: float = 1.0,
        k_risk: float = 2.5,
        k_w_far: float = 0.03,
        k_goal_far: float = 0.20,
        alive_bonus: float = 0.0,
        goal_bonus: float = 20.0,
        # anti-orbiting
        stuck_steps: int = 25,
        stuck_eps: float = 0.002,
        k_stuck: float = 0.5,
        # prisa
        time_cost: float = 0.02,
        k_path: float = 0.10,
        finish_bonus_fast: float = 5.0,
        # RL decide cada N steps
        rl_decimation: int = 1,
        # render zoom
        render_half_span: float = 2.5,
        # ---- control discreto de velocidad ----
        v_brake: float = 0.05,
        v_boost: float = 0.80,
        k_speed_cmd: float = 0.00,
        k_brake_use: float = 0.00,
        # ---- spawn rejection ----
        min_spawn_sep: float = 0.35,
        min_goal_sep: float = 0.35,
        spawn_tries: int = 50,
        # ---- evasión óptima (anti-detour + giro suave cerca) ----
        k_detour: float = 0.10,
        k_w_near: float = 0.005,
        # ---- mirror augmentation ----
        p_mirror: float = 0.35,
        # ---- penalidad pacing ----
        k_pace: float = 0.9,
        pace_dist: float = 1.25,
        pace_v_th: float = 0.15,
        pace_prog_th: float = 0.01,
        # ---- shaping "no abrirse" ----
        k_act: float = 0.12,
        k_push: float = 1.0,
        k_close: float = 0.9,
        prog_clip_lo: float = -0.05,
        prog_clip_hi: float = 0.25,
        vclose_clip: float = 1.5,
        k_spin: float = 0.03,
        # ---- boost penalty en riesgo ----
        k_boost_near: float = 0.35,
        # ---- curriculum "parallel-ahead" ----
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
        parallel_extra_margin: float = 0.35,  # activa un poco antes

        # ---- NUEVO: pass shaping (rebase) ----
        k_pass: float = 2.0,
        pass_margin: float = 0.12,
        pass_gate_dist: float = 1.8,

        # ---- NUEVO: bloqueo (no logra rebasar) ----
        blocked_steps: int = 12,
        k_blocked: float = 0.20,
        k_boost_blocked: float = 0.25,
    ):
        super().__init__()
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

        # knobs
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

        # reward weights base
        self.k_progress = float(k_progress)
        self.k_risk = float(k_risk)
        self.k_w_far = float(k_w_far)
        self.k_goal_far = float(k_goal_far)
        self.alive_bonus = float(alive_bonus)
        self.goal_bonus = float(goal_bonus)

        self.stuck_steps = int(stuck_steps)
        self.stuck_eps = float(stuck_eps)
        self.k_stuck = float(k_stuck)

        self.time_cost = float(time_cost)
        self.k_path = float(k_path)
        self.finish_bonus_fast = float(finish_bonus_fast)

        self.rl_decimation = int(max(1, rl_decimation))

        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        self.render_half_span = float(render_half_span)

        # velocidad discreta (clamps)
        self.v_brake = float(np.clip(v_brake, 0.0, self.v_max))
        self.v_boost = float(np.clip(v_boost, 0.0, self.v_max))
        if self.v_boost < self.v_brake:
            self.v_boost, self.v_brake = self.v_brake, self.v_boost

        self.k_speed_cmd = float(k_speed_cmd)
        self.k_brake_use = float(k_brake_use)

        # spawn rejection
        self.min_spawn_sep = float(min_spawn_sep)
        self.min_goal_sep = float(min_goal_sep)
        self.spawn_tries = int(max(1, spawn_tries))

        # óptimo
        self.k_detour = float(k_detour)
        self.k_w_near = float(k_w_near)

        # mirror
        self.p_mirror = float(np.clip(p_mirror, 0.0, 1.0))
        self._mirrored = False

        # pacing
        self.k_pace = float(k_pace)
        self.pace_dist = float(pace_dist) if pace_dist is not None else 0.85 * self.risk_radius
        self.pace_v_th = float(max(0.0, pace_v_th))
        self.pace_prog_th = float(max(0.0, pace_prog_th))

        # shaping
        self.k_act = float(k_act)
        self.k_push = float(k_push)
        self.k_close = float(k_close)
        self.prog_clip_lo = float(prog_clip_lo)
        self.prog_clip_hi = float(prog_clip_hi)
        self.vclose_clip = float(max(1e-6, vclose_clip))
        self.k_spin = float(k_spin)

        # boost penalty
        self.k_boost_near = float(max(0.0, k_boost_near))

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

        # ---- NUEVO: pass + bloqueo ----
        self.k_pass = float(max(0.0, k_pass))
        self.pass_margin = float(max(0.0, pass_margin))
        self.pass_gate_dist = float(max(0.1, pass_gate_dist))

        self.blocked_steps = int(max(1, blocked_steps))
        self.k_blocked = float(max(0.0, k_blocked))
        self.k_boost_blocked = float(max(0.0, k_boost_blocked))

        self._s_forward_prev = 0.0
        self._blocked_ctr = 0

        # Mundo estático desde SDF
        self.world = World2D(sdf_path=sdf_world)

        # 7 acciones
        self.action_space = spaces.Discrete(7)

        # obs: [dist_goal, cos(ang_goal), sin(ang_goal), d_dyn, cos(alpha), sin(alpha), v_closing]
        low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -5.0], dtype=np.float32)
        high = np.array([50.0, 1.0, 1.0, 50.0, 1.0, 1.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    # ---------------- utils ----------------
    def _wrap(self, ang: float) -> float:
        return float(np.arctan2(np.sin(ang), np.cos(ang)))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _mirror_action(self, a: int) -> int:
        """Mirror correcto: izquierda<->derecha."""
        a = int(a)
        if a == 1:
            return 2
        if a == 2:
            return 1
        if a == 3:
            return 4
        if a == 4:
            return 3
        return a  # keep/brake/boost iguales

    def _canonicalize_obs(self, obs_world: np.ndarray) -> np.ndarray:
        """
        Convierte observación del mundo (que puede estar mirrored) a observación CANÓNICA.

        Mirror respecto a eje X: y->-y, theta->-theta.
        En esta representación:
        - sin(ang_goal) cambia signo
        - sin(alpha) cambia signo
        El resto queda igual.
        """
        if not self._mirrored:
            return obs_world
        obs = obs_world.copy()
        obs[2] = -obs[2]  # sin(ang_goal)
        obs[5] = -obs[5]  # sin(alpha)
        return obs

    def _s_forward(self) -> float:
        """
        Proyección del obstáculo en el eje forward del robot.
        > 0 => obstáculo adelante
        < 0 => obstáculo atrás (lo rebasaste)
        """
        fx = float(np.cos(self.theta))
        fy = float(np.sin(self.theta))
        dx = float(self.cx - self.x)
        dy = float(self.cy - self.y)
        return dx * fx + dy * fy

    # ---- acción -> corrección de giro (solo 0..4) ----
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

    # ---- acción -> setpoint de velocidad (solo relevante en RL activo) ----
    def _action_to_vcmd(self, a: int, v_nominal: float) -> float:
        if a == 5:  # brake
            return float(self.v_brake)
        if a == 6:  # boost
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
        """
        Dinámico delante del robot y moviéndose casi igual.
        Fuerza el caso "paralelo" donde frenar o cambiar de lado es necesario.
        """
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
        """Mirror augmentation: refleja respecto al eje X (y -> -y)."""
        self._mirrored = (float(self.rng.uniform(0.0, 1.0)) < self.p_mirror)
        if not self._mirrored:
            return

        # posiciones
        self.y = -self.y
        self.gy = -self.gy
        self.cy = -self.cy

        # dinámico: vy cambia
        self.vy = -self.vy

        # orientación robot
        self.theta = self._wrap(-self.theta)

        # velocidad angular cambia signo
        self.w = -self.w

    def _update_dynamic(self):
        self.cx += self.vx * self.dt
        self.cy += self.vy * self.dt

        L = self.world_limit
        if self.cx < -L or self.cx > L:
            self.vx *= -1.0
            self.cx = float(np.clip(self.cx, -L, L))
        if self.cy < -L or self.cy > L:
            self.vy *= -1.0
            self.cy = float(np.clip(self.cy, -L, L))

    # ---------------- obs ----------------
    def _get_obs_world(self) -> np.ndarray:
        """Observación del mundo simulado (puede estar mirrored)."""
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

        if d_dyn < 1e-6:
            v_closing = 0.0
        else:
            v_rx = self.v * np.cos(self.theta)
            v_ry = self.v * np.sin(self.theta)
            rvx = self.vx - v_rx
            rvy = self.vy - v_ry
            v_closing = float((rvx * dx + rvy * dy) / d_dyn)

        return np.array(
            [dist_goal, np.cos(ang_goal), np.sin(ang_goal), d_dyn, np.cos(alpha), np.sin(alpha), v_closing],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        """Observación que ve el agente (siempre CANÓNICA)."""
        return self._canonicalize_obs(self._get_obs_world())

    # ---------------- selector (histeresis) ----------------
    def _update_rl_active(self, d_dyn: float, v_closing: float, alpha: float):
        danger_closing = self.use_closing_activation and (v_closing < self.closing_th) and (d_dyn < self.closing_dist)

        # Paralelo: delante + v_closing pequeño
        front = (abs(alpha) < self.parallel_front_th)
        parallelish = (abs(v_closing) < self.parallel_vclose_th)

        # Acotar para que NO active lejos
        d_parallel_gate = min(self.risk_radius + 0.25, self.activate_radius + self.parallel_extra_margin)
        danger_parallel = (d_dyn < d_parallel_gate) and front and parallelish

        danger = bool(danger_closing or danger_parallel)

        if self.rl_active:
            if d_dyn > self.deactivate_radius:
                self.rl_active = False
        else:
            if (d_dyn < self.activate_radius) or danger:
                self.rl_active = True

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

        self.prev_dist_goal = float(np.hypot(self.gx - self.x, self.gy - self.y))

        self.rl_active = False
        self._hold_action = 0
        self._no_progress = 0

        self.v_set = float(np.clip(self.v_nom, 0.0, self.v_max))
        self.v_set_prev = self.v_set

        self.world.dynamic_circles = [(self.cx, self.cy, self.r_dyn, self.vx, self.vy)]

        # NUEVO: pass/bloqueo state
        self._s_forward_prev = float(self._s_forward())
        self._blocked_ctr = 0

        return self._get_obs(), {
            "conflict": bool(getattr(self, "_conflict", False)),
            "mirrored": bool(self._mirrored),
            "scenario": str(getattr(self, "_scenario", "unknown")),
        }

    def step(self, action: int):
        # acción que eligió el agente (CANÓNICA)
        action_raw = int(action)

        # para pass detection
        s_prev = float(self._s_forward_prev)

        # obs CANÓNICA al inicio del step (selector + shaping)
        obs0 = self._get_obs()  # canónica
        d_dyn0 = float(obs0[3])
        v_closing0 = float(obs0[6])
        alpha0 = float(np.arctan2(obs0[5], obs0[4]))  # marco canónico

        self._update_rl_active(d_dyn0, v_closing0, alpha0)

        # decimation sobre acción CANÓNICA
        if self.rl_decimation > 1:
            if (self.t % self.rl_decimation) == 0:
                self._hold_action = action_raw
            action_hold = int(self._hold_action)
        else:
            action_hold = action_raw

        # acción ejecutada en el mundo (si está mirrored: swap L<->R)
        action_exec = self._mirror_action(action_hold) if self._mirrored else action_hold

        self.t += 1

        v_nom_cmd, w_nom, ang_goal = self._nominal_controller()

        if self.rl_active:
            # dinámica del mundo usa ACTION EXEC
            w_cmd = w_nom + self.k_rl * self._action_to_wcorr(action_exec)
            v_cmd = self._action_to_vcmd(action_exec, v_nom_cmd)
        else:
            w_cmd = w_nom
            v_cmd = v_nom_cmd

        self.v_set_prev = self.v_set
        self.v_set = float(v_cmd)

        # rate limiting
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

        d_dyn = float(np.hypot(self.cx - self.x, self.cy - self.y))
        collision = d_dyn <= (self.robot_radius + self.r_dyn)

        dist_goal = float(np.hypot(self.gx - self.x, self.gy - self.y))
        reached = dist_goal < 0.25

        # ---- NUEVO: pass detection (rebase) ----
        s_now = float(self._s_forward())
        self._s_forward_prev = s_now

        passed = False
        if self.k_pass > 0.0:
            # gate robusto: si estuvo cerca en algún punto del step
            d_gate = float(min(d_dyn0, d_dyn))
            if d_gate < self.pass_gate_dist:
                m = self.pass_margin
                if (s_prev > +m) and (s_now < -m):
                    passed = True

        # -------- reward --------
        reward = 0.0

        # 0) prisa
        reward -= self.time_cost
        step_dist = abs(self.v) * self.dt
        reward -= self.k_path * step_dist

        if self.k_speed_cmd > 0.0:
            reward -= self.k_speed_cmd * abs(self.v_set - self.v_set_prev)

        # penaliza abuso de freno (en espacio CANÓNICO)
        if self.k_brake_use > 0.0 and self.rl_active and action_hold == 5:
            reward -= self.k_brake_use

        # 1) progreso (clip)
        prog = self.prev_dist_goal - dist_goal
        prog_c = float(np.clip(prog, self.prog_clip_lo, self.prog_clip_hi))
        reward += self.k_progress * prog_c
        self.prev_dist_goal = dist_goal

        # 2) vivir
        reward += self.alive_bonus

        # 3) costo de acción agresiva (CANÓNICO)
        act_cost = {0: 0.0, 1: 1.0, 2: 1.0, 3: 2.2, 4: 2.2, 5: 2.0, 6: 1.8}[int(action_hold)]
        reward -= self.k_act * act_cost

        reward -= self.k_spin * (abs(self.w) / self.w_max) ** 2

        # 3b) PASS BONUS
        if passed:
            reward += self.k_pass

        # 4) riesgo + push-away + closing + boost penalty
        risk = 0.0
        push = 0.0
        close_pen = 0.0
        boost_pen = 0.0

        if d_dyn < self.risk_radius:
            risk = float((self.risk_radius - d_dyn) / max(self.risk_radius, 1e-6))  # 0..1

            reward -= self.k_risk * risk

            dd = float(np.clip(d_dyn - d_dyn0, -0.2, 0.2))
            push = dd
            reward += self.k_push * dd

            close = float(np.clip(-v_closing0, 0.0, self.vclose_clip) / self.vclose_clip)
            close_pen = close * risk
            reward -= self.k_close * close_pen

            # boost penalty (CANÓNICO)
            if self.rl_active and int(action_hold) == 6 and self.k_boost_near > 0.0:
                boost_pen = self.k_boost_near * risk
                reward -= boost_pen

            reward -= self.k_w_near * (abs(self.w) / self.w_max) ** 2

        # 4b) pacing (se mantiene)
        pace_pen = 0.0
        in_parallel = bool((abs(v_closing0) < self.pace_v_th) and (abs(alpha0) < self.parallel_front_th))
        if (d_dyn < self.pace_dist) and (abs(v_closing0) < self.pace_v_th) and (prog < self.pace_prog_th):
            d_scale = float(np.clip((self.pace_dist - d_dyn) / max(self.pace_dist, 1e-6), 0.0, 1.0))
            v_scale = float(np.clip(1.0 - (abs(v_closing0) / max(self.pace_v_th, 1e-6)), 0.0, 1.0))
            pace_pen = self.k_pace * d_scale * v_scale
            reward -= pace_pen

        # ---- NUEVO: bloqueo (no logra rebasar) ----
        ahead = (s_now > 0.0)  # obstáculo sigue adelante
        parallelish = bool((abs(v_closing0) < self.pace_v_th) and (abs(alpha0) < self.parallel_front_th))
        close_enough = (d_dyn < self.pace_dist)
        blocked_now = bool(ahead and close_enough and parallelish and (prog < self.pace_prog_th))

        if blocked_now:
            self._blocked_ctr += 1
        else:
            self._blocked_ctr = 0

        blocked_active = bool(self._blocked_ctr >= self.blocked_steps)

        if blocked_active and self.k_blocked > 0.0:
            reward -= self.k_blocked

        if blocked_active and self.k_boost_blocked > 0.0 and int(action_hold) == 6:
            reward -= self.k_boost_blocked

        # 5) anti-detour
        vx_line = float(self.gx - self.x0)
        vy_line = float(self.gy - self.y0)
        den = float(np.hypot(vx_line, vy_line)) + 1e-9
        cross_track = abs(vy_line * (self.x - self.x0) - vx_line * (self.y - self.y0)) / den

        den_r = max(self.deactivate_radius - self.activate_radius, 1e-6)
        alpha_detour = float(np.clip((d_dyn - self.activate_radius) / den_r, 0.0, 1.0))
        reward -= self.k_detour * alpha_detour * cross_track

        # 6) lejos
        if d_dyn >= self.risk_radius:
            reward -= self.k_w_far * abs(self.w)
            reward -= self.k_goal_far * abs(ang_goal)

        # 7) anti-orbiting
        if prog < self.stuck_eps:
            self._no_progress += 1
        else:
            self._no_progress = 0
        if self._no_progress >= self.stuck_steps:
            reward -= self.k_stuck

        # 8) llegar + rapidez
        if reached:
            fast_bonus = self.finish_bonus_fast * (1.0 - (self.t / max(1, self.max_steps)))
            reward += self.goal_bonus + fast_bonus

        # 9) choque
        if collision:
            reward = -100.0

        terminated = bool(collision or reached)
        truncated = bool(self.t >= self.max_steps)

        info = {
            "scenario": str(getattr(self, "_scenario", "unknown")),
            "d_dyn": float(d_dyn),
            "dist_goal": float(dist_goal),
            "prog": float(prog),
            "prog_c": float(prog_c),
            "collision": bool(collision),
            "reached": bool(reached),
            "conflict": bool(getattr(self, "_conflict", False)),
            "mirrored": bool(self._mirrored),

            "rl_active": bool(self.rl_active),
            "activate_radius": float(self.activate_radius),
            "deactivate_radius": float(self.deactivate_radius),
            "use_closing_activation": bool(self.use_closing_activation),
            "v_closing0": float(v_closing0),
            "alpha0": float(alpha0),

            "no_progress": int(self._no_progress),
            "rl_decimation": int(self.rl_decimation),
            "time_cost": float(self.time_cost),
            "step_dist": float(step_dist),

            # acciones: canónica vs ejecutada
            "action_raw": int(action_raw),
            "action_hold": int(action_hold),   # canónica aplicada (post-decimation)
            "action_exec": int(action_exec),   # ejecutada en el mundo
            "action": int(action_exec),        # compatibilidad con scripts viejos (executed)

            "act_cost": float(act_cost),
            "v_cmd": float(v_cmd),
            "v_set": float(self.v_set),
            "v": float(self.v),
            "w": float(self.w),

            "risk": float(risk),
            "push": float(push),
            "close_pen": float(close_pen),
            "boost_pen": float(boost_pen),

            "in_parallel": bool(in_parallel),
            "pace_pen": float(pace_pen),

            "cross_track": float(cross_track),
            "alpha_detour": float(alpha_detour),

            # NUEVO: pass/bloqueo debug
            "s_forward": float(s_now),
            "passed": bool(passed),
            "blocked_ctr": int(self._blocked_ctr),
            "blocked_active": bool(blocked_active),
            "blocked_now": bool(blocked_now),
        }

        # devolver obs CANÓNICA
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
        self._ax.set_autoscale_on(False)
        self._ax.set_xlim(-span, span)
        self._ax.set_ylim(-span, span)
        self._ax.set_aspect("equal", adjustable="box")

        import matplotlib.pyplot as plt
        plt.pause(0.001)
