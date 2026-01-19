# ml/sim/runtime/env_nav_avoid.py
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..core.world import World2D
from ..core.render_utils import render_world

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


class NavAvoidRuntimeEnv(gym.Env):
    """
    Runtime env (integración):
    - Goal externo (subgoal del PRM).
    - Obstáculo dinámico externo (pose publicada) O simulado aquí (opcional).
    - Control nominal apunta al goal.
    - RL (modelo entrenado) corrige w + (NUEVO) puede frenar/boost en v con acción discreta.
    - Obs compatible con env_avoid_train:
      [dist_goal, cos(ang_goal), sin(ang_goal), d_dyn, cos(alpha), sin(alpha), v_closing]

    Selector:
    - HISTÉRESIS + COOLDOWN para evitar switching nervioso.

    Acciones (7):
      0 keep
      1 small_left
      2 small_right
      3 hard_left
      4 hard_right
      5 brake
      6 boost
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        sdf_world: str | None = None,
        model_path: str | None = None,
        device: str = "cpu",
        # robot
        dt: float = 0.05,
        v_max: float = 0.8,
        w_max: float = 5.0,                # ✅ match entrenamiento
        a_lin: float = 0.8,
        a_ang: float = 5.0,                # ✅ match entrenamiento
        robot_radius: float = 0.10,
        # nominal
        v_nom: float = 0.4,
        k_w: float = 1.8,
        # RL authority
        k_rl: float = 4.0,                 # ✅ match entrenamiento
        w_max_active: float = 5.0,         # ✅ match entrenamiento
        w_nom_limit: float | None = None,  # None => w_max
        # selector (histeresis + cooldown)
        activate_radius: float = 1.0,
        deactivate_radius: float | None = None,
        cooldown_steps: int = 10,
        # (opcional) activar por peligro (v_closing)
        use_closing_activation: bool = True,
        closing_th: float = -0.20,
        closing_dist: float = 2.0,
        # dinámico (solo dibujar / colisión)
        r_dyn: float = 0.15,
        simulate_dynamic: bool = True,
        # límites
        world_limit: float = 10.0,
        max_steps: int = 400,
        render_mode: str | None = None,
        # NUEVO: velocidad discreta en RL
        v_brake: float = 0.10,
        v_boost: float = 0.80,
    ):
        super().__init__()

        self.dt = float(dt)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.a_lin = float(a_lin)
        self.a_ang = float(a_ang)
        self.robot_radius = float(robot_radius)

        self.v_nom = float(v_nom)
        self.k_w = float(k_w)

        self.k_rl = float(k_rl)
        self.w_max_active = float(w_max_active)
        self.w_nom_limit = float(w_nom_limit) if w_nom_limit is not None else float(self.w_max)

        self.activate_radius = float(activate_radius)
        if deactivate_radius is None:
            self.deactivate_radius = float(self.activate_radius + 0.5)
        else:
            self.deactivate_radius = float(deactivate_radius)
        if self.deactivate_radius <= self.activate_radius:
            self.deactivate_radius = float(self.activate_radius + 0.2)

        self.cooldown_steps = int(max(0, cooldown_steps))

        self.use_closing_activation = bool(use_closing_activation)
        self.closing_th = float(closing_th)
        self.closing_dist = float(closing_dist)

        self.r_dyn = float(r_dyn)
        self.simulate_dynamic = bool(simulate_dynamic)

        self.world_limit = float(world_limit)
        self.max_steps = int(max_steps)

        self.v_brake = float(v_brake)
        self.v_boost = float(v_boost)

        self.render_mode = render_mode
        self._fig = None
        self._ax = None

        # mundo estático
        self.world = World2D(sdf_path=sdf_world)

        # ✅ ahora 7 acciones como training
        self.action_space = spaces.Discrete(7)

        low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -5.0], dtype=np.float32)
        high = np.array([50.0, 1.0, 1.0, 50.0, 1.0, 1.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # modelo RL opcional
        self.model = None
        if model_path is not None:
            if PPO is None:
                raise ImportError("stable-baselines3 no está instalado en este entorno.")
            self.model = PPO.load(model_path, device=device)

        self.reset()

    # ---------- setters externos ----------
    def set_goal(self, gx: float, gy: float):
        self.gx = float(gx)
        self.gy = float(gy)

    def set_robot_pose(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def set_dynamic_pose(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.vx = float(vx)
        self.vy = float(vy)

    # ---------- utils ----------
    def _wrap(self, ang: float) -> float:
        return float(np.arctan2(np.sin(ang), np.cos(ang)))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _action_to_wcorr(self, a: int) -> float:
        # 0..4 -> giro, 5/6 -> velocidad
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
        if a == 5:  # brake
            return self._clip(self.v_brake, 0.0, self.v_max)
        if a == 6:  # boost
            return self._clip(self.v_boost, 0.0, self.v_max)
        return self._clip(v_nominal, 0.0, self.v_max)

    def _nominal_controller(self) -> tuple[float, float, float]:
        dx = self.gx - self.x
        dy = self.gy - self.y
        ang_goal_world = float(np.arctan2(dy, dx))
        ang_goal = self._wrap(ang_goal_world - self.theta)

        w_nom = self._clip(self.k_w * ang_goal, -self.w_nom_limit, self.w_nom_limit)
        v_cmd = self._clip(self.v_nom, 0.0, self.v_max)
        return v_cmd, w_nom, ang_goal

    def _get_obs(self) -> np.ndarray:
        # goal
        dgx = self.gx - self.x
        dgy = self.gy - self.y
        dist_goal = float(np.hypot(dgx, dgy))
        ang_goal_world = float(np.arctan2(dgy, dgx))
        ang_goal = self._wrap(ang_goal_world - self.theta)

        # dynamic
        dx = self.cx - self.x
        dy = self.cy - self.y
        d_dyn = float(np.hypot(dx, dy))
        bearing_world = float(np.arctan2(dy, dx))
        alpha = self._wrap(bearing_world - self.theta)

        # closing speed
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

    # ---------- selector con histeresis ----------
    def _update_selector(self, d_dyn: float, v_closing: float):
        # cooldown: fuerza NOMINAL por N steps tras salir de RL
        if self._cooldown > 0:
            self._cooldown -= 1
            self._rl_active = False
            return

        danger = self.use_closing_activation and (v_closing < self.closing_th) and (d_dyn < self.closing_dist)

        # entrar
        if (not self._rl_active) and ((d_dyn < self.activate_radius) or danger):
            self._rl_active = True

        # salir
        if self._rl_active and (d_dyn > self.deactivate_radius):
            self._rl_active = False
            self._cooldown = self.cooldown_steps

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # robot default
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.v, self.w = 0.0, 0.0
        self.t = 0

        # selector state
        self._rl_active = False
        self._cooldown = 0

        # defaults si no los seteaste
        if not hasattr(self, "gx"):
            self.gx, self.gy = 2.0, 0.0
        if not hasattr(self, "cx"):
            self.cx, self.cy = 999.0, 999.0
            self.vx, self.vy = 0.0, 0.0

        self.world.dynamic_circles = [(self.cx, self.cy, self.r_dyn, self.vx, self.vy)]
        return self._get_obs(), {}

    def step(self, action: int | None = None):
        """
        Si action es None:
          - si hay modelo y selector está en RL => action = model.predict(obs)
          - si no => action=0
        Si action se pasa:
          - lo respeta.
        """
        obs = self._get_obs()

        d_dyn = float(obs[3])
        v_closing = float(obs[6])
        self._update_selector(d_dyn, v_closing)
        mode = "RL" if self._rl_active else "NOMINAL"

        # decide acción
        if action is None:
            if mode == "RL" and self.model is not None:
                a, _ = self.model.predict(obs, deterministic=True)
                action = int(a)
            else:
                action = 0
        else:
            action = int(action)

        # nominal
        v_nom_cmd, w_nom, _ = self._nominal_controller()

        # mezcla
        if mode == "RL":
            v_cmd = self._action_to_vcmd(action, v_nom_cmd)
            w_cmd = w_nom + self.k_rl * self._action_to_wcorr(action)
            w_limit = self.w_max_active
        else:
            v_cmd = v_nom_cmd
            w_cmd = w_nom
            w_limit = self.w_max

        # clamp final del giro
        w_cmd = float(np.clip(w_cmd, -w_limit, w_limit))

        # rate limiting
        self.v += np.clip(v_cmd - self.v, -self.a_lin * self.dt, self.a_lin * self.dt)
        self.w += np.clip(w_cmd - self.w, -self.a_ang * self.dt, self.a_ang * self.dt)
        self.v = float(np.clip(self.v, 0.0, self.v_max))
        self.w = float(np.clip(self.w, -w_limit, w_limit))

        # kinematics
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta = self._wrap(self.theta + self.w * self.dt)

        # safety clamp
        L = self.world_limit
        self.x = float(np.clip(self.x, -L, L))
        self.y = float(np.clip(self.y, -L, L))

        # dinámico simulado (si aplica)
        if self.simulate_dynamic:
            self.cx += self.vx * self.dt
            self.cy += self.vy * self.dt

        self.world.dynamic_circles = [(self.cx, self.cy, self.r_dyn, self.vx, self.vy)]

        # termination
        dist_goal = float(np.hypot(self.gx - self.x, self.gy - self.y))
        collision = (float(np.hypot(self.cx - self.x, self.cy - self.y)) <= (self.robot_radius + self.r_dyn))
        reached = dist_goal < 0.25

        self.t += 1
        terminated = bool(collision or reached)
        truncated = bool(self.t >= self.max_steps)

        reward = 0.0

        info = {
            "mode": mode,
            "rl_active": bool(self._rl_active),
            "cooldown": int(self._cooldown),
            "action": int(action),
            "v": float(self.v),
            "w": float(self.w),
            "v_cmd": float(v_cmd),
            "w_nom": float(w_nom),
            "w_cmd": float(w_cmd),
            "k_rl": float(self.k_rl),
            "w_limit": float(w_limit),
            "activate_radius": float(self.activate_radius),
            "deactivate_radius": float(self.deactivate_radius),
            "use_closing_activation": bool(self.use_closing_activation),
            "d_dyn": float(d_dyn),
            "v_closing": float(v_closing),
            "dist_goal": float(dist_goal),
            "collision": bool(collision),
            "reached": bool(reached),
        }

        return self._get_obs(), reward, terminated, truncated, info

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

        import matplotlib.pyplot as plt
        plt.title("NavAvoidRuntime (hysteresis + brake/boost)")
        plt.pause(0.001)
