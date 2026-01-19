# ml/train/eval_avoid_empty.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Counters:
    episodes: int = 0
    reached: int = 0
    collision: int = 0
    truncated: int = 0
    conflict: int = 0
    mirrored: int = 0

    steps_total: int = 0

    mismatch_eff_exec_steps: int = 0
    steps_with_actions: int = 0

    dclear_min_list: list = None
    cross_track_mean_list: list = None
    cross_track_max_list: list = None

    a_raw_all: np.ndarray = None
    a_eff_all: np.ndarray = None
    a_exec_all: np.ndarray = None

    a_raw_active: np.ndarray = None
    a_eff_active: np.ndarray = None
    a_exec_active: np.ndarray = None

    def __post_init__(self):
        self.dclear_min_list = []
        self.cross_track_mean_list = []
        self.cross_track_max_list = []

        self.a_raw_all = np.zeros((7,), dtype=np.int64)
        self.a_eff_all = np.zeros((7,), dtype=np.int64)
        self.a_exec_all = np.zeros((7,), dtype=np.int64)

        self.a_raw_active = np.zeros((7,), dtype=np.int64)
        self.a_eff_active = np.zeros((7,), dtype=np.int64)
        self.a_exec_active = np.zeros((7,), dtype=np.int64)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def _print_action_dist(title: str, counts: np.ndarray):
    total = int(np.sum(counts))
    if total <= 0:
        print(f"\n=== {title} ===\n(no steps)")
        return

    names = [
        "a0       keep",
        "a1 small_left",
        "a2 small_right",
        "a3  hard_left",
        "a4 hard_right",
        "a5      brake",
        "a6      boost",
    ]

    print(f"\n=== {title} ===")
    for i, name in enumerate(names):
        p = 100.0 * float(counts[i]) / float(total)
        print(f"{name}: {p:6.2f}%   (count={int(counts[i])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--p-mirror", type=float, default=0.50)
    ap.add_argument("--p-conflict", type=float, default=0.85)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)

    # si el modelo es de pasillo (10D), necesitas el half width:
    ap.add_argument("--corridor-half-width", type=float, default=0.90)
    ap.add_argument("--randomize-corridor", action="store_true")

    # fuerza CPU (recomendado)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    args = ap.parse_args()

    from stable_baselines3 import PPO

    print(f"model: {args.model}")
    print(f"p_mirror: {args.p_mirror}")
    print(f"deterministic: {bool(args.deterministic)}")
    print(f"p_conflict: {args.p_conflict}")
    print(f"max_steps: {args.max_steps}")
    print(f"device: {args.device}")

    model = PPO.load(args.model, device=args.device)

    try:
        obs_dim = int(model.observation_space.shape[0])
    except Exception:
        obs_dim = None
    print(f"model_obs_dim: {obs_dim}")

    # ---- elige env según obs_dim ----
    if obs_dim == 10:
        # Eval en pasillo (10D)
        from sim.training.env_avoid_corridor_train import AvoidCorridorTrainEnv

        env = AvoidCorridorTrainEnv(
            sdf_world="../worlds/diff_drive_empty.sdf",
            render_mode=None,
            p_mirror=float(args.p_mirror),
            p_conflict=float(args.p_conflict),
            max_steps=int(args.max_steps),
            randomize_corridor=bool(args.randomize_corridor),
            corridor_half_width=float(args.corridor_half_width),
            seed=int(args.seed),
        )
        print("[env] Using AvoidCorridorTrainEnv (10D)")
        # en el pasillo sí hay pared, pero aquí NO vamos a reportar wall metrics (simple)
    else:
        # Eval en vacío (7D)
        from sim.training.env_avoid_train import AvoidTrainEnv

        env = AvoidTrainEnv(
            sdf_world="../worlds/diff_drive_empty.sdf",
            render_mode=None,
            p_mirror=float(args.p_mirror),
            p_conflict=float(args.p_conflict),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
        )
        print("[env] Using AvoidTrainEnv (7D)")

    dt = float(getattr(env, "dt", 0.05))
    r_robot = float(getattr(env, "robot_radius", 0.10))
    r_dyn = float(getattr(env, "r_dyn", 0.15))
    sum_r = r_robot + r_dyn

    C = Counters()
    rng = np.random.default_rng(int(args.seed))

    for _ep in range(int(args.episodes)):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=ep_seed)

        C.episodes += 1
        if bool(info.get("conflict", False)):
            C.conflict += 1
        if bool(info.get("mirrored", False)):
            C.mirrored += 1

        steps = 0
        dclear_min = float("inf")
        ct_vals = []

        while True:
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            a_raw = int(action)

            obs, reward, terminated, truncated, info = env.step(a_raw)
            steps += 1

            rl_active = bool(info.get("rl_active", False))

            d_dyn = float(info.get("d_dyn", np.nan))
            if np.isfinite(d_dyn):
                dclear = float(d_dyn - sum_r)
                if dclear < dclear_min:
                    dclear_min = dclear

            ct = info.get("cross_track", None)
            if ct is not None:
                ct_vals.append(float(ct))

            a_eff = int(info.get("action_hold", a_raw))
            a_exec = int(info.get("action_exec", a_eff))

            if 0 <= a_raw <= 6:
                C.a_raw_all[a_raw] += 1
            if 0 <= a_eff <= 6:
                C.a_eff_all[a_eff] += 1
            if 0 <= a_exec <= 6:
                C.a_exec_all[a_exec] += 1

            if rl_active:
                if 0 <= a_raw <= 6:
                    C.a_raw_active[a_raw] += 1
                if 0 <= a_eff <= 6:
                    C.a_eff_active[a_eff] += 1
                if 0 <= a_exec <= 6:
                    C.a_exec_active[a_exec] += 1

            C.steps_with_actions += 1
            if a_eff != a_exec:
                C.mismatch_eff_exec_steps += 1

            if terminated or truncated:
                break

        C.steps_total += steps

        if bool(info.get("reached", False)):
            C.reached += 1
        if bool(info.get("collision", False)):
            C.collision += 1
        if bool(truncated):
            C.truncated += 1

        if dclear_min == float("inf"):
            dclear_min = np.nan
        C.dclear_min_list.append(float(dclear_min))

        if len(ct_vals) > 0:
            C.cross_track_mean_list.append(float(np.mean(ct_vals)))
            C.cross_track_max_list.append(float(np.max(ct_vals)))
        else:
            C.cross_track_mean_list.append(np.nan)
            C.cross_track_max_list.append(np.nan)

    episodes = float(C.episodes)

    reach_rate = _safe_div(C.reached, C.episodes)
    collision_rate = _safe_div(C.collision, C.episodes)
    truncated_rate = _safe_div(C.truncated, C.episodes)
    conflict_rate = _safe_div(C.conflict, C.episodes)
    mirror_rate = _safe_div(C.mirrored, C.episodes)

    avg_steps_to_end = _safe_div(C.steps_total, C.episodes)
    avg_time_to_end = avg_steps_to_end * dt

    mismatch_eff_vs_exec_rate = _safe_div(C.mismatch_eff_exec_steps, C.steps_with_actions)

    dclear_min = np.array(C.dclear_min_list, dtype=np.float64)
    ct_mean = np.array(C.cross_track_mean_list, dtype=np.float64)
    ct_max = np.array(C.cross_track_max_list, dtype=np.float64)

    print("\nepisodes:", int(C.episodes))
    print(f"mirror_rate: {mirror_rate:.3f}")
    print(f"conflict_rate: {conflict_rate:.3f}")
    print(f"reach_rate: {reach_rate:.3f}")
    print(f"collision_rate: {collision_rate:.3f}")
    print(f"truncated_rate: {truncated_rate:.3f}")
    print(f"avg_steps_to_end: {avg_steps_to_end:.2f} steps")
    print(f"avg_time_to_end:  {avg_time_to_end: .2f} s   (dt={dt})")
    print(f"mismatch_eff_vs_exec_rate: {mismatch_eff_vs_exec_rate}")

    print("\n=== cross-track (proxy de 'se abre') ===")
    print(f"avg_cross_track_mean: {float(np.nanmean(ct_mean))}")
    print(f"avg_cross_track_max: {float(np.nanmean(ct_max))}")

    print("\n=== obstacle clearance (episode-averaged) ===")
    print(f"avg_obstacle_clearance_min (m): {float(np.nanmean(dclear_min))}")

    _print_action_dist("action distribution (policy RAW / predicted, all steps)", C.a_raw_all)
    _print_action_dist("action distribution (policy EFFECTIVE, all steps)", C.a_eff_all)
    _print_action_dist("action distribution (executed, all steps)", C.a_exec_all)

    _print_action_dist("action distribution (policy RAW / predicted, rl_active steps only)", C.a_raw_active)
    _print_action_dist("action distribution (policy EFFECTIVE, rl_active steps only)", C.a_eff_active)
    _print_action_dist("action distribution (executed, rl_active steps only)", C.a_exec_active)


if __name__ == "__main__":
    main()
