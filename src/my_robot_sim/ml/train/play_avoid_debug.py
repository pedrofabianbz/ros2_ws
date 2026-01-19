# ml/train/play_avoid_debug.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from stable_baselines3 import PPO

from sim.training.env_avoid_corridor_train import AvoidCorridorTrainEnv


ACTIONS = {
    0: "keep",
    1: "small_left",
    2: "small_right",
    3: "hard_left",
    4: "hard_right",
    5: "brake",
    6: "boost",
}


def _safe(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _steer_dir(a: int) -> int:
    """
    -1 => gira derecha
    +1 => gira izquierda
     0 => no gira (keep/brake/boost)
    """
    if a in (1, 3):
        return +1
    if a in (2, 4):
        return -1
    return 0


@dataclass
class StepRow:
    t: int
    d_dyn: float
    alpha: float
    v_close: float
    dist_goal: float
    ct: float
    wall: float
    rl_active: int
    a_raw: int
    a_exec: int
    r: float


def make_env(seed: int, render: bool, p_mirror: float, use_corridor_obs: bool) -> AvoidCorridorTrainEnv:
    return AvoidCorridorTrainEnv(
        sdf_world="../worlds/diff_drive_empty.sdf",
        render_mode="human" if render else None,
        seed=seed,
        p_mirror=p_mirror,
        use_corridor_obs=use_corridor_obs,
        # (si quieres, aquí puedes forzar corridor_half_width=0.9)
    )


def run_one_episode(
    model: PPO,
    seed: int,
    p_mirror: float,
    deterministic: bool,
    render: bool,
    last_k: int,
    verbose: bool,
) -> Dict[str, Any]:
    # autodetect obs dim
    model_obs_dim = int(model.observation_space.shape[0])
    use_corridor_obs = (model_obs_dim == 9)

    env = make_env(seed=seed, render=render, p_mirror=p_mirror, use_corridor_obs=use_corridor_obs)
    obs, info0 = env.reset(seed=seed)

    mirrored = bool(info0.get("mirrored", False))
    scenario = str(info0.get("scenario", "unknown"))

    hist: List[StepRow] = []
    t = 0

    wrong_side_ctr = 0
    checked_ctr = 0

    collision_wall = 0
    collision_dyn = 0

    done = False
    trunc = False
    last_info: Dict[str, Any] = {}

    while not (done or trunc):
        a_raw, _ = model.predict(obs, deterministic=deterministic)
        a_raw = int(a_raw)

        obs2, r, done, trunc, info = env.step(a_raw)
        a_exec = int(info.get("action", a_raw))

        # decode obs (CANÓNICA)
        dist_goal = float(obs2[0])
        d_dyn = float(obs2[3])
        alpha = float(np.arctan2(obs2[5], obs2[4]))
        v_close = float(obs2[6])

        # corridor obs extra si existe
        ct = float(obs2[7]) if obs2.shape[0] >= 8 else float("nan")      # ct_signed_norm (canónico)
        wall = float(obs2[8]) if obs2.shape[0] >= 9 else float("nan")    # wall_dist_norm (canónico)

        rl_active = int(bool(info.get("rl_active", False)))

        # “lado equivocado”: si obstáculo está a la izquierda (alpha>0),
        # una acción de giro izquierda suele ser mala para rebasar; y viceversa.
        # (solo evaluamos cuando está adelante-ish y cerca-ish)
        frontish = abs(alpha) < 1.2
        closeish = d_dyn < 1.6
        if frontish and closeish:
            checked_ctr += 1
            steer = _steer_dir(a_raw)  # ojo: raw es canónico (es lo que quieres diagnosticar)
            if alpha > 0.15 and steer > 0:   # obstáculo a izq, giras izq
                wrong_side_ctr += 1
            if alpha < -0.15 and steer < 0:  # obstáculo a der, giras der
                wrong_side_ctr += 1

        hist.append(
            StepRow(
                t=t,
                d_dyn=d_dyn,
                alpha=alpha,
                v_close=v_close,
                dist_goal=dist_goal,
                ct=ct,
                wall=wall,
                rl_active=rl_active,
                a_raw=a_raw,
                a_exec=a_exec,
                r=float(r),
            )
        )
        if len(hist) > last_k:
            hist.pop(0)

        if render:
            env.render()

        # collision breakdown (si tu env lo reporta)
        if bool(info.get("collision_wall", False)):
            collision_wall = 1
        if bool(info.get("collision_dyn", False)):
            collision_dyn = 1

        obs = obs2
        last_info = info
        t += 1

        if verbose and (t % 10 == 0):
            print(
                f"t={t:03d} d_dyn={d_dyn:.2f} alpha={alpha:+.2f} vclose={v_close:+.2f} "
                f"ct={ct:+.2f} wall={wall:.2f} a={ACTIONS.get(a_raw,a_raw)} rl={rl_active} r={float(r):+.3f}"
            )

    reached = bool(last_info.get("reached", False))
    collision = bool(last_info.get("collision", False))
    reason = "reached" if reached else ("collision" if collision else ("truncated" if trunc else "not_reached"))

    # print table if collision
    if collision:
        print("\n--- LAST STEPS (collision) ---")
        print(" t | d_dyn | alpha | vcls |  ct  | wall | rl | a_raw      | a_exec     |   r")
        for row in hist:
            print(
                f"{row.t:3d} | {row.d_dyn:5.2f} | {row.alpha:+5.2f} | {row.v_close:+5.2f} | "
                f"{row.ct:+5.2f} | {row.wall:5.2f} | {row.rl_active:2d} | "
                f"{ACTIONS.get(row.a_raw,row.a_raw):>10s} | {ACTIONS.get(row.a_exec,row.a_exec):>10s} | {row.r:+6.3f}"
            )
        print("--- end ---\n")

    env.close()

    return {
        "steps": t,
        "mirrored": mirrored,
        "scenario": scenario,
        "reached": reached,
        "collision": collision,
        "truncated": bool(trunc),
        "reason": reason,
        "wrong_side_rate": (wrong_side_ctr / checked_ctr) if checked_ctr > 0 else float("nan"),
        "checked": checked_ctr,
        "collision_wall": collision_wall,
        "collision_dyn": collision_dyn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=123)
    ap.add_argument("--p-mirror", type=float, default=0.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--last-k", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    model_obs_dim = int(model.observation_space.shape[0])
    print("model:", args.model)
    print("model_obs_dim:", model_obs_dim)
    print("p_mirror:", float(args.p_mirror))
    deterministic = not bool(args.stochastic)
    print("deterministic:", bool(deterministic))
    print("render:", bool(args.render))

    reached = 0
    collided = 0
    trunc = 0
    wall = 0
    dyn = 0
    wrong_rates: List[float] = []

    for ep in range(int(args.episodes)):
        seed = int(args.seed_base + ep)
        out = run_one_episode(
            model=model,
            seed=seed,
            p_mirror=float(args.p_mirror),
            deterministic=deterministic,
            render=bool(args.render),
            last_k=int(args.last_k),
            verbose=bool(args.verbose),
        )
        reached += int(out["reached"])
        collided += int(out["collision"])
        trunc += int(out["truncated"])
        wall += int(out["collision_wall"])
        dyn += int(out["collision_dyn"])
        if not np.isnan(out["wrong_side_rate"]):
            wrong_rates.append(float(out["wrong_side_rate"]))

        print(
            f"ep={ep:03d} seed={seed} "
            f"reason={out['reason']:<10s} steps={out['steps']:3d} "
            f"wall={out['collision_wall']} dyn={out['collision_dyn']} "
            f"wrong_side_rate={out['wrong_side_rate']:.3f} (n={out['checked']})"
        )

    n = int(args.episodes)
    print("\n=== SUMMARY ===")
    print("episodes:", n)
    print("reach_rate:", reached / n)
    print("collision_rate:", collided / n)
    print("truncated_rate:", trunc / n)
    if collided > 0:
        print("collision_wall_rate (over eps):", wall / n)
        print("collision_dyn_rate  (over eps):", dyn / n)
    if len(wrong_rates) > 0:
        print("avg_wrong_side_rate:", float(np.mean(wrong_rates)))


if __name__ == "__main__":
    main()
