# ml/train/eval_avoid.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import numpy as np
from stable_baselines3 import PPO

from sim.training.env_avoid_train import AvoidTrainEnv
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


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _is_num(x: Any) -> bool:
    return isinstance(x, (float, int, np.floating, np.integer)) and np.isfinite(float(x))


def _is_bool(x: Any) -> bool:
    return isinstance(x, (bool, np.bool_))


def mirror_action(a: int) -> int:
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


# =========================
# NEW: --env-kw passthrough
# =========================
def _parse_env_kw_list(items: Optional[List[str]]) -> Dict[str, Any]:
    """
    Parse lista tipo ["k=v", "flag=true", "n=10"] a dict con tipos.
    Soporta bool/int/float/str.
    También soporta "--env-kw flag" => True.
    """
    if not items:
        return {}

    def _coerce(v: str) -> Any:
        s = v.strip()
        lo = s.lower()
        if lo in ("true", "1", "yes", "y", "on"):
            return True
        if lo in ("false", "0", "no", "n", "off"):
            return False

        # int?
        try:
            # evita que "08" se lea raro -> lo dejamos como str
            if s.startswith("0") and len(s) > 1 and s[1].isdigit():
                raise ValueError
            return int(s)
        except Exception:
            pass

        # float?
        try:
            return float(s)
        except Exception:
            return s

    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            out[it.strip()] = True
            continue
        k, v = it.split("=", 1)
        out[k.strip()] = _coerce(v)
    return out
def make_env(
    seed: int,
    render: bool,
    p_mirror: float,
    obs_dim: int,
    corridor_half_width: float = 0.9,
    randomize_corridor: bool = False,
    corridor_hw_min: float = 0.55,
    corridor_hw_max: float = 1.20,
    # NEW
    env_kwargs_extra: Optional[Dict[str, Any]] = None,
):
    """
    Selecciona env según obs_dim del modelo.

    - 7  -> AvoidTrainEnv (vacío)
    - 9  -> AvoidCorridorTrainEnv (pasillo, 9D histórico)
    - 10 -> AvoidCorridorTrainEnv (pasillo, 10D con v_lat_rel)
    """
    common = dict(
        sdf_world="../worlds/diff_drive_empty.sdf",
        render_mode="human" if render else None,
        p_conflict=0.8,
        v_dyn_min=0.15,
        v_dyn_max=0.8,
        v_nom=0.4,
        risk_radius=1.0,
        max_steps=250,
        seed=seed,
        p_mirror=p_mirror,
    )

    # merge extra kwargs (CLI)
    if env_kwargs_extra:
        common.update(env_kwargs_extra)

    if obs_dim == 7:
        return AvoidTrainEnv(**common)

    if obs_dim in (9, 10):
        env_kwargs = dict(
            **common,
            corridor_half_width=float(corridor_half_width),
            use_corridor_obs=True,
            randomize_corridor=bool(randomize_corridor),
            corridor_hw_min=float(corridor_hw_min),
            corridor_hw_max=float(corridor_hw_max),
        )

        # Compat opcional: algunos envs soportan apagar/prender v_lat_rel explícitamente.
        # Si no existe, lo ignoramos (y evitamos warning).
        try:
            env_kwargs["use_v_lat_rel_obs"] = bool(obs_dim == 10)
            return AvoidCorridorTrainEnv(**env_kwargs)
        except TypeError:
            env_kwargs.pop("use_v_lat_rel_obs", None)
            return AvoidCorridorTrainEnv(**env_kwargs)

    raise ValueError(f"Unsupported model obs_dim={obs_dim}. Esperaba 7, 9 o 10.")


@dataclass
class EpisodeStats:
    steps: int
    steps_rl: int
    dt: float
    mirrored: bool
    scenario: str
    conflict: bool
    reached: bool
    collision: bool
    collision_wall: bool
    collision_dyn: bool
    truncated: bool
    reason: str

    act_policy_raw: np.ndarray
    act_policy_effective: np.ndarray
    act_exec: np.ndarray

    act_policy_raw_rl: np.ndarray
    act_policy_effective_rl: np.ndarray
    act_exec_rl: np.ndarray

    mismatch_effective_vs_exec: int

    cross_track_mean: float
    cross_track_max: float

    # wall/clearance debug
    wall_seen: bool
    wall_dist_min: float
    wall_dist_norm_min: float
    wall_pen_sum: float
    wall_risk_mean: float
    wall_v_pen_sum: float
    override_brake_steps: int
    override_wall_steps: int
    override_cpa_steps: int
    brake_wall_bonus_sum: float

    d_coll_min: float
    d_safe_min: float
    obstacle_clearance_min: float
    wall_safe: float
    d_coll: float
    d_safe: float


def _pick_runover_flag(info: Dict[str, Any]) -> bool:
    v = info.get("anti_runover_active", None)
    if _is_bool(v):
        return bool(v)
    v = info.get("runover_guard", None)
    if _is_bool(v):
        return bool(v)
    return False


def _as_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _get_alpha0(info: Dict[str, Any], obs: Any) -> float:
    a = info.get("alpha0", None)
    if _is_num(a):
        return float(a)

    ca = info.get("cos_alpha", None)
    sa = info.get("sin_alpha", None)
    if _is_num(ca) and _is_num(sa):
        return float(np.arctan2(float(sa), float(ca)))

    try:
        if obs is not None and len(obs) >= 6:
            ca2 = obs[4]
            sa2 = obs[5]
            if _is_num(ca2) and _is_num(sa2):
                return float(np.arctan2(float(sa2), float(ca2)))
    except Exception:
        pass

    return 0.0


def _get_v_closing0(info: Dict[str, Any], obs: Any) -> float:
    v = info.get("v_closing0", None)
    if _is_num(v):
        return float(v)
    try:
        if obs is not None and len(obs) >= 7:
            v2 = obs[6]
            if _is_num(v2):
                return float(v2)
    except Exception:
        pass
    return 0.0


def _format_replay_line(t: int, a_raw: int, a_eff: int, info: Dict[str, Any], obs_after_step: Any) -> str:
    a_exec = int(info.get("action_exec", info.get("action", a_raw)))
    ov = str(info.get("override_reason", "none"))

    wall = _as_float(info.get("wall_dist_world", info.get("wall_dist0", 99.0)), 99.0)
    d = _as_float(info.get("d_dyn", info.get("d_dyn0", 99.0)), 99.0)
    ttc0 = _as_float(info.get("ttc0", 99.0), 99.0)
    v = _as_float(info.get("v", 0.0), 0.0)
    v_cmd = _as_float(info.get("v_cmd", 0.0), 0.0)

    alpha0 = _get_alpha0(info, obs_after_step)
    v_closing0 = _get_v_closing0(info, obs_after_step)

    wg = bool(info.get("wall_guard_active", False))
    tv = bool(info.get("trap_vcap_active", False))
    stop = bool(info.get("stop_no_gap", False))
    runover = _pick_runover_flag(info)

    return (
        f"t={t:03d} "
        f"a_raw={a_raw} a_eff={a_eff} a_exec={a_exec} "
        f"ov={ov} "
        f"wall={wall:.2f} d={d:.2f} ttc0={ttc0:.2f} "
        f"alpha0={alpha0:+.2f} vclose0={v_closing0:+.2f} "
        f"v={v:.2f} v_cmd={v_cmd:.2f} "
        f"wg={wg} tv={tv} stop={stop} runover={runover}"
    )
def run_episode(
    model: PPO,
    seed: int,
    render: bool = False,
    deterministic: bool = True,
    p_mirror: float = 0.0,
    obs_dim: int = 7,
    print_info_keys: bool = False,
    corridor_half_width: float = 0.9,
    randomize_corridor: bool = False,
    corridor_hw_min: float = 0.55,
    corridor_hw_max: float = 1.20,
    # NEW
    env_kwargs_extra: Optional[Dict[str, Any]] = None,
    # ===== replay debug options =====
    replay_debug: bool = False,
    replay_log_mode: str = "interesting",  # interesting | near | all
    replay_print_last: int = 12,
    replay_near_d: float = 1.0,
    replay_near_wall: float = 0.25,
    replay_slow_vcmd: float = 0.25,
    replay_slow_v: float = 0.18,
) -> EpisodeStats:
    env = make_env(
        seed=seed,
        render=render,
        p_mirror=p_mirror,
        obs_dim=obs_dim,
        corridor_half_width=corridor_half_width,
        randomize_corridor=randomize_corridor,
        corridor_hw_min=corridor_hw_min,
        corridor_hw_max=corridor_hw_max,
        env_kwargs_extra=env_kwargs_extra,
    )
    obs, info0 = env.reset(seed=seed)

    mirrored = bool(info0.get("mirrored", False))
    scenario = str(info0.get("scenario", "unknown"))
    conflict0 = bool(info0.get("conflict", False))

    nA = int(env.action_space.n)
    act_policy_raw = np.zeros(nA, dtype=np.int64)
    act_policy_eff = np.zeros(nA, dtype=np.int64)
    act_exec = np.zeros(nA, dtype=np.int64)

    act_policy_raw_rl = np.zeros(nA, dtype=np.int64)
    act_policy_eff_rl = np.zeros(nA, dtype=np.int64)
    act_exec_rl = np.zeros(nA, dtype=np.int64)

    steps = 0
    steps_rl = 0
    mismatch_eff_exec = 0

    ct_sum = 0.0
    ct_max = 0.0
    ct_n = 0

    wall_seen = False
    wall_dist_min = float("inf")
    wall_dist_norm_min = float("inf")
    wall_pen_sum = 0.0
    wall_v_pen_sum = 0.0
    wall_risk_sum = 0.0
    wall_risk_n = 0

    override_brake_steps = 0
    override_wall_steps = 0
    override_cpa_steps = 0
    brake_wall_bonus_sum = 0.0

    d_coll_min = float("inf")
    d_safe_min = float("inf")
    obstacle_clear_min = float("inf")

    wall_safe = float("nan")
    d_coll = float("nan")
    d_safe = float("nan")

    done = False
    trunc = False
    last_info: Dict = {}

    replay_lines: List[str] = []

    def _push_replay_line(line: str):
        replay_lines.append(line)
        lim = max(50, int(replay_print_last) * 5)
        if len(replay_lines) > lim:
            del replay_lines[: len(replay_lines) - lim]

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=deterministic)
        a_raw = int(action)
        a_eff = mirror_action(a_raw) if mirrored else a_raw

        obs, r, done, trunc, last_info = env.step(a_raw)

        if replay_debug:
            wall = _as_float(last_info.get("wall_dist_world", last_info.get("wall_dist0", 99.0)), 99.0)
            d = _as_float(last_info.get("d_dyn", last_info.get("d_dyn0", 99.0)), 99.0)
            v = _as_float(last_info.get("v", 0.0), 0.0)
            v_cmd = _as_float(last_info.get("v_cmd", 0.0), 0.0)

            near = (d < float(replay_near_d)) or (wall < float(replay_near_wall))
            slow = (v_cmd < float(replay_slow_vcmd)) or (v < float(replay_slow_v))

            if replay_log_mode == "all":
                log_it = True
            elif replay_log_mode == "near":
                log_it = near
            else:
                log_it = near and slow

            if log_it:
                _push_replay_line(_format_replay_line(steps, a_raw, a_eff, last_info, obs))

        a_exec = int(last_info.get("action", a_raw))

        if 0 <= a_raw < nA:
            act_policy_raw[a_raw] += 1
        if 0 <= a_eff < nA:
            act_policy_eff[a_eff] += 1
        if 0 <= a_exec < nA:
            act_exec[a_exec] += 1

        if a_eff != a_exec:
            mismatch_eff_exec += 1

        steps += 1

        ct = last_info.get("cross_track", None)
        if _is_num(ct):
            ct = float(ct)
            ct_sum += ct
            ct_max = max(ct_max, ct)
            ct_n += 1

        rl_active = last_info.get("rl_active", None)
        if _is_bool(rl_active) and bool(rl_active):
            steps_rl += 1
            if 0 <= a_raw < nA:
                act_policy_raw_rl[a_raw] += 1
            if 0 <= a_eff < nA:
                act_policy_eff_rl[a_eff] += 1
            if 0 <= a_exec < nA:
                act_exec_rl[a_exec] += 1

        wdist = last_info.get("wall_dist_world", None)
        if _is_num(wdist):
            wall_seen = True
            wall_dist_min = min(wall_dist_min, float(wdist))

        wdistn = last_info.get("wall_dist_norm_world", None)
        if _is_num(wdistn):
            wall_dist_norm_min = min(wall_dist_norm_min, float(wdistn))

        wpen = last_info.get("wall_pen", None)
        if _is_num(wpen):
            wall_pen_sum += float(wpen)

        wvpen = last_info.get("wall_v_pen", None)
        if _is_num(wvpen):
            wall_v_pen_sum += float(wvpen)

        wrisk = last_info.get("wall_risk", None)
        if _is_num(wrisk):
            wall_risk_sum += float(wrisk)
            wall_risk_n += 1

        ob = last_info.get("override_brake", None)
        if _is_bool(ob) and bool(ob):
            override_brake_steps += 1
            reason = str(last_info.get("override_reason", "unknown"))
            if reason == "wall_hard_brake":
                override_wall_steps += 1
            elif reason == "cpa_brake":
                override_cpa_steps += 1

        bbw = last_info.get("brake_wall_bonus", None)
        if _is_num(bbw):
            brake_wall_bonus_sum += float(bbw)

        dc = last_info.get("d_coll", None)
        if _is_num(dc):
            dc = float(dc)
            d_coll_min = min(d_coll_min, dc)
            d_coll = dc

        ds = last_info.get("d_safe", None)
        if _is_num(ds):
            ds = float(ds)
            d_safe_min = min(d_safe_min, ds)
            d_safe = ds

        oc = last_info.get("obstacle_clearance", None)
        if _is_num(oc):
            obstacle_clear_min = min(obstacle_clear_min, float(oc))

        ws = last_info.get("wall_safe", None)
        if _is_num(ws):
            wall_safe = float(ws)

        if render:
            env.render()

    if print_info_keys:
        print("INFO KEYS:", sorted(list(last_info.keys())))

    reached = bool(last_info.get("reached", False))
    collision = bool(last_info.get("collision", False))
    collision_wall = bool(last_info.get("collision_wall", False))
    collision_dyn = bool(last_info.get("collision_dyn", False))
    conflict = bool(last_info.get("conflict", conflict0))

    if reached:
        reason = "reached"
    elif collision:
        reason = "collision"
    elif trunc:
        reason = "truncated"
    else:
        reason = "not_reached"

    if replay_debug:
        print(f"--- replay_tail (mode={replay_log_mode}, last={replay_print_last}) ---")
        tail = replay_lines[-int(replay_print_last):] if replay_print_last > 0 else replay_lines
        if len(tail) == 0:
            print("(no replay lines matched filters; try --replay-log-mode near o all)")
        else:
            for ln in tail:
                print(ln)

    dt = float(getattr(env, "dt", 0.05))
    env.close()

    if wall_dist_min == float("inf"):
        wall_dist_min = 0.0
    if wall_dist_norm_min == float("inf"):
        wall_dist_norm_min = 0.0
    if d_coll_min == float("inf"):
        d_coll_min = 0.0
    if d_safe_min == float("inf"):
        d_safe_min = 0.0
    if obstacle_clear_min == float("inf"):
        obstacle_clear_min = 0.0

    cross_track_mean = _safe_div(ct_sum, ct_n)
    cross_track_max = float(ct_max) if ct_n > 0 else 0.0
    wall_risk_mean = _safe_div(wall_risk_sum, wall_risk_n)

    return EpisodeStats(
        steps=int(steps),
        steps_rl=int(steps_rl),
        dt=dt,
        mirrored=bool(mirrored),
        scenario=str(last_info.get("scenario", scenario)),
        conflict=bool(conflict),
        reached=bool(reached),
        collision=bool(collision),
        collision_wall=bool(collision_wall),
        collision_dyn=bool(collision_dyn),
        truncated=bool(trunc),
        reason=str(reason),
        act_policy_raw=act_policy_raw,
        act_policy_effective=act_policy_eff,
        act_exec=act_exec,
        act_policy_raw_rl=act_policy_raw_rl,
        act_policy_effective_rl=act_policy_eff_rl,
        act_exec_rl=act_exec_rl,
        mismatch_effective_vs_exec=int(mismatch_eff_exec),
        cross_track_mean=float(cross_track_mean),
        cross_track_max=float(cross_track_max),
        wall_seen=bool(wall_seen),
        wall_dist_min=float(wall_dist_min),
        wall_dist_norm_min=float(wall_dist_norm_min),
        wall_pen_sum=float(wall_pen_sum),
        wall_risk_mean=float(wall_risk_mean),
        wall_v_pen_sum=float(wall_v_pen_sum),
        override_brake_steps=int(override_brake_steps),
        override_wall_steps=int(override_wall_steps),
        override_cpa_steps=int(override_cpa_steps),
        brake_wall_bonus_sum=float(brake_wall_bonus_sum),
        d_coll_min=float(d_coll_min),
        d_safe_min=float(d_safe_min),
        obstacle_clearance_min=float(obstacle_clear_min),
        wall_safe=float(wall_safe),
        d_coll=float(d_coll),
        d_safe=float(d_safe),
    )
def eval_model(
    model_path: str,
    n_episodes: int = 200,
    seed_base: int = 123,
    replay_fails: bool = False,
    max_replays: int = 20,
    p_mirror: float = 0.0,
    deterministic: bool = True,
    print_info_keys: bool = False,
    wall_touch_th: float = 0.02,
    corridor_half_width: float = 0.9,
    randomize_corridor: bool = False,
    corridor_hw_min: float = 0.55,
    corridor_hw_max: float = 1.20,
    # NEW
    env_kwargs_extra: Optional[Dict[str, Any]] = None,
    # replay debug config (CLI)
    replay_log_mode: str = "interesting",
    replay_print_last: int = 12,
    replay_near_d: float = 1.0,
    replay_near_wall: float = 0.25,
    replay_slow_vcmd: float = 0.25,
    replay_slow_v: float = 0.18,
):
    print("model:", model_path)
    print("p_mirror:", float(p_mirror))
    print("deterministic:", bool(deterministic))
    print("corridor_half_width:", float(corridor_half_width))
    print("randomize_corridor:", bool(randomize_corridor))
    print("env_kwargs_extra:", env_kwargs_extra if env_kwargs_extra else {})

    model = PPO.load(model_path, device="cpu")
    obs_dim = int(model.observation_space.shape[0])
    print("model_obs_dim:", obs_dim)

    reached_count = 0
    collided_count = 0
    collided_wall_count = 0
    collided_dyn_count = 0
    conflict_count = 0
    truncated_count = 0
    mirrored_eps = 0

    steps_to_end_sum = 0
    dt_ref: Optional[float] = None

    nA = 7
    act_policy_raw_total = np.zeros(nA, dtype=np.int64)
    act_policy_eff_total = np.zeros(nA, dtype=np.int64)
    act_exec_total = np.zeros(nA, dtype=np.int64)

    act_policy_raw_total_rl = np.zeros(nA, dtype=np.int64)
    act_policy_eff_total_rl = np.zeros(nA, dtype=np.int64)
    act_exec_total_rl = np.zeros(nA, dtype=np.int64)

    steps_total = 0
    steps_total_rl = 0
    mismatch_total = 0

    ct_mean_sum = 0.0
    ct_max_sum = 0.0

    wall_dist_min_sum = 0.0
    wall_dist_norm_min_sum = 0.0
    wall_pen_sum = 0.0
    wall_risk_mean_sum = 0.0
    wall_v_pen_sum = 0.0
    override_brake_steps_sum = 0
    override_wall_steps_sum = 0
    override_cpa_steps_sum = 0
    brake_wall_bonus_sum = 0.0

    d_coll_min_sum = 0.0
    d_safe_min_sum = 0.0
    obstacle_clear_min_sum = 0.0

    ep_count = 0

    wall_seen_eps = 0
    wall_touch_eps = 0
    wall_safe_zone_eps = 0

    fail_seeds: List[int] = []
    fail_reasons: List[str] = []

    for ep in range(n_episodes):
        seed = seed_base + ep
        st = run_episode(
            model,
            seed=seed,
            render=False,
            deterministic=deterministic,
            p_mirror=p_mirror,
            obs_dim=obs_dim,
            print_info_keys=(print_info_keys and ep == 0),
            corridor_half_width=corridor_half_width,
            randomize_corridor=randomize_corridor,
            corridor_hw_min=corridor_hw_min,
            corridor_hw_max=corridor_hw_max,
            env_kwargs_extra=env_kwargs_extra,
            replay_debug=False,
        )

        ep_count += 1

        reached_count += int(st.reached)
        collided_count += int(st.collision)
        collided_wall_count += int(st.collision_wall)
        collided_dyn_count += int(st.collision_dyn)
        conflict_count += int(st.conflict)
        truncated_count += int(st.truncated)
        mirrored_eps += int(st.mirrored)

        steps_to_end_sum += int(st.steps)
        if dt_ref is None:
            dt_ref = float(st.dt)

        steps_total += int(st.steps)
        steps_total_rl += int(st.steps_rl)
        mismatch_total += int(st.mismatch_effective_vs_exec)

        act_policy_raw_total += st.act_policy_raw
        act_policy_eff_total += st.act_policy_effective
        act_exec_total += st.act_exec

        act_policy_raw_total_rl += st.act_policy_raw_rl
        act_policy_eff_total_rl += st.act_policy_effective_rl
        act_exec_total_rl += st.act_exec_rl

        ct_mean_sum += float(st.cross_track_mean)
        ct_max_sum += float(st.cross_track_max)

        wall_dist_min_sum += float(st.wall_dist_min)
        wall_dist_norm_min_sum += float(st.wall_dist_norm_min)
        wall_pen_sum += float(st.wall_pen_sum)
        wall_risk_mean_sum += float(st.wall_risk_mean)
        wall_v_pen_sum += float(st.wall_v_pen_sum)
        override_brake_steps_sum += int(st.override_brake_steps)
        override_wall_steps_sum += int(st.override_wall_steps)
        override_cpa_steps_sum += int(st.override_cpa_steps)
        brake_wall_bonus_sum += float(st.brake_wall_bonus_sum)

        d_coll_min_sum += float(st.d_coll_min)
        d_safe_min_sum += float(st.d_safe_min)
        obstacle_clear_min_sum += float(st.obstacle_clearance_min)

        if st.wall_seen:
            wall_seen_eps += 1
            if float(st.wall_dist_min) < float(wall_touch_th):
                wall_touch_eps += 1
            if _is_num(st.wall_safe) and float(st.wall_dist_min) < float(st.wall_safe):
                wall_safe_zone_eps += 1

        if not st.reached:
            fail_seeds.append(seed)
            fail_reasons.append(st.reason)

    avg_steps_to_end = _safe_div(steps_to_end_sum, n_episodes)
    dt_use = 0.05 if dt_ref is None else float(dt_ref)
    avg_time_to_end_s = avg_steps_to_end * dt_use

    print("\nepisodes:", n_episodes)
    print("mirror_rate:", _safe_div(mirrored_eps, n_episodes))
    print("conflict_rate:", _safe_div(conflict_count, n_episodes))
    print("reach_rate:", _safe_div(reached_count, n_episodes))
    print("collision_rate:", _safe_div(collided_count, n_episodes))
    print("collision_wall_rate:", _safe_div(collided_wall_count, n_episodes))
    print("collision_dyn_rate:", _safe_div(collided_dyn_count, n_episodes))
    print("truncated_rate:", _safe_div(truncated_count, n_episodes))
    print(f"avg_steps_to_end: {avg_steps_to_end:.2f} steps")
    print(f"avg_time_to_end:  {avg_time_to_end_s:.2f} s   (dt={dt_use})")

    if steps_total > 0:
        print("mismatch_eff_vs_exec_rate:", _safe_div(mismatch_total, steps_total))

    print("\n=== cross-track (proxy de 'se abre') ===")
    print("avg_cross_track_mean:", _safe_div(ct_mean_sum, ep_count))
    print("avg_cross_track_max:", _safe_div(ct_max_sum, ep_count))

    print("\n=== wall/clearance diagnostics (episode-averaged) ===")
    print("avg_wall_dist_min (m):", _safe_div(wall_dist_min_sum, ep_count))
    print("avg_wall_dist_norm_min:", _safe_div(wall_dist_norm_min_sum, ep_count))
    print("avg_wall_pen_sum:", _safe_div(wall_pen_sum, ep_count))
    print("avg_wall_risk_mean:", _safe_div(wall_risk_mean_sum, ep_count))
    print("avg_wall_v_pen_sum:", _safe_div(wall_v_pen_sum, ep_count))
    print("avg_override_brake_steps:", _safe_div(override_brake_steps_sum, ep_count))
    print("avg_override_wall_steps:", _safe_div(override_wall_steps_sum, ep_count))
    print("avg_override_cpa_steps:", _safe_div(override_cpa_steps_sum, ep_count))
    print("avg_brake_wall_bonus_sum:", _safe_div(brake_wall_bonus_sum, ep_count))

    print("\n=== wall contact rates ===")
    print("wall_touch_th (m):", float(wall_touch_th))
    print("wall_seen_rate:", _safe_div(wall_seen_eps, ep_count))
    print("wall_touch_rate (wall_dist_min < th):", _safe_div(wall_touch_eps, wall_seen_eps))
    print("wall_safe_zone_rate (wall_dist_min < wall_safe):", _safe_div(wall_safe_zone_eps, wall_seen_eps))

    print("\n=== obstacle clearance (episode-averaged) ===")
    print("avg_d_coll_min (m):", _safe_div(d_coll_min_sum, ep_count))
    print("avg_d_safe_min (m):", _safe_div(d_safe_min_sum, ep_count))
    print("avg_obstacle_clearance_min (m):", _safe_div(obstacle_clear_min_sum, ep_count))

    def _print_dist(title: str, counts: np.ndarray, denom: int):
        print(f"\n=== {title} ===")
        for a in range(nA):
            pct = 100.0 * _safe_div(int(counts[a]), denom)
            print(f"a{a} {ACTIONS[a]:>10s}: {pct:6.2f}%   (count={int(counts[a])})")

    _print_dist("action distribution (policy RAW / predicted, all steps)", act_policy_raw_total, steps_total)
    _print_dist("action distribution (policy EFFECTIVE, all steps)", act_policy_eff_total, steps_total)
    _print_dist("action distribution (executed, all steps)", act_exec_total, steps_total)

    if steps_total_rl > 0:
        _print_dist(
            "action distribution (policy RAW / predicted, rl_active steps only)",
            act_policy_raw_total_rl,
            steps_total_rl,
        )
        _print_dist(
            "action distribution (policy EFFECTIVE, rl_active steps only)",
            act_policy_eff_total_rl,
            steps_total_rl,
        )
        _print_dist(
            "action distribution (executed, rl_active steps only)",
            act_exec_total_rl,
            steps_total_rl,
        )

    if replay_fails:
        print("\n--- Replays (solo fallos: NO reached) ---")
        if len(fail_seeds) == 0:
            print("No hay fallos para reproducir ✅")
            return

        n = min(len(fail_seeds), max_replays)
        print(f"Voy a reproducir {n}/{len(fail_seeds)} fallos con render...")

        for i in range(n):
            seed = fail_seeds[i]
            reason = fail_reasons[i]
            print(f"\nReplay {i+1}/{n} | seed={seed} | reason={reason}")

            st = run_episode(
                model,
                seed=seed,
                render=True,
                deterministic=deterministic,
                p_mirror=p_mirror,
                obs_dim=obs_dim,
                print_info_keys=False,
                corridor_half_width=corridor_half_width,
                randomize_corridor=randomize_corridor,
                corridor_hw_min=corridor_hw_min,
                corridor_hw_max=corridor_hw_max,
                env_kwargs_extra=env_kwargs_extra,
                replay_debug=True,
                replay_log_mode=replay_log_mode,
                replay_print_last=replay_print_last,
                replay_near_d=replay_near_d,
                replay_near_wall=replay_near_wall,
                replay_slow_vcmd=replay_slow_vcmd,
                replay_slow_v=replay_slow_v,
            )
            print(
                "end:",
                {
                    "reached": st.reached,
                    "collision": st.collision,
                    "collision_wall": st.collision_wall,
                    "collision_dyn": st.collision_dyn,
                    "truncated": st.truncated,
                    "mirrored": st.mirrored,
                    "conflict": st.conflict,
                    "scenario": st.scenario,
                    "reason": st.reason,
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--p-mirror", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--replay-fails", action="store_true")
    parser.add_argument("--max-replays", type=int, default=20)
    parser.add_argument("--print-info-keys", action="store_true", help="Imprime INFO KEYS del primer episodio")

    parser.add_argument("--wall-touch-th", type=float, default=0.02, help="Umbral (m) para 'rozó pared' (wall_dist_min < th)")

    parser.add_argument("--corridor-half-width", type=float, default=0.9, help="Half-width fijo (m). Ignorado si --randomize-corridor")
    parser.add_argument("--randomize-corridor", action="store_true", help="Randomiza el half-width por episodio")
    parser.add_argument("--corridor-hw-min", type=float, default=0.55, help="Half-width mínimo (m) si randomize")
    parser.add_argument("--corridor-hw-max", type=float, default=1.20, help="Half-width máximo (m) si randomize")

    # ===== replay debug flags =====
    parser.add_argument(
        "--replay-log-mode",
        type=str,
        default="interesting",
        choices=["interesting", "near", "all"],
        help="Qué pasos loguear durante replay: interesting=(near & slow), near=(near), all=(todos).",
    )
    parser.add_argument("--replay-print-last", type=int, default=12, help="Cuántas líneas finales del buffer imprimir por replay.")
    parser.add_argument("--replay-near-d", type=float, default=1.0, help="Umbral de cercanía al dinámico (m) para near.")
    parser.add_argument("--replay-near-wall", type=float, default=0.25, help="Umbral de cercanía a pared (m) para near.")
    parser.add_argument("--replay-slow-vcmd", type=float, default=0.25, help="Umbral de v_cmd baja para 'slow'.")
    parser.add_argument("--replay-slow-v", type=float, default=0.18, help="Umbral de v baja para 'slow'.")

    # ===== NEW: passthrough kwargs al env =====
    parser.add_argument(
        "--env-kw",
        action="append",
        default=[],
        help="Passthrough kwargs al env: --env-kw k=v (repetible). Ej: --env-kw anti_runover_enable=0",
    )

    args = parser.parse_args()

    det = True
    if args.stochastic:
        det = False
    if args.deterministic:
        det = True

    env_kwargs_extra = _parse_env_kw_list(args.env_kw)

    eval_model(
        model_path=args.model,
        n_episodes=args.episodes,
        seed_base=args.seed_base,
        replay_fails=args.replay_fails,
        max_replays=args.max_replays,
        p_mirror=args.p_mirror,
        deterministic=det,
        print_info_keys=bool(args.print_info_keys),
        wall_touch_th=float(args.wall_touch_th),
        corridor_half_width=float(args.corridor_half_width),
        randomize_corridor=bool(args.randomize_corridor),
        corridor_hw_min=float(args.corridor_hw_min),
        corridor_hw_max=float(args.corridor_hw_max),
        env_kwargs_extra=env_kwargs_extra,
        replay_log_mode=str(args.replay_log_mode),
        replay_print_last=int(args.replay_print_last),
        replay_near_d=float(args.replay_near_d),
        replay_near_wall=float(args.replay_near_wall),
        replay_slow_vcmd=float(args.replay_slow_vcmd),
        replay_slow_v=float(args.replay_slow_v),
    )
