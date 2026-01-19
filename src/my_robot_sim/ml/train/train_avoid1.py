# ml/train/train_avoid.py
import os
from typing import Callable

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from sim.training.env_avoid_corridor_train import AvoidCorridorTrainEnv

START_MODEL_PATH = "/home/pedro/ros2_ws/src/my_robot_sim/ml/models/avoid_corridor_continue_10d_trapwall.zip"


def const_schedule(val: float) -> Callable[[float], float]:
    v = float(val)
    return lambda _progress: v


def make_env():
    def _init():
        return AvoidCorridorTrainEnv(
            sdf_world="../worlds/diff_drive_empty.sdf",
            render_mode=None,

            # distribución
            p_conflict=0.8,
            v_dyn_min=0.15,
            v_dyn_max=0.8,
            v_nom=0.4,
            max_steps=250,

            # selector
            risk_radius=0.7,
            activate_radius=0.6,
            deactivate_radius=0.9,

            # RL
            k_rl=4.0,
            rl_decimation=1,

            # mirror
            p_mirror=0.5,

            # paralelo
            p_parallel=0.35,
            parallel_ahead_min=0.8,
            parallel_ahead_max=1.7,
            parallel_lat_min=-0.30,
            parallel_lat_max=0.30,
            parallel_dir_noise=0.30,
            parallel_speed_scale_min=0.85,
            parallel_speed_scale_max=1.20,
            parallel_front_th=0.75,
            parallel_vclose_th=0.12,
            parallel_extra_margin=0.45,

            # acciones
            v_brake=0.0,
            v_boost=0.80,
            k_boost_near=0.60,

            # reward base
            k_progress=1.0,
            time_cost=0.03,
            goal_bonus=20.0,
            finish_bonus_fast=3.0,

            # clearance
            obstacle_clearance=0.20,
            k_risk=3.0,
            k_close=1.2,

            # pared
            wall_safe=0.25,
            k_wall=3.5,
            k_wall_v=8.0,
            k_brake_wall=0.0,
            wall_activate_dist=0.45,
            hard_brake_at_wall_dist=0.14,

            # wall guard/touch
            wall_guard_dist=0.45,
            k_wall_guard_w=5.0,
            wall_touch_th=0.08,
            k_wall_touch=4.0,
            wall_collision_penalty=250.0,
            dyn_collision_penalty=150.0,
            wall_guard_vcap_scale=1.0,

            # TRAP-WALL + Caso A/B
            trap_wall_dist=1.20,
            trap_wall_alpha_min=0.35,
            trap_wall_alpha_max=1.45,
            trap_wall_vlat_min=0.10,
            trap_wall_wall_dist_th=0.18,
            trap_wall_vcap_scale=0.90,
            trap_vn_th=0.12,
            trap_prefer_steer=True,
            trap_brake_if_no_room=False,
            trap_wall_robot_dist_th=0.12,
            trap_disable_boost=True,

            escape_v_floor=0.30,
            escape_d_gate=1.10,
            trap_stop_no_gap=True,
            trap_stop_gap_margin=0.02,

            # 10D
            use_corridor_obs=True,

            # pasillo fijo
            randomize_corridor=False,
            corridor_half_width=0.90,

            # CPA
            use_cpa_brake=True,
            cpa_t_th=0.85,
            cpa_dist_margin=0.12,
            cpa_vlat_th=0.10,
            cpa_vclose_abs_th=0.70,

            # lado malo
            k_bad_side=0.35,

            # ==========================
            # NUEVO: activación temprana
            # ==========================
            rear_end_enable=True,
            rear_end_front_th=0.35,
            rear_end_dist=1.8,
            rear_end_vclosing_th=0.05,

            use_ttc_activation=True,
            ttc_th=4.5,
            ttc_dist_max=2.5,
            ttc_vclosing_min=0.05,

            # ==========================
            # NUEVO: anti-conservador
            # ==========================
            brake_penalty=0.010,
            stall_penalty=0.015,
            stall_v_th=0.08,
            stall_d_th=1.0,

            # ==========================
            # NUEVO: pasa cerca ~0.5
            # ==========================
            k_pass_close=0.10,
            pass_close_target=0.25,
            pass_close_width=0.15,

            seed=None,
        )
    return _init


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_avoid", exist_ok=True)

    env = DummyVecEnv([make_env() for _ in range(8)])
    env = VecMonitor(env)

    device = "cuda" if th.cuda.is_available() else "cpu"

    # continuar entrenamiento desde checkpoint
    model = PPO.load(START_MODEL_PATH, env=env, device=device, verbose=1, tensorboard_log="tb_avoid")

    # “despegue” moderado (si te explota, baja LR)
    model.learning_rate = const_schedule(5.0e-5)
    model.ent_coef = 0.001
    model.clip_range = const_schedule(0.1)

    ckpt = CheckpointCallback(
        save_freq=50_000,
        save_path="models",
        name_prefix="avoid_corridor_continue_10d_round3",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=600_000, callback=ckpt)

    out = "models/avoid_corridor_continue_10d_round3"
    model.save(out)
    print(f"saved -> {out}.zip")
