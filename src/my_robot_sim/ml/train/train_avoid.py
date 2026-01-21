# ml/train/train_avoid.py
import os
from typing import Callable

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sim.training.env_avoid_corridor_train import AvoidCorridorTrainEnv

START_MODEL_PATH = "/home/pedro/ros2_ws/src/my_robot_sim/ml/models/avoid_corridor_continue_10d_trapwall.zip"

"""
train_avoid.py (ml/train/train_avoid.py)
=======================================

Qué hace
--------
Script de entrenamiento (o *fine-tuning*) con Stable-Baselines3 **PPO** para la política discreta
de evasión en pasillo usando el entorno:

    sim.training.env_avoid_corridor_train.AvoidCorridorTrainEnv

Este archivo está pensado para **continuar** entrenamiento desde un modelo previo (`START_MODEL_PATH`)
y mejorar comportamiento sin “olvidar” lo ya aprendido, guardando:

- Checkpoints periódicos en `ml/models/`
- Logs de TensorBoard en `ml/tb_avoid/`
- El “best model” según evaluación en `ml/models/`

Cómo se corre
-------------
Desde la carpeta `ml/` (recomendado):

    python3 -m train.train_avoid

(o equivalente si lo ejecutas directo, según tu PYTHONPATH).

Requisitos:
- `stable-baselines3`, `gymnasium`, `torch`
- Tu repo debe resolver imports de `sim.*`

Inputs (configurables en el archivo)
------------------------------------
1) START_MODEL_PATH
   Ruta al `.zip` del modelo PPO desde el que vas a continuar.

2) make_env(train=True/False)
   Crea instancias del entorno con hiperparámetros específicos:
   - `train=True`: más augment (p_mirror alto), distribución de escenarios para aprender robustez.
   - `train=False`: eval con p_mirror un poco menor (más “realista” y estable como métrica).

3) n_envs
   Número de entornos paralelos (DummyVecEnv). Aquí: `n_envs = 8`.

4) Hiperparámetros PPO ajustados para “mejorar sin destruir”:
   - learning_rate: constante baja (2.5e-5)
   - ent_coef: 0.0025 (sube exploración, empuja a steering y evita colapso a brake)
   - clip_range: 0.10 (updates conservadores)
   - target_kl: 0.02 (freno anti-cambios bruscos)

Qué está entrenando exactamente
-------------------------------
- Acción: Discrete(7) (keep/steer/brake/boost) definida por el env.
- Observación: 10D (porque `use_corridor_obs=True`).
- El env incluye gating `rl_active` y shaping/overrides (CPA, trap-wall, front-conflict, anti-runover, etc.).
  O sea: PPO aprende “correcciones” y el entorno fuerza seguridad/consistencia.

Callbacks / outputs generados
-----------------------------
1) CheckpointCallback
   - `save_freq=50_000` pasos de entrenamiento
   - Guarda en `models/` con prefijo:
       avoid_corridor_continue_10d_round4_front_*.zip

2) EvalCallback
   - Evalúa cada `eval_freq=25_000` pasos
   - Corre `n_eval_episodes=25`
   - Guarda el mejor modelo en `models/` (best_model_save_path)
   - Loguea métricas en `tb_avoid/`

3) TensorBoard
   - `tensorboard_log="tb_avoid"`
   Luego puedes ver:
       tensorboard --logdir tb_avoid

Salida final:
-------------
Al terminar `model.learn(...)`, guarda el modelo final en:

    models/avoid_corridor_continue_10d_round4_front.zip

y lo imprime por consola.

En una frase
------------
Este script levanta 8 entornos de pasillo, carga un PPO ya entrenado y lo sigue entrenando con
updates suaves + evaluación periódica, dejando checkpoints y trazas para TensorBoard.
"""


def const_schedule(val: float) -> Callable[[float], float]:
    v = float(val)
    return lambda _progress: v


def make_env(train: bool = True):
    def _init():
        return AvoidCorridorTrainEnv(
            sdf_world="../worlds/diff_drive_empty.sdf",
            render_mode=None,

            # distribución
            p_conflict=0.85 if train else 0.85,
            v_dyn_min=0.15,
            v_dyn_max=0.8,
            v_nom=0.4,
            max_steps=250,

            # selector (un poco más temprano = más tiempo para “abrirse”)
            risk_radius=1.0,
            activate_radius=0.85,
            deactivate_radius=1.25,

            # RL
            k_rl=4.0,
            rl_decimation=1,

            # mirror (sube para matar sesgo izquierda/derecha)
            p_mirror=0.60 if train else 0.50,

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
            v_brake=0.15,
            v_boost=0.78,          # un pelín menos para reducir choques “por velocidad”
            k_boost_near=0.60,

            # ===== PATCH: anti-runover (no te quedes en crawl si viene cerrando) =====
            anti_runover_enable=True,
            anti_runover_dist=0.90,
            anti_runover_vclosing_th=0.08,
            anti_runover_v_floor=0.30,
            anti_runover_wall_min=0.12,

            # ===== PATCH: trap STOP selectivo =====
            trap_stop_ttc_th=0.25,
            trap_stop_dist_margin=0.06,

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

            # pasillo fijo (entrenas “en lo real”)
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

            # activación temprana
            rear_end_enable=True,
            rear_end_front_th=0.35,
            rear_end_dist=1.8,
            rear_end_vclosing_th=0.05,

            use_ttc_activation=True,
            ttc_th=4.5,
            ttc_dist_max=2.5,
            ttc_vclosing_min=0.05,

            # anti-conservador (más fuerte el brake “gratis” para romper attractor)
            brake_penalty=0.030,
            stall_penalty=0.015,
            stall_v_th=0.08,
            stall_d_th=1.0,

            # pasa cerca ~0.5
            k_pass_close=0.10,
            pass_close_target=0.25,
            pass_close_width=0.15,

            # ===== FRONT-CONFLICT shaping =====
            front_enable=True,
            front_alpha_th=0.70,
            front_dist=2.0,
            front_vclosing_th=0.12,
            front_ttc_soft=2.2,
            front_brake_penalty=0.060,     # consistente con env: rompe attractor de brake
            front_boost_penalty=0.025,
            front_steer_bonus_small=0.006,
            front_steer_bonus_hard=0.012,
            front_bonus_wall_min=0.18,

            seed=None,
        )
    return _init


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_avoid", exist_ok=True)

    n_envs = 8
    env = DummyVecEnv([make_env(train=True) for _ in range(n_envs)])
    env = VecMonitor(env)

    # Eval env separado (misma dinámica, sin render, y con p_mirror menor)
    eval_env = DummyVecEnv([make_env(train=False)])
    eval_env = VecMonitor(eval_env)

    device = "cuda" if th.cuda.is_available() else "cpu"

    # continuar entrenamiento desde checkpoint
    model = PPO.load(START_MODEL_PATH, env=env, device=device, verbose=1, tensorboard_log="tb_avoid")

    # Hiperparámetros pensados para “mejorar sin olvidar”
    model.learning_rate = const_schedule(2.5e-5)
    model.ent_coef = 0.0025               # más exploración => más steer, menos brake-colapso
    model.clip_range = const_schedule(0.10)
    model.target_kl = 0.02                # freno de cambios bruscos (evita destruir lo bueno)

    ckpt = CheckpointCallback(
        save_freq=50_000,
        save_path="models",
        name_prefix="avoid_corridor_continue_10d_round4_front",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Guarda el “mejor” según reward promedio del eval
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="tb_avoid",
        eval_freq=25_000,
        n_eval_episodes=25,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=1_200_000, callback=[ckpt, eval_cb])

    out = "models/avoid_corridor_continue_10d_round4_front"
    model.save(out)
    print(f"saved -> {out}.zip")
