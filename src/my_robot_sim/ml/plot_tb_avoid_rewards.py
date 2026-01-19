#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TB_DIR = "tb_avoid"
OUT_DIR = "tb_avoid/plots"


def load_eval_npz(npz_path: str):
    data = np.load(npz_path)
    timesteps = data["timesteps"]                 # shape (N,)
    results = data["results"]                     # shape (N, n_eval_episodes)
    ep_lengths = data["ep_lengths"]               # shape (N, n_eval_episodes)

    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)

    mean_len = ep_lengths.mean(axis=1)
    std_len = ep_lengths.std(axis=1)

    return timesteps, mean_r, std_r, mean_len, std_len


def find_latest_event_file(tb_dir: str):
    # Busca el último run PPO_* (por número más alto)
    runs = glob.glob(os.path.join(tb_dir, "PPO_*"))
    if not runs:
        return None

    def run_id(p):
        base = os.path.basename(p)
        try:
            return int(base.split("_")[1])
        except Exception:
            return -1

    runs = sorted(runs, key=run_id)
    latest = runs[-1]

    ev = glob.glob(os.path.join(latest, "events.out.tfevents.*"))
    if not ev:
        return None

    # Si hay más de uno, toma el más reciente por mtime
    ev = sorted(ev, key=lambda p: os.path.getmtime(p))
    return ev[-1]


def read_scalar_from_event(event_path: str, tag: str):
    # Carga SOLO lo necesario (más rápido)
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()

    tags = hint_tags(ea)
    if tag not in tags:
        return None, None

    scalars = ea.Scalars(tag)
    steps = np.array([s.step for s in scalars], dtype=np.int64)
    vals = np.array([s.value for s in scalars], dtype=np.float64)
    return steps, vals


def hint_tags(ea: EventAccumulator):
    # Compatibilidad según versión TB
    try:
        return ea.Tags().get("scalars", [])
    except Exception:
        return []


def save_csv(path: str, x, y, y2=None, header=("x", "y", "y2")):
    arr = np.column_stack([x, y] if y2 is None else [x, y, y2])
    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="")
    print(f"[OK] CSV -> {path}")


def plot_with_band(x, y, ystd, title, xlabel, ylabel, out_pdf):
    plt.figure()
    plt.plot(x, y)
    plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"[OK] PDF -> {out_pdf}")


def plot_simple(x, y, title, xlabel, ylabel, out_pdf):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"[OK] PDF -> {out_pdf}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- A) Evaluaciones (recomendado para tesis) ---
    npz_path = os.path.join(TB_DIR, "evaluations.npz")
    if os.path.exists(npz_path):
        t, mr, sr, ml, sl = load_eval_npz(npz_path)

        plot_with_band(
            t, mr, sr,
            title="Curva de aprendizaje (Evaluación): reward promedio ± std",
            xlabel="Timesteps",
            ylabel="Mean episode reward (eval)",
            out_pdf=os.path.join(OUT_DIR, "eval_mean_reward.pdf"),
        )
        save_csv(
            os.path.join(OUT_DIR, "eval_mean_reward.csv"),
            t, mr, sr,
            header=("timesteps", "mean_reward", "std_reward")
        )

        plot_with_band(
            t, ml, sl,
            title="Longitud de episodio (Evaluación): media ± std",
            xlabel="Timesteps",
            ylabel="Episode length (steps)",
            out_pdf=os.path.join(OUT_DIR, "eval_ep_len.pdf"),
        )
        save_csv(
            os.path.join(OUT_DIR, "eval_ep_len.csv"),
            t, ml, sl,
            header=("timesteps", "mean_ep_len", "std_ep_len")
        )
    else:
        print(f"[WARN] No encontré {npz_path}")

    # --- B) Training desde tfevents (opcional) ---
    latest_event = find_latest_event_file(TB_DIR)
    if latest_event is None:
        print("[WARN] No encontré events.out.tfevents.* en tb_avoid/PPO_*")
        return

    print(f"[INFO] Usando event file: {latest_event}")

    # Tags típicos en SB3:
    # - rollout/ep_rew_mean
    # - rollout/ep_len_mean
    # - train/entropy_loss, train/value_loss, train/policy_gradient_loss
    # - train/approx_kl, train/clip_fraction, etc.
    steps, vals = read_scalar_from_event(latest_event, "rollout/ep_rew_mean")
    if steps is not None:
        plot_simple(
            steps, vals,
            title="Training: rollout/ep_rew_mean",
            xlabel="Timesteps",
            ylabel="Episode reward mean (train)",
            out_pdf=os.path.join(OUT_DIR, "train_ep_rew_mean.pdf"),
        )
        save_csv(
            os.path.join(OUT_DIR, "train_ep_rew_mean.csv"),
            steps, vals,
            header=("timesteps", "ep_rew_mean")
        )
    else:
        print("[WARN] No encontré tag rollout/ep_rew_mean en el event file (puede variar por versión).")

    steps2, vals2 = read_scalar_from_event(latest_event, "rollout/ep_len_mean")
    if steps2 is not None:
        plot_simple(
            steps2, vals2,
            title="Training: rollout/ep_len_mean",
            xlabel="Timesteps",
            ylabel="Episode length mean (train)",
            out_pdf=os.path.join(OUT_DIR, "train_ep_len_mean.pdf"),
        )
        save_csv(
            os.path.join(OUT_DIR, "train_ep_len_mean.csv"),
            steps2, vals2,
            header=("timesteps", "ep_len_mean")
        )


if __name__ == "__main__":
    main()
