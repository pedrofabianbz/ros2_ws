#!/usr/bin/env python3
import numpy as np

from simulador_2d import World2D, H, W, LABEL_IGNORE, LABEL_DYNAMIC, LABEL_STATIC

K = 2          # frames de historia
N_STEPS = 2000 # pasos de simulación
DT = 0.1       # igual que en simulador


def main():
    world = World2D()

    occ_history = []
    lab_history = []

    # Simulamos N_STEPS y guardamos todo
    for t in range(N_STEPS):
        occ, lab = world.step(DT)
        occ_history.append(occ.astype(np.int8))
        lab_history.append(lab.astype(np.int8))

    occ_history = np.stack(occ_history, axis=0)   # (T, H, W)
    lab_history = np.stack(lab_history, axis=0)   # (T, H, W)
    T = occ_history.shape[0]

    X_list = []
    Y_list = []

    for t in range(K - 1, T):
        # ventana [t-K+1 ... t] -> entrada
        x_seq = occ_history[t-K+1:t+1]  # (K, H, W)
        y_lab = lab_history[t]          # (H, W)

        # Aquí podrías filtrar solo frames con dinámicos si quisieras
        X_list.append(x_seq)
        Y_list.append(y_lab)

    X = np.stack(X_list, axis=0)  # (N, K, H, W)
    Y = np.stack(Y_list, axis=0)  # (N, H, W)

    np.savez_compressed("dataset_dyn_cells.npz", X=X, Y=Y)
    print("✅ Dataset guardado en dataset_dyn_cells.npz")
    print("   X:", X.shape, "Y:", Y.shape)

    total_dyn = np.sum(Y == LABEL_DYNAMIC)
    total_static = np.sum(Y == LABEL_STATIC)
    total_ignore = np.sum(Y == LABEL_IGNORE)
    print(f"   dinámicas: {total_dyn}, estáticas: {total_static}, ignore: {total_ignore}")


if __name__ == "__main__":
    main()
