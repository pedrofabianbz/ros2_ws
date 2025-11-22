import numpy as np
import torch

from my_robot_sim.launch.model_cnn import DynCellNet
from simulador_2d import LABEL_DYNAMIC, LABEL_STATIC, LABEL_IGNORE


def main():
    data = np.load("dataset_dyn_cells.npz")
    X = data["X"]  # (N,4,100,100)
    Y = data["Y"]  # (N,100,100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DynCellNet()
    net.load_state_dict(torch.load("dyn_cells_cnn.pth", map_location=device))
    net.to(device)
    net.eval()

    # Evaluamos en unos pocos ejemplos
    N = min(100, X.shape[0])
    X_eval = X[:N]
    Y_eval = Y[:N]

    all_pred = []
    all_true = []

    with torch.no_grad():
        for i in range(N):
            x = torch.tensor(X_eval[i:i+1], dtype=torch.float32, device=device)  # (1,4,100,100)
            y = Y_eval[i]  # (100,100)

            mask = (y != LABEL_IGNORE)
            y_bin = (y == LABEL_DYNAMIC)

            pred = net(x)[0, 0].cpu().numpy()  # (100,100)
            pred_bin = pred > 0.5

            all_pred.append(pred_bin[mask])
            all_true.append(y_bin[mask])

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    acc = (all_pred == all_true).mean()
    dyn_recall = (all_pred[all_true == 1].sum() / max(1, (all_true == 1).sum()))
    dyn_precision = (all_true[all_pred == 1].sum() / max(1, (all_pred == 1).sum()))

    print(f"Accuracy (solo celdas ocupadas): {acc:.3f}")
    print(f"Recall dinámicos: {dyn_recall:.3f}")
    print(f"Precision dinámicos: {dyn_precision:.3f}")


if __name__ == "__main__":
    main()
