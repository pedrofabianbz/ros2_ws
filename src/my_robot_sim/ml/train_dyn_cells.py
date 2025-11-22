import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_cnn import DynCellNet
from simulador_2d import LABEL_IGNORE, LABEL_DYNAMIC, LABEL_STATIC


class DynCellsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (K,100,100)
        y = self.Y[idx]  # (100,100)

        mask = (y != LABEL_IGNORE).astype(np.float32)
        y_bin = (y == LABEL_DYNAMIC).astype(np.float32)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_bin, dtype=torch.float32).unsqueeze(0),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        )


def main():
    data = np.load("dataset_dyn_cells.npz")
    X = data["X"]  # (N,K,100,100)
    Y = data["Y"]  # (N,100,100)

    K = X.shape[1]
    print("Usando K =", K)

    ds = DynCellsDataset(X, Y)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando:", device)

    net = DynCellNet(in_channels=K).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCELoss(reduction="none")

    for epoch in range(10):
        total_loss = 0.0
        for x, y, mask in dl:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = net(x)
            loss = bce(pred, y)
            loss = (loss * mask).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss:.4f}")

    torch.save(net.state_dict(), "dyn_cells_cnn.pth")
    print("Modelo guardado en dyn_cells_cnn.pth")


if __name__ == "__main__":
    main()
