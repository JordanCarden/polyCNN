import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from train_targets import Simple1DCNN


class ListDataset(Dataset):
    """Dataset wrapping a list of samples."""

    def __init__(self, data: List[Dict[str, object]], target_key: str):
        self.data = data
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        matrix = torch.tensor(item["input_matrix"], dtype=torch.float32)
        target = torch.tensor([item[self.target_key]], dtype=torch.float32)
        return matrix, target


def train_and_eval(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train model and evaluate on validation set."""
    model = Simple1DCNN(input_size=20).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds.extend(output.cpu().numpy().flatten().tolist())
            gts.extend(target.cpu().numpy().flatten().tolist())

    rmse = float(np.sqrt(mean_squared_error(gts, preds)))
    mae = float(mean_absolute_error(gts, preds))
    r2 = float(r2_score(gts, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2, "preds": preds, "gts": gts}


def plot_results(
    y_true: List[float], y_pred: List[float], fold_dir: str
) -> None:
    """Save parity and residual plots."""
    os.makedirs(fold_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "parity.png"))
    plt.close()

    residuals = np.array(y_pred) - np.array(y_true)
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "residuals.png"))
    plt.close()


def main() -> None:
    """Perform K-fold cross-validation."""
    parser = argparse.ArgumentParser(description="CNN cross-validation")
    parser.add_argument(
        "--data", required=True, help="Processed data directory"
    )
    parser.add_argument("--target", required=True, help="Target name")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data = []
    for split in ("train", "val", "test"):
        path = os.path.join(args.data, f"{split}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data.extend(json.load(f))
    if not data:
        raise ValueError("No data found in provided directory")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.join(args.outdir, "runs", f"cv_{args.target}")
    os.makedirs(base_dir, exist_ok=True)

    metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), start=1):
        train_samples = [data[i] for i in train_idx]
        val_samples = [data[i] for i in val_idx]

        train_loader = DataLoader(
            ListDataset(train_samples, args.target),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ListDataset(val_samples, args.target),
            batch_size=args.batch_size,
            shuffle=False,
        )

        result = train_and_eval(
            train_loader, val_loader, args.epochs, args.lr, device
        )
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        plot_results(result["gts"], result["preds"], fold_dir)
        metrics = {
            k: v for k, v in result.items() if k not in {"gts", "preds"}
        }
        metrics_list.append(metrics)
        with open(
            os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=2)

    aggregate = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    with open(
        os.path.join(base_dir, "aggregate_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate, f, indent=2)


if __name__ == "__main__":
    main()
