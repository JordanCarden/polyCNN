import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from util import MatrixDataset, Simple1DCNN


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    target: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> Tuple[Simple1DCNN, dict, List[str]]:
    """Train a single target model with early stopping.

    Args:
        target: Target name.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        patience: Early stopping patience.
        device: Torch device.

    Returns:
        Tuple of (best model, metrics dict, training log lines).
    """
    model = Simple1DCNN(input_size=20).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    train_log: List[str] = []
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for data, target_vals in train_loader:
            data = data.to(device)
            target_vals = target_vals.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target_vals)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        preds = []
        gts = []
        with torch.no_grad():
            for data, target_vals in val_loader:
                data = data.to(device)
                target_vals = target_vals.to(device)
                outputs = model(data)
                loss = criterion(outputs, target_vals)
                total_val_loss += loss.item() * data.size(0)
                preds.extend(outputs.cpu().numpy().flatten().tolist())
                gts.extend(target_vals.cpu().numpy().flatten().tolist())
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        log_line = (
            f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
            f"val_loss={avg_val_loss:.6f}"
        )
        train_log.append(log_line)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            rmse = float(np.sqrt(mean_squared_error(gts, preds)))
            mae = float(mean_absolute_error(gts, preds))
            r2 = float(r2_score(gts, preds))
            best_metrics = {
                "val_loss": best_val_loss,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_loss": avg_train_loss,
            }
            best_state = model.state_dict()
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, **best_metrics}, train_log


def main() -> None:
    """Train and save one model per target."""
    parser = argparse.ArgumentParser(
        description="Train CNN models for multiple targets"
    )
    parser.add_argument(
        "--data", required=True, help="Directory with processed train/val data"
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["area_avg", "rg_avg", "rdf_peak"],
        help="Target names to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    train_path = os.path.join(args.data, "train.json")
    val_path = os.path.join(args.data, "val.json")

    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "runs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target in args.targets:
        train_dataset = MatrixDataset(train_path, target_key=target)
        val_dataset = MatrixDataset(val_path, target_key=target)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        model, metrics, log_lines = train_one(
            target,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.patience,
            device,
        )

        model_path = os.path.join(args.outdir, "models", f"{target}.pth")
        torch.save(model.state_dict(), model_path)

        run_dir = os.path.join(args.outdir, "runs", target)
        os.makedirs(run_dir, exist_ok=True)
        with open(
            os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=2)
        with open(
            os.path.join(run_dir, "train_log.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
