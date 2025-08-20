import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train_targets import Simple1DCNN


class InferenceDataset(Dataset):
    """Dataset for inference inputs."""

    def __init__(self, samples: List[Dict[str, object]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        matrix = torch.tensor(sample["input_matrix"], dtype=torch.float32)
        return sample["id"], matrix


def load_models(
    model_paths: List[str], input_size: int, device: torch.device
) -> Dict[str, Simple1DCNN]:
    """Load models from disk.

    Args:
        model_paths: List of model file paths.
        input_size: Width of the input matrix.
        device: Torch device.

    Returns:
        Mapping from target name to loaded model.
    """
    models = {}
    for path in model_paths:
        target = os.path.splitext(os.path.basename(path))[0]
        model = Simple1DCNN(input_size=input_size)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models[target] = model
    return models


def main() -> None:
    """Run batch inference on new inputs."""
    parser = argparse.ArgumentParser(description="Batch inference")
    parser.add_argument(
        "--model", nargs="+", required=True, help="Model file paths"
    )
    parser.add_argument("--inputs", required=True, help="Path to inputs JSON")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    args = parser.parse_args()

    with open(args.inputs, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dataset = InferenceDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    input_size = len(samples[0]["input_matrix"][0])
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    models = load_models(args.model, input_size, device)

    results = {"id": []}
    for target in models:
        results[target + "_pred"] = []

    with torch.no_grad():
        for ids, matrices in loader:
            matrices = matrices.to(device)
            results["id"].extend(ids)
            for target, model in models.items():
                preds = model(matrices).cpu().numpy().flatten().tolist()
                results[target + "_pred"].extend(preds)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
