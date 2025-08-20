import argparse
import json
import os
import random
from typing import Dict, List

import ast

import numpy as np
import pandas as pd


def parse_input_list(input_str: str) -> List[tuple]:
    """Parse an input list string from the CSV.

    Args:
        input_str: String formatted like "[(1, 'E4'), (2, 'S2'), ...]".

    Returns:
        A list of ``(position, type)`` tuples.
    """
    input_str = input_str.strip()
    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError):
        return []


def convert_to_matrix(
    input_list: List[tuple], max_length: int = 20
) -> np.ndarray:
    """Convert parsed tuples into the 3x``max_length`` matrix format.

    Args:
        input_list: Parsed ``(position, type)`` tuples.
        max_length: Maximum sequence length.

    Returns:
        A ``numpy.ndarray`` representing the sequence.
    """
    matrix = np.zeros((3, max_length))
    for pos, type_code in input_list:
        if 1 <= pos <= max_length and type_code not in {"E0", "S0"}:
            idx = pos - 1
            matrix[0, idx] = 1
            if type_code.startswith("E"):
                matrix[1, idx] = int(type_code[1:])
            elif type_code.startswith("S"):
                matrix[2, idx] = int(type_code[1:])
    return matrix


def build_sample(row: pd.Series) -> Dict[str, object]:
    """Build a single dataset sample from a CSV row.

    Args:
        row: A row from the raw CSV.

    Returns:
        A dictionary with the model input matrix and target values.
    """
    input_list = parse_input_list(row["Input List"])
    matrix = convert_to_matrix(input_list)

    sample = {
        "id": row["Name"],
        "input_matrix": matrix.tolist(),
    }

    for col in row.index:
        if col in {"Input List", "Name"}:
            continue
        sample[col] = float(row[col])
    return sample


def compute_stats(
    data: List[Dict[str, object]], target_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute mean and standard deviation for each target."""
    stats: Dict[str, Dict[str, float]] = {}
    for name in target_names:
        values = np.array([d[name] for d in data], dtype=float)
        stats[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return stats


def main() -> None:
    """Entry point for CSV preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess CSV into model-ready datasets"
    )
    parser.add_argument("--csv", required=True, help="Path to raw CSV file")
    parser.add_argument(
        "--outdir", default="Data/processed", help="Output directory"
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for splits"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1, help="Validation fraction"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.1, help="Test fraction"
    )
    args = parser.parse_args()

    if args.val_frac + args.test_frac >= 1:
        raise ValueError("val_frac and test_frac must sum to less than 1")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    samples = [build_sample(row) for _, row in df.iterrows()]

    random.Random(args.split_seed).shuffle(samples)

    n_total = len(samples)
    n_val = int(n_total * args.val_frac)
    n_test = int(n_total * args.test_frac)
    n_train = n_total - n_val - n_test

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    targets = [c for c in df.columns if c not in {"Input List", "Name"}]

    splits = {"train": train, "val": val, "test": test}
    for name, data in splits.items():
        with open(
            os.path.join(args.outdir, f"{name}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2)

    meta = {
        "targets": targets,
        "feature_shape": [3, 20],
        "num_samples": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "seed": args.split_seed,
        "normalization": compute_stats(train, targets),
    }

    with open(
        os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
