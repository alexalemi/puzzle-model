#!/usr/bin/env python3
"""Compress explorer_data.json by rounding floats and subsampling large arrays."""

import json
import random
import sys
from pathlib import Path

def round_dict(d, decimals):
    """Round all float values in a dict to given decimal places."""
    return {k: round(v, decimals) if isinstance(v, float) else v for k, v in d.items()}

def subsample(lst, n, seed=42):
    """Deterministically subsample a list to n items."""
    if len(lst) <= n:
        return lst
    rng = random.Random(seed)
    return rng.sample(lst, n)

def main():
    path = Path(__file__).resolve().parent.parent / "explorer_data.json"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    data = json.loads(path.read_text())
    original_size = path.stat().st_size

    # 1. Puzzlers: round floats to 3 decimal places
    data["puzzlers"] = [round_dict(p, 3) for p in data["puzzlers"]]

    # 2. Scatter: subsample to 1500, round to 2 decimals
    data["scatter"] = [round_dict(p, 2) for p in subsample(data["scatter"], 1500)]

    # 3. Models: subsample residuals to 500, round to 3 decimals; round loss to 1 decimal
    for model_key, model_data in data.get("models", {}).items():
        if "residuals" in model_data:
            model_data["residuals"] = [
                round_dict(r, 3) for r in subsample(model_data["residuals"], 500)
            ]
        if "loss_curve" in model_data:
            model_data["loss_curve"] = [round(v, 1) for v in model_data["loss_curve"]]

    # 4. Puzzles: round floats to 3 decimal places
    if "puzzles" in data:
        data["puzzles"] = [round_dict(p, 3) for p in data["puzzles"]]

    # Write back with compact separators
    output = json.dumps(data, separators=(",", ":"))
    path.write_text(output)

    new_size = len(output.encode())
    print(f"Original: {original_size:,} bytes")
    print(f"Compressed: {new_size:,} bytes")
    print(f"Reduction: {(1 - new_size / original_size) * 100:.1f}%")

if __name__ == "__main__":
    main()
