#!/usr/bin/env python3
"""Fit Model 4 (Student-t robust) via SVI and add results to explorer_data.json."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import Predictive

from puzzle_model.data import (
    load_solo_completed,
    create_puzzle_id,
    encode_indices,
    train_test_split,
    prepare_model_data,
)
from puzzle_model.model import MODELS
from puzzle_model.inference import run_svi
from puzzle_model.evaluate import evaluate_predictions, naive_baselines
from puzzle_model.basis import compute_basis, normalize_basis


def main():
    # Load and prepare data (same pipeline as other models)
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    train_data = prepare_model_data(train_df)
    test_data = prepare_model_data(test_df)

    # Compute basis normalization from training data
    phi_train = compute_basis(train_data["pieces"])
    _, basis_mean, basis_std = normalize_basis(phi_train)
    train_data["basis_mean"] = basis_mean
    train_data["basis_std"] = basis_std
    test_data["basis_mean"] = basis_mean
    test_data["basis_std"] = basis_std

    print(f"Train: {len(train_df):,} obs, Test: {len(test_df):,} obs")
    print(f"Puzzlers: {train_data['n_puzzlers']:,}, Puzzles: {train_data['n_puzzles']:,}")

    # Fit Model 4 via SVI
    model_fn = MODELS["model_4"]
    print("\nFitting Model 4 (Student-t) via SVI...")
    guide, svi_result = run_svi(model_fn, train_data, num_steps=5000, lr=0.005, seed=0)

    loss_curve = [float(v) for v in svi_result.losses[::10]]  # every 10th step
    print(f"Final loss: {loss_curve[-1]:.1f}")

    # Generate predictions
    rng = jax.random.PRNGKey(1)
    predictive = Predictive(model_fn, guide=guide, params=svi_result.params, num_samples=200)

    # Test predictions
    test_no_obs = {k: v for k, v in test_data.items() if k != "log_time"}
    pred_test = predictive(rng, **test_no_obs)
    test_metrics = evaluate_predictions(
        np.array(pred_test["log_time"]),
        np.array(test_data["log_time"]),
    )

    # Train predictions
    train_no_obs = {k: v for k, v in train_data.items() if k != "log_time"}
    pred_train = predictive(jax.random.PRNGKey(2), **train_no_obs)
    train_metrics = evaluate_predictions(
        np.array(pred_train["log_time"]),
        np.array(train_data["log_time"]),
    )

    print(f"\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Train RMSE (log): {train_metrics['rmse_log']:.4f}")

    # Build residuals (subsample for the explorer)
    pred_mean = np.mean(np.array(pred_test["log_time"]), axis=0)
    true_log = np.array(test_data["log_time"])
    pieces = np.array(test_data["pieces"])
    n_test = len(true_log)
    rng_np = np.random.default_rng(42)
    idx = rng_np.choice(n_test, size=min(500, n_test), replace=False)
    idx.sort()

    residuals = [
        {"tr": round(float(true_log[i]), 3),
         "pr": round(float(pred_mean[i]), 3),
         "pc": int(pieces[i])}
        for i in idx
    ]

    # Round loss curve
    loss_curve_rounded = [round(v, 1) for v in loss_curve]

    # Round metrics
    test_metrics = {k: round(v, 4) for k, v in test_metrics.items()}
    train_metrics = {k: round(v, 4) for k, v in train_metrics.items()}

    # Add to explorer_data.json
    json_path = Path(__file__).resolve().parent.parent / "explorer_data.json"
    data = json.loads(json_path.read_text())

    data["models"]["model_4"] = {
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "loss_curve": loss_curve_rounded,
        "residuals": residuals,
    }

    output = json.dumps(data, separators=(",", ":"))
    json_path.write_text(output)
    print(f"\nWrote model_4 to {json_path} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
