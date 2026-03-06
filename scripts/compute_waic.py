#!/usr/bin/env python3
"""Fit all models via SVI and compute WAIC for model comparison.

WAIC (Widely Applicable Information Criterion) estimates out-of-sample
predictive accuracy from training data:
  WAIC = -2 * (lppd - p_waic)
where:
  lppd  = sum_i log(mean_s p(y_i | theta_s))   -- fit quality
  p_waic = sum_i var_s(log p(y_i | theta_s))    -- complexity penalty
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import log_likelihood

from puzzle_model.data import (
    load_solo_completed,
    create_puzzle_id,
    encode_indices,
    train_test_split,
    prepare_model_data,
)
from puzzle_model.model import MODELS
from puzzle_model.inference import run_svi
from puzzle_model.basis import compute_basis, normalize_basis


def compute_waic(pointwise_log_lik: np.ndarray) -> dict:
    """Compute WAIC from pointwise log-likelihood matrix.

    Args:
        pointwise_log_lik: (n_samples, n_obs) array of log p(y_i | theta_s)

    Returns:
        dict with waic, lppd, p_waic, and per-observation values
    """
    # lppd_i = log(mean_s exp(log_lik_si)) = logsumexp - log(S)
    n_samples = pointwise_log_lik.shape[0]
    lppd_i = np.logaddexp.reduce(pointwise_log_lik, axis=0) - np.log(n_samples)
    lppd = np.sum(lppd_i)

    # p_waic_i = var_s(log_lik_si)  (variance across posterior samples)
    p_waic_i = np.var(pointwise_log_lik, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)

    waic = -2 * (lppd - p_waic)
    se = 2 * np.sqrt(len(lppd_i) * np.var(-2 * (lppd_i - p_waic_i)))

    return {
        "waic": float(waic),
        "lppd": float(lppd),
        "p_waic": float(p_waic),
        "se": float(se),
        "n_obs": int(pointwise_log_lik.shape[1]),
    }


def main():
    # Load and prepare data
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)
    train_data = prepare_model_data(train_df)

    # Basis normalization for models 2-4
    phi_train = compute_basis(train_data["pieces"])
    _, basis_mean, basis_std = normalize_basis(phi_train)
    train_data_with_basis = {**train_data, "basis_mean": basis_mean, "basis_std": basis_std}

    print(f"Train: {len(train_df):,} obs")
    print(f"{'Model':<20} {'WAIC':>10} {'lppd':>10} {'p_waic':>8} {'SE':>8}")
    print("-" * 60)

    results = {}
    for name in ["model_0", "model_1", "model_2", "model_3", "model_4"]:
        model_fn = MODELS[name]
        data = train_data_with_basis if name in ("model_2", "model_3", "model_4") else train_data

        # Fit via SVI
        guide, svi_result = run_svi(model_fn, data, num_steps=5000, lr=0.005, seed=0)

        # Draw posterior samples from the guide
        posterior_samples = guide.sample_posterior(
            jax.random.PRNGKey(1), svi_result.params, sample_shape=(500,)
        )

        # Compute pointwise log-likelihood
        ll = log_likelihood(model_fn, posterior_samples, **data)
        ll_matrix = np.array(ll["log_time"])  # (n_samples, n_obs)

        # Compute WAIC
        w = compute_waic(ll_matrix)
        results[name] = w
        print(f"{name:<20} {w['waic']:>10.1f} {w['lppd']:>10.1f} {w['p_waic']:>8.1f} {w['se']:>8.1f}")

    # Compute deltas from best
    best_waic = min(r["waic"] for r in results.values())
    print(f"\n{'Model':<20} {'dWAIC':>10}")
    print("-" * 32)
    for name in sorted(results, key=lambda k: results[k]["waic"]):
        delta = results[name]["waic"] - best_waic
        print(f"{name:<20} {delta:>10.1f}")

    # Add to explorer_data.json
    json_path = Path(__file__).resolve().parent.parent / "explorer_data.json"
    data = json.loads(json_path.read_text())

    for name, w in results.items():
        if name in data["models"]:
            data["models"][name]["waic"] = round(w["waic"], 1)
            data["models"][name]["lppd"] = round(w["lppd"], 1)
            data["models"][name]["p_waic"] = round(w["p_waic"], 1)
            data["models"][name]["waic_se"] = round(w["se"], 1)

    output = json.dumps(data, separators=(",", ":"))
    json_path.write_text(output)
    print(f"\nWAIC values added to {json_path}")


if __name__ == "__main__":
    main()
