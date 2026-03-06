#!/usr/bin/env python3
"""Diagnostic: fit model_2c on all sources, check for systematic residuals by source."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import Predictive, log_likelihood

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


def main():
    # Load ALL sources
    df = load_solo_completed(source=None)
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    train_data = prepare_model_data(train_df)
    mu_fixed = train_data["mu_fixed"]
    test_data = prepare_model_data(test_df, mu_fixed=mu_fixed)

    print(f"mu_fixed = {mu_fixed:.3f} mB")
    print(f"Train: {len(train_df):,} obs, Test: {len(test_df):,} obs")
    print(f"Puzzlers: {train_data['n_puzzlers']:,}, Puzzles: {train_data['n_puzzles']:,}")
    print(f"\nTrain source breakdown:")
    print(train_df["source"].value_counts().to_string())
    print(f"\nTest source breakdown:")
    print(test_df["source"].value_counts().to_string())

    # Basis normalization
    phi_train = compute_basis(train_data["pieces"])
    _, basis_mean, basis_std = normalize_basis(phi_train)

    # Fit model_2c
    model_fn = MODELS["model_2c"]
    tr = dict(train_data)
    tr["basis_mean"] = basis_mean
    tr["basis_std"] = basis_std

    print(f"\nFitting model_2c on all sources...")
    guide, svi_result = run_svi(model_fn, tr, num_steps=5000, lr=0.005, seed=0)

    # Posterior samples
    posterior_samples = guide.sample_posterior(
        jax.random.PRNGKey(1), svi_result.params, sample_shape=(500,)
    )

    # Test predictions
    te = dict(test_data)
    te["basis_mean"] = basis_mean
    te["basis_std"] = basis_std
    te_no_obs = {k: v for k, v in te.items() if k != "log_time"}
    predictive = Predictive(model_fn, guide=guide, params=svi_result.params, num_samples=200)
    pred_test = predictive(jax.random.PRNGKey(1), **te_no_obs)

    pred_mean = np.mean(np.array(pred_test["log_time"]), axis=0)
    true_log = np.array(test_data["log_time"])
    residuals = pred_mean - true_log  # positive = model overpredicts (thinks slower)

    # Train predictions
    tr_no_obs = {k: v for k, v in tr.items() if k != "log_time"}
    pred_train = predictive(jax.random.PRNGKey(2), **tr_no_obs)
    train_pred_mean = np.mean(np.array(pred_train["log_time"]), axis=0)
    train_residuals = train_pred_mean - np.array(train_data["log_time"])

    # Test log-likelihood by source
    ll_test = log_likelihood(model_fn, posterior_samples, **te)
    ll_test_matrix = np.array(ll_test["log_time"])  # (500, n_test)
    n_samples = ll_test_matrix.shape[0]
    lppd_test_i = np.logaddexp.reduce(ll_test_matrix, axis=0) - np.log(n_samples)

    print(f"\n{'='*70}")
    print(f"RESIDUAL ANALYSIS BY SOURCE (test set)")
    print(f"{'='*70}")
    print(f"{'Source':<20} {'N':>7} {'Mean Resid':>12} {'Std Resid':>12} {'RMSE':>10} {'Mean LPD':>10}")
    print("-" * 71)

    for source in sorted(test_df["source"].unique()):
        mask = test_df["source"].values == source
        r = residuals[mask]
        lpd = lppd_test_i[mask]
        rmse = np.sqrt(np.mean(r**2))
        print(f"{source:<20} {mask.sum():>7} {np.mean(r):>12.2f} {np.std(r):>12.2f} {rmse:>10.2f} {np.mean(lpd):>10.4f}")

    # Overall
    rmse_all = np.sqrt(np.mean(residuals**2))
    print(f"{'ALL':<20} {len(residuals):>7} {np.mean(residuals):>12.2f} {np.std(residuals):>12.2f} {rmse_all:>10.2f} {np.mean(lppd_test_i):>10.4f}")

    print(f"\n{'='*70}")
    print(f"RESIDUAL ANALYSIS BY SOURCE (train set)")
    print(f"{'='*70}")
    print(f"{'Source':<20} {'N':>7} {'Mean Resid':>12} {'Std Resid':>12} {'RMSE':>10}")
    print("-" * 61)
    for source in sorted(train_df["source"].unique()):
        mask = train_df["source"].values == source
        r = train_residuals[mask]
        rmse = np.sqrt(np.mean(r**2))
        print(f"{source:<20} {mask.sum():>7} {np.mean(r):>12.2f} {np.std(r):>12.2f} {rmse:>10.2f}")

    # Shared puzzle analysis
    print(f"\n{'='*70}")
    print("SHARED PUZZLE ANALYSIS (puzzles appearing in multiple sources)")
    print(f"{'='*70}")
    puzzle_sources = df.groupby("puzzle_id")["source"].nunique()
    shared_puzzles = puzzle_sources[puzzle_sources > 1].index
    print(f"Number of shared puzzles: {len(shared_puzzles)}")

    if len(shared_puzzles) > 0:
        beta_samples = np.array(posterior_samples["beta"])
        beta_mean = np.mean(beta_samples, axis=0)
        beta_std = np.std(beta_samples, axis=0)
        inv_puzzle = {v: k for k, v in puzzle_lookup.items()}

        # For shared puzzles, compare actual mean log_time by source vs model prediction
        print(f"\n{'Puzzle':<40} {'Source':<15} {'N':>5} {'Obs Mean':>10} {'Beta':>10} {'Beta Std':>10}")
        print("-" * 95)
        shown = 0
        for pid in sorted(shared_puzzles)[:30]:
            pidx = puzzle_lookup.get(pid)
            if pidx is None:
                continue
            for src in sorted(df[df["puzzle_id"] == pid]["source"].unique()):
                sub = df[(df["puzzle_id"] == pid) & (df["source"] == src)]
                print(f"{pid[:38]:<40} {src:<15} {len(sub):>5} {sub['log_time'].mean():>10.1f} {beta_mean[pidx]:>10.1f} {beta_std[pidx]:>10.1f}")
            shown += 1
            if shown >= 15:
                break

    # Key model params
    print(f"\n{'='*70}")
    print("KEY MODEL PARAMETERS")
    print(f"{'='*70}")
    for name in ["sigma", "nu", "sigma_alpha", "sigma_beta", "delta_0", "sigma_delta"]:
        vals = np.array(posterior_samples[name])
        print(f"  {name:<20} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    c_basis = np.array(posterior_samples["c_basis"])
    for k in range(c_basis.shape[1]):
        print(f"  c_basis[{k}]          mean={np.mean(c_basis[:,k]):.4f}  std={np.std(c_basis[:,k]):.4f}")


if __name__ == "__main__":
    main()
