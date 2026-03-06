#!/usr/bin/env python3
"""Refit all models on current data and regenerate explorer_data.json."""

import json
import random
from pathlib import Path

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
from puzzle_model.evaluate import evaluate_predictions, naive_baselines
from puzzle_model.basis import compute_basis, normalize_basis


def compute_waic(pointwise_log_lik: np.ndarray) -> dict:
    n_samples = pointwise_log_lik.shape[0]
    lppd_i = np.logaddexp.reduce(pointwise_log_lik, axis=0) - np.log(n_samples)
    lppd = np.sum(lppd_i)
    p_waic_i = np.var(pointwise_log_lik, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)
    waic = -2 * (lppd - p_waic)
    se = 2 * np.sqrt(len(lppd_i) * np.var(-2 * (lppd_i - p_waic_i)))
    return {"waic": float(waic), "lppd": float(lppd), "p_waic": float(p_waic), "se": float(se)}


def subsample(lst, n, seed=42):
    rng = random.Random(seed)
    return rng.sample(lst, min(n, len(lst)))


def main():
    # ── Load and prepare data ──
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    train_data = prepare_model_data(train_df)
    test_data = prepare_model_data(test_df)

    # Basis normalization
    phi_train = compute_basis(train_data["pieces"])
    _, basis_mean, basis_std = normalize_basis(phi_train)

    n_puzzlers = int(train_data["n_puzzlers"])
    n_puzzles = int(train_data["n_puzzles"])
    n_train = len(train_df)
    n_test = len(test_df)

    # Source lookup for reporting
    if "source" in df.columns:
        source_names = sorted(df["source"].unique())
        n_sources = len(source_names)
    else:
        source_names = []
        n_sources = 0

    print(f"Train: {n_train:,} obs, Test: {n_test:,} obs")
    print(f"Puzzlers: {n_puzzlers:,}, Puzzles: {n_puzzles:,}")
    if source_names:
        print(f"Sources: {source_names}")
        for s in source_names:
            n_s = int((train_df["source"] == s).sum())
            print(f"  {s}: {n_s:,} train obs")

    # ── Baselines ──
    baselines = naive_baselines(
        np.array(train_data["log_time"]),
        np.array(test_data["log_time"]),
    )

    # ── Fit all models ──
    model_names = ["model_0", "model_1", "model_1b", "model_2", "model_3", "model_4", "model_5"]
    needs_basis = {"model_1b", "model_2", "model_3", "model_4", "model_5"}
    model_results = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"Fitting {name}...")
        model_fn = MODELS[name]

        # Prepare data dicts with basis if needed
        tr = dict(train_data)
        te = dict(test_data)
        if name in needs_basis:
            tr["basis_mean"] = basis_mean
            tr["basis_std"] = basis_std
            te["basis_mean"] = basis_mean
            te["basis_std"] = basis_std

        # Fit via SVI
        guide, svi_result = run_svi(model_fn, tr, num_steps=5000, lr=0.005, seed=0)
        loss_curve = [float(v) for v in svi_result.losses[::10]]

        # Posterior samples for predictions and WAIC
        posterior_samples = guide.sample_posterior(
            jax.random.PRNGKey(1), svi_result.params, sample_shape=(500,)
        )

        # Test predictions
        te_no_obs = {k: v for k, v in te.items() if k != "log_time"}
        predictive = Predictive(model_fn, guide=guide, params=svi_result.params, num_samples=200)
        pred_test = predictive(jax.random.PRNGKey(1), **te_no_obs)
        test_metrics = evaluate_predictions(
            np.array(pred_test["log_time"]),
            np.array(test_data["log_time"]),
        )

        # Train predictions
        tr_no_obs = {k: v for k, v in tr.items() if k != "log_time"}
        pred_train = predictive(jax.random.PRNGKey(2), **tr_no_obs)
        train_metrics = evaluate_predictions(
            np.array(pred_train["log_time"]),
            np.array(train_data["log_time"]),
        )

        # WAIC
        ll = log_likelihood(model_fn, posterior_samples, **tr)
        ll_matrix = np.array(ll["log_time"])
        waic_result = compute_waic(ll_matrix)

        # Residuals (subsample)
        pred_mean = np.mean(np.array(pred_test["log_time"]), axis=0)
        true_log = np.array(test_data["log_time"])
        pieces_arr = np.array(test_data["pieces"])
        rng_np = np.random.default_rng(42)
        idx = rng_np.choice(len(true_log), size=min(500, len(true_log)), replace=False)
        idx.sort()
        residuals = [
            {"tr": round(float(true_log[i]), 3),
             "pr": round(float(pred_mean[i]), 3),
             "pc": int(pieces_arr[i])}
            for i in idx
        ]

        # Source effects (for model_5)
        source_effects = None
        if name == "model_5" and "gamma" in posterior_samples:
            gamma = np.array(posterior_samples["gamma"])  # (500, n_sources)
            source_effects = {
                source_names[j]: {
                    "mean": round(float(np.mean(gamma[:, j])), 4),
                    "std": round(float(np.std(gamma[:, j])), 4),
                }
                for j in range(gamma.shape[1])
            }
            print("Source effects (gamma):")
            for s, v in source_effects.items():
                print(f"  {s}: {v['mean']:+.4f} ± {v['std']:.4f}")

        print(f"  Test RMSE(log): {test_metrics['rmse_log']:.4f}  WAIC: {waic_result['waic']:.1f}")

        model_results[name] = {
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
            "loss_curve": [round(v, 1) for v in loss_curve],
            "residuals": residuals,
            "waic": round(waic_result["waic"], 1),
            "lppd": round(waic_result["lppd"], 1),
            "p_waic": round(waic_result["p_waic"], 1),
            "waic_se": round(waic_result["se"], 1),
        }
        if source_effects:
            model_results[name]["source_effects"] = source_effects

    # ── WAIC comparison ──
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'WAIC':>10} {'p_waic':>8} {'dWAIC':>8}")
    print("-" * 48)
    best_waic = min(r["waic"] for r in model_results.values())
    for name in sorted(model_results, key=lambda k: model_results[k]["waic"]):
        r = model_results[name]
        delta = r["waic"] - best_waic
        print(f"{name:<20} {r['waic']:>10.1f} {r['p_waic']:>8.1f} {delta:>+8.1f}")

    # ── Build explorer data ──
    # Stats
    log_times = np.array(df["log_time"])
    piece_counts = sorted(df["puzzle_pieces"].unique())

    # Get Model 1 params from its posterior
    guide_1, svi_1 = run_svi(MODELS["model_1"], train_data, num_steps=5000, lr=0.005, seed=0)
    params_1 = guide_1.sample_posterior(jax.random.PRNGKey(0), svi_1.params, sample_shape=(200,))
    mu_val = round(float(np.mean(np.array(params_1["mu"]))), 3)
    c_val = round(float(np.mean(np.array(params_1["c_pieces"]))), 3)
    sigma_val = round(float(np.mean(np.array(params_1["sigma"]))), 3)

    stats = {
        "n_records": len(df),
        "n_puzzlers": n_puzzlers,
        "n_puzzles": n_puzzles,
        "piece_counts": [int(p) for p in piece_counts],
        "log_time_mean": round(float(np.mean(log_times)), 3),
        "log_time_std": round(float(np.std(log_times)), 3),
        "mu": mu_val,
        "c_pieces": c_val,
        "sigma": sigma_val,
    }
    if source_names:
        stats["sources"] = source_names

    # Scatter (subsample)
    scatter_idx = random.Random(42).sample(range(len(df)), min(1500, len(df)))
    scatter = [
        {"p": int(df.iloc[i]["puzzle_pieces"]),
         "t": round(float(df.iloc[i]["time_seconds"]), 1),
         "lt": round(float(df.iloc[i]["log_time"]), 2)}
        for i in sorted(scatter_idx)
    ]

    # Piece distribution
    pc_counts = df["puzzle_pieces"].value_counts().sort_index()
    piece_dist = [{"pieces": int(p), "count": int(c)} for p, c in pc_counts.items()]

    # Histograms for common piece counts
    common_pcs = [p for p, c in pc_counts.items() if c >= 100][:9]
    histograms = {}
    for pc in common_pcs:
        vals = df[df["puzzle_pieces"] == pc]["log_time"].values
        counts, edges = np.histogram(vals, bins=30)
        histograms[str(int(pc))] = {
            "n": len(vals),
            "counts": counts.tolist(),
            "edges": [round(float(e), 3) for e in edges],
        }

    # Puzzler frequency
    obs_per_puzzler = df.groupby("puzzler_idx").size().values
    freq_counts, freq_edges = np.histogram(obs_per_puzzler, bins=50)
    puzzler_freq = {
        "counts": freq_counts.tolist(),
        "edges": [round(float(e), 1) for e in freq_edges],
    }

    # Puzzler rankings (from Model 1 posterior)
    alpha_mean = np.mean(np.array(params_1["alpha"]), axis=0)
    alpha_std = np.std(np.array(params_1["alpha"]), axis=0)
    inv_puzzler = {v: k for k, v in puzzler_lookup.items()}
    obs_counts = df.groupby("puzzler_idx").size()
    puzzlers_list = []
    for i in np.argsort(alpha_mean):
        puzzlers_list.append({
            "name": inv_puzzler[i],
            "alpha": round(float(alpha_mean[i]), 3),
            "std": round(float(alpha_std[i]), 3),
            "n": int(obs_counts.get(i, 0)),
        })

    # Puzzle rankings (from Model 1 posterior)
    beta_mean = np.mean(np.array(params_1["beta"]), axis=0)
    beta_std = np.std(np.array(params_1["beta"]), axis=0)
    inv_puzzle = {v: k for k, v in puzzle_lookup.items()}
    puzzle_obs = df.groupby("puzzle_idx").size()
    # Get piece count per puzzle
    puzzle_pieces = df.groupby("puzzle_idx")["puzzle_pieces"].first()
    puzzles_list = []
    for i in np.argsort(-beta_mean):
        puzzles_list.append({
            "name": inv_puzzle[i],
            "beta": round(float(beta_mean[i]), 3),
            "std": round(float(beta_std[i]), 3),
            "pc": int(puzzle_pieces.get(i, 0)),
            "n": int(puzzle_obs.get(i, 0)),
        })

    # ── Assemble and write ──
    explorer_data = {
        "stats": stats,
        "baselines": {k: round(v, 4) for k, v in baselines.items()},
        "models": model_results,
        "scatter": scatter,
        "piece_dist": piece_dist,
        "histograms": histograms,
        "puzzler_freq": puzzler_freq,
        "puzzlers": puzzlers_list,
        "puzzles": puzzles_list,
    }

    out_path = Path(__file__).resolve().parent.parent / "explorer_data.json"
    output = json.dumps(explorer_data, separators=(",", ":"))
    out_path.write_text(output)
    print(f"\nWrote {out_path} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
