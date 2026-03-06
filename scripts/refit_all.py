#!/usr/bin/env python3
"""Refit all models on current data and regenerate explorer_data.json."""

import json
import math
import os
import random
import re
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
from puzzle_model.model import MODELS, N_REF, PHYS_BASIS_NAMES
from puzzle_model.inference import run_svi
from puzzle_model.evaluate import evaluate_predictions, naive_baselines


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
    mu_fixed = train_data["mu_fixed"]
    test_data = prepare_model_data(test_df, mu_fixed=mu_fixed)
    print(f"mu_fixed = {mu_fixed:.3f} mB")

    n_puzzlers = int(train_data["n_puzzlers"])
    n_puzzles = int(train_data["n_puzzles"])
    n_train = len(train_df)
    n_test = len(test_df)

    print(f"Train: {n_train:,} obs, Test: {n_test:,} obs")
    print(f"Puzzlers: {n_puzzlers:,}, Puzzles: {n_puzzles:,}")

    # ── Baselines ──
    baselines = naive_baselines(
        np.array(train_data["log_time"]),
        np.array(test_data["log_time"]),
    )

    # ── Fit all models ──
    model_names = ["model_1t", "model_2c"]
    model_results = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"Fitting {name}...")
        model_fn = MODELS[name]

        tr = dict(train_data)
        te = dict(test_data)

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

        # WAIC (train)
        ll = log_likelihood(model_fn, posterior_samples, **tr)
        ll_matrix = np.array(ll["log_time"])
        waic_result = compute_waic(ll_matrix)

        # Held-out predictive log-likelihood
        ll_test = log_likelihood(model_fn, posterior_samples, **te)
        ll_test_matrix = np.array(ll_test["log_time"])  # (500, n_test)
        n_samples = ll_test_matrix.shape[0]
        lppd_test_i = np.logaddexp.reduce(ll_test_matrix, axis=0) - np.log(n_samples)
        lppd_test = float(np.sum(lppd_test_i))
        mean_lpd = float(np.mean(lppd_test_i))

        # Residuals (subsample)
        pred_mean = np.mean(np.array(pred_test["log_time"]), axis=0)
        true_log = np.array(test_data["log_time"])
        pieces_arr = np.array(test_data["pieces"])
        puzzler_names = test_df["competitor_name"].values
        puzzle_names = test_df["puzzle_id"].values
        rng_np = np.random.default_rng(42)
        idx = rng_np.choice(len(true_log), size=min(500, len(true_log)), replace=False)
        idx.sort()
        residuals = [
            {"tr": round(float(true_log[i]), 3),
             "pr": round(float(pred_mean[i]), 3),
             "pc": int(pieces_arr[i]),
             "puzzler": str(puzzler_names[i]),
             "puzzle": str(puzzle_names[i])}
            for i in idx
        ]

        print(f"  Test mean lpd: {mean_lpd:.4f}  WAIC: {waic_result['waic']:.1f}")

        model_results[name] = {
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
            "loss_curve": [round(v, 1) for v in loss_curve],
            "residuals": residuals,
            "waic": round(waic_result["waic"], 1),
            "lppd": round(waic_result["lppd"], 1),
            "p_waic": round(waic_result["p_waic"], 1),
            "waic_se": round(waic_result["se"], 1),
            "test_lppd": round(lppd_test, 1),
            "test_mean_lpd": round(mean_lpd, 4),
        }
        # Save model_2c posterior for rankings (velocity-aware)
        if name == "model_2c":
            ranking_samples = posterior_samples

    # ── Model comparison ──
    print(f"\n{'='*60}")
    print(f"{'Model':<15} {'WAIC':>12} {'Test LPPD':>12} {'Mean LPD':>10}")
    print("-" * 52)
    best_waic = min(r["waic"] for r in model_results.values())
    for name in sorted(model_results, key=lambda k: -model_results[k]["test_lppd"]):
        r = model_results[name]
        delta = r["waic"] - best_waic
        print(f"{name:<15} {r['waic']:>12.1f} {r['test_lppd']:>12.1f} {r['test_mean_lpd']:>10.4f}")

    # ── Build explorer data ──
    # Stats
    log_times = np.array(df["log_time"])
    piece_counts = sorted(df["puzzle_pieces"].unique())

    # Use Model 2c posterior (velocity-aware) for rankings
    from puzzle_model.model import YEAR_CENTER
    RANKING_YEAR = 2026
    params_1 = ranking_samples
    sigma_val = round(float(np.mean(np.array(params_1["sigma"]))), 3)
    delta_0_val = round(float(np.mean(np.array(params_1["delta_0"]))), 4)
    sigma_delta_val = round(float(np.mean(np.array(params_1["sigma_delta"]))), 4)

    # Physical basis: log_w posterior means
    log_w_samples = np.array(params_1["log_w"])  # (500, 4)
    log_w_mean = np.mean(log_w_samples, axis=0)
    log_w_vals = [round(float(v), 4) for v in log_w_mean]

    stats = {
        "n_records": len(df),
        "n_puzzlers": n_puzzlers,
        "n_puzzles": n_puzzles,
        "piece_counts": [int(p) for p in piece_counts],
        "log_time_mean": round(float(np.mean(log_times)), 3),
        "log_time_std": round(float(np.std(log_times)), 3),
        "mu": round(mu_fixed, 3),
        "log_w": log_w_vals,
        "N_REF": N_REF,
        "phys_basis_names": PHYS_BASIS_NAMES,
        "sigma": sigma_val,
        "delta_0": delta_0_val,
        "sigma_delta": sigma_delta_val,
        "year_center": YEAR_CENTER,
        "ranking_year": RANKING_YEAR,
        "sigma_alpha": round(float(np.mean(np.array(params_1["sigma_alpha"]))), 3),
        "sigma_beta": round(float(np.mean(np.array(params_1["sigma_beta"]))), 3),
        "nu": round(float(np.mean(np.array(params_1["nu"]))), 3),
    }

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

    # Puzzler rankings (from model_2c posterior, projected to RANKING_YEAR)
    ELO_SCALE = 1
    alpha_samples = np.array(params_1["alpha"])  # (500, n_puzzlers)
    delta_samples = np.array(params_1["delta"])  # (500, n_puzzlers)
    delta_0_samples = np.array(params_1["delta_0"])  # (500,)
    beta_samples_raw = np.array(params_1["beta"])  # (500, n_puzzles)
    mu_val = round(mu_fixed, 3)
    stats["mu"] = mu_val

    dt = RANKING_YEAR - YEAR_CENTER
    projected = alpha_samples + (delta_0_samples[:, None] + delta_samples) * dt
    proj_mean = np.mean(projected, axis=0)
    proj_std = np.std(projected, axis=0)
    proj_upper = proj_mean + 3 * proj_std
    ELO_CENTER = round(1500 + float(np.median(proj_mean)))
    stats["elo_center"] = ELO_CENTER

    alpha_mean = np.mean(alpha_samples, axis=0)
    velocity_samples = -(delta_0_samples[:, None] + delta_samples)
    velocity_mean = np.mean(velocity_samples, axis=0)
    velocity_std = np.std(velocity_samples, axis=0)

    inv_puzzler = {v: k for k, v in puzzler_lookup.items()}
    obs_counts = df.groupby("puzzler_idx").size()
    year_range = df.groupby("puzzler_idx")["year"].agg(["min", "max"])
    puzzlers_list = []
    for i in np.argsort(proj_upper):
        elo = ELO_CENTER - ELO_SCALE * float(proj_mean[i])
        elo_wilson = ELO_CENTER - ELO_SCALE * float(proj_upper[i])
        puzzlers_list.append({
            "name": inv_puzzler[i],
            "alpha": round(float(alpha_mean[i]), 3),
            "alpha_proj": round(float(proj_mean[i]), 3),
            "std": round(float(proj_std[i]), 3),
            "velocity": round(float(velocity_mean[i]), 4),
            "velocity_std": round(float(velocity_std[i]), 4),
            "elo": round(elo),
            "elo_wilson": round(elo_wilson),
            "n": int(obs_counts.get(i, 0)),
            "yr_min": int(year_range.loc[i, "min"]) if i in year_range.index else 0,
            "yr_max": int(year_range.loc[i, "max"]) if i in year_range.index else 0,
        })

    # Puzzle rankings
    beta_mean = np.mean(beta_samples_raw, axis=0)
    beta_std = np.std(beta_samples_raw, axis=0)
    beta_lower = beta_mean - 3 * beta_std
    inv_puzzle = {v: k for k, v in puzzle_lookup.items()}
    puzzle_obs = df.groupby("puzzle_idx").size()
    puzzle_pieces = df.groupby("puzzle_idx")["puzzle_pieces"].first()
    beta_upper = beta_mean + 3 * beta_std

    # Build image lookup
    img_dir = Path(__file__).resolve().parent.parent / "puzzle_images"
    img_by_prefix = {}
    if img_dir.is_dir():
        for f in img_dir.iterdir():
            if f.suffix in (".jpg", ".jpeg", ".png"):
                prefix = f.name.split("-")[0]
                img_by_prefix[prefix] = f"puzzle_images/{f.name}"
    img_lookup = {}
    for _, row in df.drop_duplicates("puzzle_id").iterrows():
        eid = str(row.get("event_id", ""))
        m = re.match(r"msp_([0-9a-f]+)", eid)
        if m and m.group(1) in img_by_prefix:
            img_lookup[row["puzzle_id"]] = img_by_prefix[m.group(1)]

    mu_float = mu_fixed
    puzzles_list = []
    for i in np.argsort(-beta_lower):
        p_elo = ELO_CENTER + ELO_SCALE * (mu_float + float(beta_mean[i]))
        p_elo_hard = ELO_CENTER + ELO_SCALE * (mu_float + float(beta_lower[i]))
        p_elo_easy = ELO_CENTER + ELO_SCALE * (mu_float + float(beta_upper[i]))
        entry = {
            "name": inv_puzzle[i],
            "beta": round(float(beta_mean[i]), 3),
            "std": round(float(beta_std[i]), 3),
            "wilson_hard": round(float(beta_lower[i]), 3),
            "wilson_easy": round(float(beta_upper[i]), 3),
            "elo": round(p_elo),
            "elo_hard": round(p_elo_hard),
            "elo_easy": round(p_elo_easy),
            "pc": int(puzzle_pieces.get(i, 0)),
            "n": int(puzzle_obs.get(i, 0)),
        }
        if inv_puzzle[i] in img_lookup:
            entry["img"] = img_lookup[inv_puzzle[i]]
        puzzles_list.append(entry)

    # ── Model 2c Deep Dive data ──
    from scipy import stats as sp_stats

    # Scalar params with mean/std
    scalar_names = ["mu", "sigma", "nu", "sigma_alpha", "sigma_beta", "delta_0", "sigma_delta"]
    scalar_params = {}
    for name in scalar_names:
        vals = np.array(params_1[name])
        if name == "mu":
            scalar_params[name] = {"mean": round(mu_fixed, 4), "std": 0.0}
            continue
        scalar_params[name] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    # Add individual log_w coefficients
    for k in range(4):
        scalar_params[f"log_w_{k}"] = {
            "mean": round(float(np.mean(log_w_samples[:, k])), 4),
            "std": round(float(np.std(log_w_samples[:, k])), 4),
        }
        # Also store exponentiated weight for display
        w_samples_k = np.exp(log_w_samples[:, k])
        scalar_params[f"w_{k}"] = {
            "mean": round(float(np.mean(w_samples_k)), 4),
            "std": round(float(np.std(w_samples_k)), 4),
        }

    # Student-t vs Normal comparison
    nu_mean = float(np.mean(np.array(params_1["nu"])))
    sigma_mean = float(np.mean(np.array(params_1["sigma"])))
    x_range = np.linspace(-4 * sigma_mean, 4 * sigma_mean, 200)
    student_pdf = sp_stats.t.pdf(x_range, df=nu_mean, scale=sigma_mean)
    normal_pdf = sp_stats.norm.pdf(x_range, scale=sigma_mean)
    student_t_comparison = {
        "nu": round(nu_mean, 2),
        "sigma": round(sigma_mean, 2),
        "x": [round(float(v), 2) for v in x_range],
        "student_pdf": [round(float(v), 6) for v in student_pdf],
        "normal_pdf": [round(float(v), 6) for v in normal_pdf],
    }

    # Basis correction curve using physical basis
    # Extended range since extrapolation is now sane
    pieces_range = np.array([50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 13500])
    mB_scale = 1000.0 / np.log(10.0)

    # Per-sample correction: (500, n_pieces)
    w_samples = np.exp(log_w_samples)  # (500, 4)
    g_curve = np.column_stack([
        np.sqrt(pieces_range), pieces_range,
        pieces_range * np.log(pieces_range), pieces_range ** 2,
    ])  # (n_pieces, 4)
    g_ref = np.array([np.sqrt(N_REF), N_REF, N_REF * np.log(N_REF), N_REF ** 2])

    time_curve = g_curve @ w_samples.T  # (n_pieces, 500)
    time_ref = g_ref @ w_samples.T      # (500,)
    correction_samples = mB_scale * (np.log(time_curve) - np.log(time_ref[None, :]))  # (n_pieces, 500)
    correction_samples = correction_samples.T  # (500, n_pieces)

    basis_mean_curve = np.mean(correction_samples, axis=0)
    deciles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    quantile_data = {}
    for q in deciles:
        quantile_data[f"p{q}"] = [round(float(v), 2) for v in np.percentile(correction_samples, q, axis=0)]
    basis_correction = {
        "pieces": pieces_range.tolist(),
        "mean": [round(float(v), 2) for v in basis_mean_curve],
        "quantiles": quantile_data,
    }

    # Per-component mB contribution curves (for the basis chart)
    basis_components = {}
    for k, bname in enumerate(PHYS_BASIS_NAMES):
        # Contribution of component k alone, averaged over posterior
        w_k_mean = float(np.mean(w_samples[:, k]))
        g_k = g_curve[:, k]  # (n_pieces,)
        g_k_ref = g_ref[k]
        # Component contribution in mB (relative to N_REF)
        component_mB = mB_scale * (np.log(w_k_mean * g_k) - np.log(w_k_mean * g_k_ref))
        # Simplifies to mB_scale * (log(g_k) - log(g_k_ref)) — weight cancels!
        # But we keep it explicit for clarity
        basis_components[bname] = [round(float(v), 2) for v in component_mB]
    basis_components["pieces"] = pieces_range.tolist()

    model2c_detail = {
        "scalar_params": scalar_params,
        "student_t_comparison": student_t_comparison,
        "basis_correction": basis_correction,
        "basis_components": basis_components,
    }

    # Puzzle difficulty distribution
    puzzle_mB = beta_mean + mu_fixed
    puzzle_beta_all = {"values": [round(float(v), 3) for v in puzzle_mB],
                       "n": [int(puzzle_obs.get(i, 0)) for i in range(len(beta_mean))]}
    beta_hist_counts, beta_hist_edges = np.histogram(puzzle_mB, bins=50)
    puzzle_beta_dist = {
        "counts": beta_hist_counts.tolist(),
        "edges": [round(float(e), 3) for e in beta_hist_edges],
    }
    for label, nmin, nmax in [("n1", 1, 1), ("n2_5", 2, 5), ("n6plus", 6, 999999)]:
        mask = np.array([(nmin <= int(puzzle_obs.get(i, 0)) <= nmax) for i in range(len(beta_mean))])
        if mask.any():
            c, e = np.histogram(puzzle_mB[mask], bins=beta_hist_edges)
            puzzle_beta_dist[label] = c.tolist()

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
        "model2c_detail": model2c_detail,
        "puzzle_beta_dist": puzzle_beta_dist,
    }

    out_path = Path(__file__).resolve().parent.parent / "explorer_data.json"

    def sanitize(obj):
        """Replace inf/nan floats with None for JSON compatibility."""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    explorer_data = sanitize(explorer_data)
    output = json.dumps(explorer_data, separators=(",", ":"))
    out_path.write_text(output)
    print(f"\nWrote {out_path} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
