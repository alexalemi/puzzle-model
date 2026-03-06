#!/usr/bin/env python3
"""Extract detailed Model 4 parameters for the explorer."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from puzzle_model.data import (
    load_solo_completed, create_puzzle_id, encode_indices,
    train_test_split, prepare_model_data,
)
from puzzle_model.model import MODELS
from puzzle_model.inference import run_svi
from puzzle_model.basis import compute_basis, normalize_basis


def main():
    # Load data
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)
    train_data = prepare_model_data(train_df)

    phi_train = compute_basis(train_data["pieces"])
    _, basis_mean, basis_std = normalize_basis(phi_train)
    train_data["basis_mean"] = basis_mean
    train_data["basis_std"] = basis_std

    inv_puzzler = {v: k for k, v in puzzler_lookup.items()}
    inv_puzzle = {v: k for k, v in puzzle_lookup.items()}
    puzzle_pieces = train_df.groupby("puzzle_idx")["puzzle_pieces"].first()
    puzzler_obs = train_df.groupby("puzzler_idx").size()
    puzzle_obs = train_df.groupby("puzzle_idx").size()

    # Fit Model 4
    print("Fitting Model 4...")
    model_fn = MODELS["model_4"]
    guide, svi_result = run_svi(model_fn, train_data, num_steps=5000, lr=0.005, seed=0)

    # Sample posterior
    samples = guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params, sample_shape=(500,))

    def summarize(key):
        arr = np.array(samples[key])
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    # 1. Scalar parameters
    scalar_params = {}
    for key in ["mu", "sigma", "nu", "c_pieces", "sigma_alpha", "sigma_beta"]:
        scalar_params[key] = summarize(key)
        print(f"  {key}: {scalar_params[key]['mean']:.4f} ± {scalar_params[key]['std']:.4f}")

    for k in range(3):
        key = f"sigma_b_{k}"
        scalar_params[key] = summarize(key)
        print(f"  {key}: {scalar_params[key]['mean']:.4f} ± {scalar_params[key]['std']:.4f}")

    # 2. Student-t vs Normal comparison data
    nu_mean = scalar_params["nu"]["mean"]
    sigma_mean = scalar_params["sigma"]["mean"]
    x_range = np.linspace(-3 * sigma_mean, 3 * sigma_mean, 200)
    from scipy.stats import t as t_dist, norm
    student_pdf = t_dist.pdf(x_range, df=nu_mean, scale=sigma_mean)
    normal_pdf = norm.pdf(x_range, scale=sigma_mean)
    tail_ratio = student_pdf / np.maximum(normal_pdf, 1e-30)

    student_t_comparison = {
        "x": [round(float(v), 4) for v in x_range],
        "student_pdf": [round(float(v), 6) for v in student_pdf],
        "normal_pdf": [round(float(v), 6) for v in normal_pdf],
        "nu": round(nu_mean, 2),
        "sigma": round(sigma_mean, 4),
    }

    # 3. Latent factors for puzzlers (subsample for viz)
    alpha = np.mean(np.array(samples["alpha"]), axis=0)
    n_puzzlers = len(alpha)

    a_factors = []
    for k in range(3):
        a_factors.append(np.mean(np.array(samples[f"a_factor_{k}"]), axis=0))

    # Select puzzlers with enough observations for stable estimates
    min_obs = 3
    good_puzzlers = [i for i in range(n_puzzlers) if puzzler_obs.get(i, 0) >= min_obs]
    # Subsample if too many
    rng = np.random.default_rng(42)
    if len(good_puzzlers) > 800:
        good_puzzlers = sorted(rng.choice(good_puzzlers, 800, replace=False))

    puzzler_factors = []
    for i in good_puzzlers:
        puzzler_factors.append({
            "name": inv_puzzler[i],
            "alpha": round(float(alpha[i]), 3),
            "a0": round(float(a_factors[0][i]), 3),
            "a1": round(float(a_factors[1][i]), 3),
            "a2": round(float(a_factors[2][i]), 3),
            "n": int(puzzler_obs.get(i, 0)),
        })

    # 4. Latent factors for puzzles
    beta = np.mean(np.array(samples["beta"]), axis=0)
    n_puzzles = len(beta)
    b_factors = []
    for k in range(3):
        b_factors.append(np.mean(np.array(samples[f"b_factor_{k}"]), axis=0))

    good_puzzles = [j for j in range(n_puzzles) if puzzle_obs.get(j, 0) >= 3]
    puzzle_factors = []
    for j in good_puzzles:
        puzzle_factors.append({
            "name": inv_puzzle[j],
            "beta": round(float(beta[j]), 3),
            "b0": round(float(b_factors[0][j]), 3),
            "b1": round(float(b_factors[1][j]), 3),
            "b2": round(float(b_factors[2][j]), 3),
            "pc": int(puzzle_pieces.get(j, 0)),
            "n": int(puzzle_obs.get(j, 0)),
        })

    # 5. Piece-count response curves for selected puzzlers
    # Show how predicted log-time varies with N for a few representative puzzlers
    # Pick: fastest, median, slowest, and a couple interesting ones
    alpha_order = np.argsort(alpha)
    # Filter to well-observed puzzlers
    well_observed = [i for i in alpha_order if puzzler_obs.get(i, 0) >= 10]
    selected_idx = [
        well_observed[0],           # fastest
        well_observed[len(well_observed)//4],  # 25th pctile
        well_observed[len(well_observed)//2],  # median
        well_observed[3*len(well_observed)//4],  # 75th pctile
        well_observed[-1],          # slowest
    ]

    mu_mean = scalar_params["mu"]["mean"]
    c_mean = scalar_params["c_pieces"]["mean"]
    piece_range = np.array([50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000])
    phi_range = compute_basis(piece_range)
    phi_norm, _, _ = normalize_basis(phi_range, basis_mean, basis_std)

    response_curves = []
    for i in selected_idx:
        log_N = np.log(piece_range.astype(np.float32))
        interaction = sum(
            float(a_factors[k][i]) * np.array(b_factors[k]).mean() * np.array(phi_norm[:, k])
            for k in range(3)
        )
        predicted = mu_mean + float(alpha[i]) + c_mean * log_N + interaction
        response_curves.append({
            "name": inv_puzzler[i],
            "alpha": round(float(alpha[i]), 3),
            "n": int(puzzler_obs.get(i, 0)),
            "pieces": piece_range.tolist(),
            "log_time": [round(float(v), 3) for v in predicted],
        })

    # 6. Factor correlation with piece count (for puzzles)
    # Do puzzle factors correlate with piece count?
    factor_by_pieces = []
    for j in good_puzzles:
        factor_by_pieces.append({
            "pc": int(puzzle_pieces.get(j, 0)),
            "b0": round(float(b_factors[0][j]), 3),
            "b1": round(float(b_factors[1][j]), 3),
            "b2": round(float(b_factors[2][j]), 3),
            "beta": round(float(beta[j]), 3),
        })

    # Write to JSON
    json_path = Path(__file__).resolve().parent.parent / "explorer_data.json"
    data = json.loads(json_path.read_text())

    data["model4_detail"] = {
        "scalar_params": scalar_params,
        "student_t_comparison": student_t_comparison,
        "puzzler_factors": puzzler_factors,
        "puzzle_factors": puzzle_factors,
        "response_curves": response_curves,
        "factor_by_pieces": factor_by_pieces,
    }

    output = json.dumps(data, separators=(",", ":"))
    json_path.write_text(output)
    print(f"\nWrote model4_detail to {json_path} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
