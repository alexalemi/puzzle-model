#!/usr/bin/env python3
"""Compare model_2r vs model_3d (discrimination + heteroscedastic noise).

Fits both models on the same train/test split and reports:
  - Test mean log predictive density
  - Fitted values of disc and eta_noise
  - WAIC comparison
"""

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
    add_repeat_features,
    to_fractional_year,
)
from puzzle_model.model import MODELS
from puzzle_model.inference import run_svi

numpyro.set_host_device_count(1)


def compute_calibration(guide, model, svi_params, test_data, n_samples=500, seed=42):
    """Compute calibration: does the X% interval contain X% of test obs?

    Returns dict with overall coverage at several levels, plus
    coverage broken down by alpha quartile.
    """
    # Posterior predictive samples on test data
    predictive = Predictive(model, guide=guide, params=svi_params, num_samples=n_samples)
    te_no_obs = {k: v for k, v in test_data.items() if k != "log_time"}
    pp_samples = np.array(predictive(jax.random.PRNGKey(seed + 10), **te_no_obs)["log_time"])
    # pp_samples: (n_samples, n_test)

    y_true = np.array(test_data["log_time"])
    n_test = len(y_true)

    # Compute PIT (probability integral transform): fraction of samples below observed
    pit = np.mean(pp_samples < y_true[None, :], axis=0)  # (n_test,)

    # Coverage at nominal levels
    levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    coverage = {}
    for lvl in levels:
        lo = np.percentile(pp_samples, 100 * (1 - lvl) / 2, axis=0)
        hi = np.percentile(pp_samples, 100 * (1 + lvl) / 2, axis=0)
        cov = np.mean((y_true >= lo) & (y_true <= hi))
        coverage[lvl] = float(cov)

    # Coverage by alpha quartile
    # Get alpha for each test puzzler from posterior samples
    post = guide.sample_posterior(jax.random.PRNGKey(seed + 20), svi_params, sample_shape=(1,))
    alpha_all = np.array(post["alpha"]).ravel()  # (n_puzzlers,)
    puzzler_idx = np.array(test_data["puzzler_idx"])
    alpha_test = alpha_all[puzzler_idx]  # (n_test,)

    quartile_edges = np.percentile(alpha_test, [0, 25, 50, 75, 100])
    quartile_labels = ["Q1 (best)", "Q2", "Q3", "Q4 (worst)"]
    by_quartile = {}
    for qi in range(4):
        lo_a, hi_a = quartile_edges[qi], quartile_edges[qi + 1]
        mask = (alpha_test >= lo_a) & (alpha_test <= hi_a) if qi == 3 else \
               (alpha_test >= lo_a) & (alpha_test < hi_a)
        if mask.sum() == 0:
            continue
        q_coverages = {}
        for lvl in levels:
            lo = np.percentile(pp_samples[:, mask], 100 * (1 - lvl) / 2, axis=0)
            hi = np.percentile(pp_samples[:, mask], 100 * (1 + lvl) / 2, axis=0)
            q_coverages[lvl] = float(np.mean((y_true[mask] >= lo) & (y_true[mask] <= hi)))
        by_quartile[quartile_labels[qi]] = {
            "n": int(mask.sum()),
            "alpha_range": f"[{lo_a:.0f}, {hi_a:.0f}]",
            "coverage": q_coverages,
        }

    return {"overall": coverage, "by_quartile": by_quartile, "pit_mean": float(np.mean(pit))}


def compute_test_lpd(guide, model, svi_params, test_data, n_samples=500, seed=42):
    """Compute mean log predictive density on test set."""
    # Get posterior samples from the guide (handles reparam correctly)
    posterior_samples = guide.sample_posterior(
        jax.random.PRNGKey(seed), svi_params, sample_shape=(n_samples,)
    )

    # Compute log-likelihood on test data
    ll = log_likelihood(model, posterior_samples, **test_data)
    ll_matrix = np.array(ll["log_time"])  # (n_samples, n_test)

    # Mean log predictive density: log(1/S * sum_s p(y|theta_s))
    lpd_per_obs = np.logaddexp.reduce(ll_matrix, axis=0) - np.log(n_samples)
    return float(np.mean(lpd_per_obs))


def main():
    print("Loading data...")
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df = add_repeat_features(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    # Prepare data dicts (all data including repeats, like model_2r in refit_all)
    train_data = prepare_model_data(train_df)
    test_data = prepare_model_data(test_df, mu_fixed=train_data["mu_fixed"])

    # Add repeat features
    for d, sub_df in [(train_data, train_df), (test_data, test_df)]:
        d["solve_number"] = sub_df["solve_number"].values
        if "year" not in d:
            d["year"] = to_fractional_year(sub_df)

    print(f"Train: {len(train_df):,} obs, Test: {len(test_df):,} obs")
    print(f"Puzzlers: {train_data['n_puzzlers']}, Puzzles: {train_data['n_puzzles']}")

    models_to_compare = ["model_2r", "model_3disc", "model_3het", "model_3d"]
    results = {}

    for model_name in models_to_compare:
        print(f"\n{'='*60}")
        print(f"Fitting {model_name}...")
        print(f"{'='*60}")

        model_fn = MODELS[model_name]
        guide, svi_result = run_svi(model_fn, train_data, num_steps=5000, lr=0.005)

        # Get fitted parameters via posterior sample
        posterior_samples = guide.sample_posterior(
            jax.random.PRNGKey(99), svi_result.params, sample_shape=(1,)
        )
        print(f"\nFitted scalar parameters:")
        scalar_params = {}
        for key in sorted(posterior_samples.keys()):
            val = posterior_samples[key]
            if val.ndim <= 1 and val.size == 1:
                v = float(val.ravel()[0])
                print(f"  {key}: {v:.6f}")
                scalar_params[key] = v
            elif val.ndim == 2 and val.shape[1] <= 4:
                vs = [float(x) for x in val[0]]
                print(f"  {key}: {[f'{v:.4f}' for v in vs]}")
                scalar_params[key] = vs

        # Compute test log predictive density
        print(f"\nComputing test LPD and calibration...")
        lpd = compute_test_lpd(guide, model_fn, svi_result.params, test_data)
        cal = compute_calibration(guide, model_fn, svi_result.params, test_data)
        print(f"  Test mean LPD: {lpd:.4f}")
        print(f"  Calibration (nominal → actual):")
        for lvl, cov in cal["overall"].items():
            delta = cov - lvl
            print(f"    {lvl:.0%} → {cov:.1%}  ({delta:+.1%})")

        results[model_name] = {"lpd": lpd, "params": scalar_params, "cal": cal}

    # Summary comparison
    baseline = results["model_2r"]["lpd"]
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'Test LPD':>10} {'Δ vs 2r':>10}  Notes")
    print(f"  {'-'*14:<16} {'-'*10:>10} {'-'*10:>10}  -----")
    for name, r in results.items():
        diff = r["lpd"] - baseline
        notes = []
        p = r["params"]
        if "disc" in p:
            notes.append(f"disc={p['disc']:.5f}")
        if "eta_noise" in p:
            notes.append(f"η={p['eta_noise']:.3f} (σ×{np.exp(p['eta_noise']):.2f} at +1σα)")
        print(f"  {name:<16} {r['lpd']:>10.4f} {diff:>+10.4f}  {', '.join(notes)}")

    # Calibration comparison
    levels = [0.5, 0.8, 0.9, 0.95]
    print(f"\n  Overall calibration (nominal → actual coverage):")
    header = f"  {'Model':<16}" + "".join(f"  {'%d%%' % (l*100):>6}" for l in levels)
    print(header)
    print(f"  {'(nominal)':<16}" + "".join(f"  {l:>5.0%} " for l in levels))
    print(f"  {'-'*14:<16}" + "  ------" * len(levels))
    for name, r in results.items():
        row = f"  {name:<16}"
        for l in levels:
            cov = r["cal"]["overall"][l]
            delta = cov - l
            row += f"  {cov:>5.1%} "
        print(row)

    # Calibration by alpha quartile
    print(f"\n  90% coverage by puzzler skill quartile:")
    print(f"  {'Model':<16}  {'Q1 (best)':>10}  {'Q2':>10}  {'Q3':>10}  {'Q4 (worst)':>10}")
    print(f"  {'-'*14:<16}" + "  ----------" * 4)
    for name, r in results.items():
        row = f"  {name:<16}"
        for ql in ["Q1 (best)", "Q2", "Q3", "Q4 (worst)"]:
            q = r["cal"]["by_quartile"].get(ql, {})
            cov = q.get("coverage", {}).get(0.9, float('nan'))
            row += f"  {cov:>9.1%} "
        print(row)


if __name__ == "__main__":
    main()
