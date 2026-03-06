#!/usr/bin/env python3
"""Fit model_2c variants and compare: original 5-basis, log-only, and physical."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import Predictive, log_likelihood
from numpyro.infer.reparam import LocScaleReparam

from puzzle_model.data import (
    load_solo_completed, create_puzzle_id, encode_indices,
    train_test_split, prepare_model_data,
)
from puzzle_model.model import MODELS, YEAR_CENTER
from puzzle_model.inference import run_svi
from puzzle_model.evaluate import evaluate_predictions
from puzzle_model.basis import compute_basis, normalize_basis

# Reference piece count for centering the physical basis correction
N_REF = 500.0

# Physical basis functions: [sqrt(N), N, N*log(N), N^2]
PHYS_BASIS_NAMES = ["sqrt_N", "N", "N_log_N", "N_sq"]


def model_2c_log(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None, **kwargs,
):
    """Model 2c with only c*log(N) piece scaling (no basis functions)."""
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 500.0))

    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    velocity_effect = (delta_0 + delta[puzzler_idx]) * t

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N + velocity_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


def model_2c_phys(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None, **kwargs,
):
    """Model 2c with physically-motivated piece-count scaling.

    Physical processes contribute additively in TIME, not log-time:
        time_pieces(N) = w0*sqrt(N) + w1*N + w2*N*log(N) + w3*N^2

    Converted to mB and centered at N_REF so the correction is zero
    at the reference piece count:
        piece_correction = 1000*log10(time_pieces(N)) - 1000*log10(time_pieces(N_REF))

    This gives sensible extrapolation: at large N, the dominant term
    wins and log(time) grows as at most 2*log(N), not exponentially.
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    # Log-weights for physical processes (exp gives positive weights)
    log_w = numpyro.sample("log_w", dist.Normal(0, 5.0).expand([4]))

    # Velocity
    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    # Physical basis functions in time domain
    N = jnp.asarray(pieces, dtype=jnp.float32)
    g = jnp.column_stack([jnp.sqrt(N), N, N * jnp.log(N), N ** 2])

    # Reference basis at N_REF (for centering)
    g_ref = jnp.array([jnp.sqrt(N_REF), N_REF, N_REF * jnp.log(N_REF), N_REF ** 2])

    # Weighted sum in time domain (positive weights via exp)
    w = jnp.exp(log_w)
    time_contrib = jnp.dot(g, w)        # (n_obs,)
    time_ref = jnp.dot(g_ref, w)        # scalar

    # Convert to mB, centered at N_REF
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Velocity
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    velocity_effect = (delta_0 + delta[puzzler_idx]) * t

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + piece_correction + velocity_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


model_2c_log_nc = reparam(model_2c_log, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

model_2c_phys_nc = reparam(model_2c_phys, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})


def compute_waic(pointwise_log_lik):
    n_samples = pointwise_log_lik.shape[0]
    lppd_i = np.logaddexp.reduce(pointwise_log_lik, axis=0) - np.log(n_samples)
    p_waic_i = np.var(pointwise_log_lik, axis=0, ddof=1)
    lppd = float(np.sum(lppd_i))
    p_waic = float(np.sum(p_waic_i))
    return {"lppd": lppd, "p_waic": p_waic, "waic": -2 * (lppd - p_waic)}


def fit_and_evaluate(name, model_fn, train_data, test_data, needs_basis=False):
    tr = dict(train_data)
    te = dict(test_data)
    if needs_basis:
        phi_train = compute_basis(tr["pieces"])
        _, basis_mean, basis_std = normalize_basis(phi_train)
        tr["basis_mean"] = basis_mean
        tr["basis_std"] = basis_std
        te["basis_mean"] = basis_mean
        te["basis_std"] = basis_std

    guide, svi_result = run_svi(model_fn, tr, num_steps=5000, lr=0.005, seed=0)

    posterior_samples = guide.sample_posterior(
        jax.random.PRNGKey(1), svi_result.params, sample_shape=(500,)
    )

    # Test predictions
    te_no_obs = {k: v for k, v in te.items() if k != "log_time"}
    predictive = Predictive(model_fn, guide=guide, params=svi_result.params, num_samples=200)
    pred_test = predictive(jax.random.PRNGKey(1), **te_no_obs)
    test_metrics = evaluate_predictions(
        np.array(pred_test["log_time"]), np.array(test_data["log_time"]),
    )

    # Train predictions
    tr_no_obs = {k: v for k, v in tr.items() if k != "log_time"}
    pred_train = predictive(jax.random.PRNGKey(2), **tr_no_obs)
    train_metrics = evaluate_predictions(
        np.array(pred_train["log_time"]), np.array(train_data["log_time"]),
    )

    # Test log-likelihood
    ll_test = log_likelihood(model_fn, posterior_samples, **te)
    ll_test_matrix = np.array(ll_test["log_time"])
    n_samples = ll_test_matrix.shape[0]
    lppd_test_i = np.logaddexp.reduce(ll_test_matrix, axis=0) - np.log(n_samples)
    mean_lpd = float(np.mean(lppd_test_i))

    # Train WAIC
    ll_train = log_likelihood(model_fn, posterior_samples, **tr)
    waic = compute_waic(np.array(ll_train["log_time"]))

    print(f"\n{name}:")
    print(f"  Train RMSE: {train_metrics['rmse_log']:.2f}  MAE: {train_metrics['mae_log']:.2f}")
    print(f"  Test  RMSE: {test_metrics['rmse_log']:.2f}  MAE: {test_metrics['mae_log']:.2f}  Cov90: {test_metrics['coverage_90']:.3f}")
    print(f"  Test mean LPD: {mean_lpd:.4f}")
    print(f"  WAIC: {waic['waic']:.1f}")

    # Print key params
    for pname in ["sigma", "nu", "c_pieces", "delta_0", "sigma_delta"]:
        if pname in posterior_samples:
            vals = np.array(posterior_samples[pname])
            print(f"  {pname}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    if "c_basis" in posterior_samples:
        cb = np.mean(np.array(posterior_samples["c_basis"]), axis=0)
        print(f"  c_basis: {[f'{v:.2f}' for v in cb]}")
    if "log_w" in posterior_samples:
        lw = np.array(posterior_samples["log_w"])
        lw_mean = np.mean(lw, axis=0)
        lw_std = np.std(lw, axis=0)
        print(f"  log_w (physical basis weights):")
        for i, bname in enumerate(PHYS_BASIS_NAMES):
            print(f"    {bname:>8}: log_w={lw_mean[i]:.3f}±{lw_std[i]:.3f}  "
                  f"(w={np.exp(lw_mean[i]):.4g})")

    return {"test_metrics": test_metrics, "train_metrics": train_metrics,
            "mean_lpd": mean_lpd, "waic": waic, "posterior": posterior_samples}


def main():
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    train_data = prepare_model_data(train_df)
    mu_fixed = train_data["mu_fixed"]
    test_data = prepare_model_data(test_df, mu_fixed=mu_fixed)

    print(f"mu_fixed = {mu_fixed:.3f} mB")
    print(f"Train: {len(train_df):,} obs, Test: {len(test_df):,} obs")
    print(f"Puzzlers: {train_data['n_puzzlers']:,}, Puzzles: {train_data['n_puzzles']:,}")

    # Fit all models for comparison
    results = {}
    results["model_1t"] = fit_and_evaluate(
        "model_1t", MODELS["model_1t"], train_data, test_data)
    results["model_2c"] = fit_and_evaluate(
        "model_2c", MODELS["model_2c"], train_data, test_data, needs_basis=True)
    results["model_2c_log"] = fit_and_evaluate(
        "model_2c_log", model_2c_log_nc, train_data, test_data)
    results["model_2c_phys"] = fit_and_evaluate(
        "model_2c_phys", model_2c_phys_nc, train_data, test_data)

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Model':<16} {'Test RMSE':>10} {'Test MAE':>10} {'Cov90':>8} {'Mean LPD':>10} {'WAIC':>12}")
    print("-" * 75)
    for name in ["model_1t", "model_2c_log", "model_2c_phys", "model_2c"]:
        r = results[name]
        print(f"{name:<16} {r['test_metrics']['rmse_log']:>10.2f} {r['test_metrics']['mae_log']:>10.2f} "
              f"{r['test_metrics']['coverage_90']:>8.3f} {r['mean_lpd']:>10.4f} {r['waic']['waic']:>12.1f}")

    # Show extrapolation behavior for model_2c_phys
    if "model_2c_phys" in results:
        post = results["model_2c_phys"]["posterior"]
        lw_samples = np.array(post["log_w"])  # (500, 4)
        print(f"\n{'='*75}")
        print("Extrapolation: model_2c_phys (avg puzzler, avg puzzle)")
        print(f"{'Pieces':>10}  {'Correction mB':>14}  {'Total mB':>10}  {'Time':>15}")
        for N in [100, 500, 1000, 2000, 5000, 13500, 50000, 100000, 1000000, 5000000]:
            g = np.array([np.sqrt(N), N, N * np.log(N), N ** 2])
            g_ref = np.array([np.sqrt(N_REF), N_REF, N_REF * np.log(N_REF), N_REF ** 2])
            # Per-sample correction
            w_samples = np.exp(lw_samples)
            time_N = w_samples @ g
            time_ref = w_samples @ g_ref
            corr_samples = 1000 / np.log(10) * (np.log(time_N) - np.log(time_ref))
            corr_mean = np.mean(corr_samples)
            total = mu_fixed + corr_mean
            t_sec = 10 ** (total / 1000)
            if t_sec < 3600:
                ts = f"{t_sec/60:.1f} min"
            elif t_sec < 86400:
                ts = f"{t_sec/3600:.1f} hr"
            elif t_sec < 86400 * 365.25:
                ts = f"{t_sec/86400:.1f} days"
            else:
                ts = f"{t_sec/86400/365.25:.1f} yr"
            print(f"  {N:>8}  {corr_mean:>14.1f}  {total:>10.1f}  {ts:>15}")


if __name__ == "__main__":
    main()
