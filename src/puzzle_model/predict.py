"""Prediction utilities for fitted puzzle models."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import Predictive


def predict_in_sample(mcmc, model, data: dict, seed: int = 1) -> dict:
    """Generate posterior predictive for in-sample data."""
    predictive = Predictive(model, mcmc.get_samples())
    obs_data = {k: v for k, v in data.items() if k != "log_time"}
    return predictive(jax.random.PRNGKey(seed), **obs_data)


def predict_new(mcmc, model, data: dict, seed: int = 1) -> dict:
    """Predict for held-out data (puzzlers/puzzles seen during training)."""
    predictive = Predictive(model, mcmc.get_samples())
    obs_data = {k: v for k, v in data.items() if k != "log_time"}
    return predictive(jax.random.PRNGKey(seed), **obs_data)


def predict_cold_start(mcmc, model, puzzle_idx, pieces, n_puzzlers, n_puzzles, seed: int = 1):
    """Predict for a new puzzler (shrink to population mean, alpha=0).

    Uses the mean puzzler effect (alpha=0) for cold-start prediction.
    Returns predictions averaged over puzzle difficulty posterior.
    """
    samples = mcmc.get_samples()
    mu = samples["mu"]
    beta = samples["beta"][:, puzzle_idx]
    sigma = samples["sigma"]

    if "c_pieces" in samples:
        log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
        mean = mu + beta + samples["c_pieces"] * log_N
    else:
        mean = mu + beta

    # New puzzler: alpha = 0 (shrunk to mean)
    pred_log_time = mean
    pred_time = jnp.exp(pred_log_time + sigma**2 / 2)  # log-normal mean correction

    return {
        "pred_log_time_mean": float(jnp.mean(pred_log_time)),
        "pred_log_time_std": float(jnp.std(pred_log_time)),
        "pred_time_mean": float(jnp.mean(pred_time)),
        "pred_time_median": float(jnp.median(jnp.exp(pred_log_time))),
    }


def puzzler_rankings(mcmc, puzzler_lookup: dict) -> list[tuple[str, float, float]]:
    """Rank puzzlers by posterior mean of alpha (lower = faster)."""
    samples = mcmc.get_samples()
    alpha = samples["alpha"]  # (n_samples, n_puzzlers)
    means = np.array(jnp.mean(alpha, axis=0))
    stds = np.array(jnp.std(alpha, axis=0))

    inv_lookup = {v: k for k, v in puzzler_lookup.items()}
    rankings = [(inv_lookup[i], means[i], stds[i]) for i in range(len(means))]
    rankings.sort(key=lambda x: x[1])
    return rankings


def puzzle_rankings(mcmc, puzzle_lookup: dict) -> list[tuple[str, float, float]]:
    """Rank puzzles by posterior mean of beta (higher = harder)."""
    samples = mcmc.get_samples()
    beta = samples["beta"]  # (n_samples, n_puzzles)
    means = np.array(jnp.mean(beta, axis=0))
    stds = np.array(jnp.std(beta, axis=0))

    inv_lookup = {v: k for k, v in puzzle_lookup.items()}
    rankings = [(inv_lookup[i], means[i], stds[i]) for i in range(len(means))]
    rankings.sort(key=lambda x: -x[1])
    return rankings
