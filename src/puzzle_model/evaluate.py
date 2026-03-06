"""Model evaluation and comparison metrics."""

import jax.numpy as jnp
import numpy as np


def rmse_log(pred_log_time: np.ndarray, true_log_time: np.ndarray) -> float:
    """RMSE on log scale."""
    return float(np.sqrt(np.mean((pred_log_time - true_log_time) ** 2)))


def rmse_original(pred_log_time: np.ndarray, true_log_time: np.ndarray) -> float:
    """RMSE on original (seconds) scale."""
    pred = np.exp(pred_log_time)
    true = np.exp(true_log_time)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae_log(pred_log_time: np.ndarray, true_log_time: np.ndarray) -> float:
    """MAE on log scale."""
    return float(np.mean(np.abs(pred_log_time - true_log_time)))


def prediction_interval_coverage(
    pred_samples: np.ndarray, true_log_time: np.ndarray, level: float = 0.9
) -> float:
    """Fraction of true values falling within prediction intervals.

    pred_samples: (n_samples, n_obs) array of posterior predictive samples
    true_log_time: (n_obs,) array of true log times
    """
    alpha = (1 - level) / 2
    lower = np.percentile(pred_samples, 100 * alpha, axis=0)
    upper = np.percentile(pred_samples, 100 * (1 - alpha), axis=0)
    covered = (true_log_time >= lower) & (true_log_time <= upper)
    return float(np.mean(covered))


def evaluate_predictions(
    pred_samples: np.ndarray, true_log_time: np.ndarray
) -> dict:
    """Compute all evaluation metrics."""
    pred_mean = np.mean(pred_samples, axis=0)
    return {
        "rmse_log": rmse_log(pred_mean, true_log_time),
        "rmse_seconds": rmse_original(pred_mean, true_log_time),
        "mae_log": mae_log(pred_mean, true_log_time),
        "coverage_90": prediction_interval_coverage(pred_samples, true_log_time, 0.9),
        "coverage_50": prediction_interval_coverage(pred_samples, true_log_time, 0.5),
    }


def model_comparison(mcmc_results: dict, model):
    """Compute WAIC and LOO for model comparison using ArviZ."""
    import arviz as az
    idata = az.from_numpyro(mcmc_results)
    waic = az.waic(idata)
    loo = az.loo(idata)
    return {"waic": waic, "loo": loo}


def naive_baselines(train_log_time: np.ndarray, test_log_time: np.ndarray) -> dict:
    """Compute naive baseline metrics for comparison."""
    global_mean = np.mean(train_log_time)
    return {
        "global_mean_rmse_log": rmse_log(
            np.full_like(test_log_time, global_mean), test_log_time
        ),
        "global_mean_rmse_seconds": rmse_original(
            np.full_like(test_log_time, global_mean), test_log_time
        ),
    }
