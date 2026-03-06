"""NumPyro model definitions for speed puzzling latent factor model.

All parameters are on the milliBel (mB) scale where 1000 mB = 1 Bel = factor of 10 in time.
Response variable: 1000 * log10(time_seconds).

mu is fixed to the empirical mean of log_time (passed as mu_fixed kwarg) to eliminate
the additive identifiability between mu, alpha, and beta. This makes alpha=0 genuinely
mean "average puzzler" so hierarchical shrinkage works correctly.

Model hierarchy (all use Student-t likelihood):
  Model 1t: mu_fixed + alpha_i + beta_j + c*log(N)       (robust baseline)
  Model 2:  + sum_k c_k * phi_k(N)                       (+ global basis)
  Model 2c: Model 2 + per-puzzler velocity                (+ velocity)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from .basis import compute_basis, normalize_basis


def model_1t(puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
             mu_fixed=None, log_time=None, **kwargs):
    """Student-t + log(N) piece-count effect with fixed mu."""
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 500.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


def _compute_phi(pieces, basis_mean, basis_std):
    """Compute and normalize basis functions."""
    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)
    return phi


def model_2(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Student-t + global coefficients for all 5 basis functions with fixed mu."""
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    with numpyro.plate("basis_fns", 5):
        c_basis = numpyro.sample("c_basis", dist.Normal(0, 500.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    phi = _compute_phi(pieces, basis_mean, basis_std)
    basis_effect = jnp.dot(phi, c_basis)
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + basis_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


YEAR_CENTER = 2025.0


def model_2c(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Model 2 + per-puzzler velocity (linear time trend) with fixed mu.

    Each puzzler gets: alpha_i + delta_i * (year - 2025)
    Plus a population-level trend delta_0.
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    with numpyro.plate("basis_fns", 5):
        c_basis = numpyro.sample("c_basis", dist.Normal(0, 500.0))

    # Velocity: population trend + per-puzzler deviation (mB/yr)
    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    phi = _compute_phi(pieces, basis_mean, basis_std)
    basis_effect = jnp.dot(phi, c_basis)

    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    velocity_effect = (delta_0 + delta[puzzler_idx]) * t

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + basis_effect + velocity_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


# Non-centered versions for SVI/MCMC efficiency
model_1t_nc = reparam(model_1t, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_2_nc = reparam(model_2, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_2c_nc = reparam(model_2c, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

MODELS = {
    "model_1t": model_1t_nc,
    "model_2": model_2_nc,
    "model_2c": model_2c_nc,
}
