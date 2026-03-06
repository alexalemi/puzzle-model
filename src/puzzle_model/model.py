"""NumPyro model definitions for speed puzzling latent factor model.

All parameters are on the milliBel (mB) scale where 1000 mB = 1 Bel = factor of 10 in time.
Response variable: 1000 * log10(time_seconds).

mu is fixed to the empirical mean of log_time (passed as mu_fixed kwarg) to eliminate
the additive identifiability between mu, alpha, and beta. This makes alpha=0 genuinely
mean "average puzzler" so hierarchical shrinkage works correctly.

Model hierarchy (all use Student-t likelihood):
  Model 1t: mu_fixed + alpha_i + beta_j + c*log(N)       (robust baseline)
  Model 2c: + physical piece-count correction + velocity  (production model)

Piece-count correction uses physically-motivated basis functions that sum additively
in time space: time(N) = w0*sqrt(N) + w1*N + w2*N*log(N) + w3*N^2, then converts
to mB centered at N_REF. This gives sensible extrapolation (log-time grows as at
most 2*log(N), not exponentially).
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

# Reference piece count for centering the physical basis correction
N_REF = 500.0

# Physical basis function names: [sqrt(N), N, N*log(N), N^2]
PHYS_BASIS_NAMES = ["sqrt_N", "N", "N_log_N", "N_sq"]


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


YEAR_CENTER = 2025.0


def model_2c(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None, **kwargs,
):
    """Physical basis model + per-puzzler velocity with fixed mu.

    Physical processes contribute additively in TIME, not log-time:
        time_pieces(N) = w0*sqrt(N) + w1*N + w2*N*log(N) + w3*N^2

    Converted to mB and centered at N_REF so the correction is zero
    at the reference piece count:
        piece_correction = 1000*log10(time_pieces(N)) - 1000*log10(time_pieces(N_REF))

    Each puzzler gets: alpha_i + delta_i * (year - 2025)
    Plus a population-level trend delta_0.
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    # Log-weights for physical processes (exp gives positive weights)
    log_w = numpyro.sample("log_w", dist.Normal(0, 5.0).expand([4]))

    # Velocity: population trend + per-puzzler deviation (mB/yr)
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
    time_contrib = jnp.dot(g, w)
    time_ref = jnp.dot(g_ref, w)

    # Convert to mB, centered at N_REF
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Velocity
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    velocity_effect = (delta_0 + delta[puzzler_idx]) * t

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + piece_correction + velocity_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


def model_2r(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None,
    solve_number=None, **kwargs,
):
    """Physical basis + velocity + log-practice effect (no forgetting).

    Extends model_2c with:
        repeat_effect = gamma * log(solve_number)

    solve_number is computed from observed data sequences, with a floor of 2
    for rows where MSP reports first_attempt=False but we lack the prior solve.
    For first attempts: solve_number=1 → log(1)=0, so the effect vanishes.
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    sigma = numpyro.sample("sigma", dist.HalfNormal(500.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    # Physical basis weights
    log_w = numpyro.sample("log_w", dist.Normal(0, 5.0).expand([4]))

    # Velocity
    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    # Practice effect (global)
    gamma = numpyro.sample("gamma", dist.Normal(0, 200.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    # Physical basis correction (same as 2c)
    N = jnp.asarray(pieces, dtype=jnp.float32)
    g = jnp.column_stack([jnp.sqrt(N), N, N * jnp.log(N), N ** 2])
    g_ref = jnp.array([jnp.sqrt(N_REF), N_REF, N_REF * jnp.log(N_REF), N_REF ** 2])
    w = jnp.exp(log_w)
    time_contrib = jnp.dot(g, w)
    time_ref = jnp.dot(g_ref, w)
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Velocity
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    velocity_effect = (delta_0 + delta[puzzler_idx]) * t

    # Practice: log(solve_number), zero for first attempts
    sn = jnp.asarray(solve_number, dtype=jnp.float32) if solve_number is not None else jnp.ones(len(puzzler_idx))
    repeat_effect = gamma * jnp.log(sn)

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + piece_correction + velocity_effect + repeat_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


# Non-centered versions for SVI/MCMC efficiency
model_1t_nc = reparam(model_1t, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_2c_nc = reparam(model_2c, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})
model_2r_nc = reparam(model_2r, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

MODELS = {
    "model_1t": model_1t_nc,
    "model_2c": model_2c_nc,
    "model_2r": model_2r_nc,
}
