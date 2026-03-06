"""NumPyro model definitions for speed puzzling latent factor model.

Model hierarchy (build incrementally):
  Model 0: mu + alpha_i + beta_j                         (basic IRT)
  Model 1: + c * log(N_j)                                (global piece-count)
  Model 2: + a_i1 * b_j1 * phi_1(N)                      (K=1 interaction)
  Model 3: + sum_k(a_ik * b_jk * phi_k(N))               (K=3 interactions)
  Model 4: Student-t likelihood                           (robust)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from .basis import compute_basis, normalize_basis


def model_0(puzzler_idx, puzzle_idx, n_puzzlers, n_puzzles, log_time=None, **kwargs):
    """Basic IRT: mu + alpha_i + beta_j + noise."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx]
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


def model_1(puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles, log_time=None, **kwargs):
    """Model 0 + global log(N) piece-count effect."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


def model_1b(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Model 0 + global coefficients for all 5 basis functions."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))

    with numpyro.plate("basis_fns", 5):
        c_basis = numpyro.sample("c_basis", dist.Normal(0, 1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)

    basis_effect = jnp.dot(phi, c_basis)
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + basis_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


def model_2(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Model 1 + single scaling interaction (K=1, using log(N))."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 1.0))
    sigma_b_factor = numpyro.sample("sigma_b_factor", dist.HalfNormal(1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        a_factor = numpyro.sample("a_factor", dist.Normal(0, 1.0))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))
        b_factor = numpyro.sample("b_factor", dist.Normal(0, sigma_b_factor))

    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)
    phi_log_N = phi[:, 0]  # K=1: use log(N) only

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    interaction = a_factor[puzzler_idx] * b_factor[puzzle_idx] * phi_log_N
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N + interaction

    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


def _sample_factors(n_puzzlers, n_puzzles, K):
    """Sample latent factor vectors for puzzlers and puzzles."""
    a_list, b_list = [], []
    for k in range(K):
        with numpyro.plate(f"puzzlers_k{k}", n_puzzlers):
            a_list.append(numpyro.sample(f"a_factor_{k}", dist.Normal(0, 1.0)))
        sigma_b_k = numpyro.sample(f"sigma_b_{k}", dist.HalfNormal(1.0))
        with numpyro.plate(f"puzzles_k{k}", n_puzzles):
            b_list.append(numpyro.sample(f"b_factor_{k}", dist.Normal(0, sigma_b_k)))
    return a_list, b_list


def model_3(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    K=3, basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Full model with K scaling interaction dimensions."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    a_list, b_list = _sample_factors(n_puzzlers, n_puzzles, K)

    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    interaction = jnp.zeros(len(puzzler_idx))
    for k in range(min(K, phi.shape[1])):
        interaction = interaction + (
            a_list[k][puzzler_idx] * b_list[k][puzzle_idx] * phi[:, k]
        )

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N + interaction
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


def model_4(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    K=3, basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Model 3 with Student-t likelihood for robustness."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))  # degrees of freedom
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 1.0))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    a_list, b_list = _sample_factors(n_puzzlers, n_puzzles, K)

    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    interaction = jnp.zeros(len(puzzler_idx))
    for k in range(min(K, phi.shape[1])):
        interaction = interaction + (
            a_list[k][puzzler_idx] * b_list[k][puzzle_idx] * phi[:, k]
        )

    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N + interaction
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma), obs=log_time)


def model_5(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    source_idx=None, n_sources=2,
    K=3, basis_mean=None, basis_std=None, log_time=None, **kwargs,
):
    """Model 3 + source fixed effect to capture platform differences."""
    mu = numpyro.sample("mu", dist.Normal(8.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
    c_pieces = numpyro.sample("c_pieces", dist.Normal(0, 1.0))

    with numpyro.plate("sources", n_sources):
        gamma = numpyro.sample("gamma", dist.Normal(0, 0.5))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    a_list, b_list = _sample_factors(n_puzzlers, n_puzzles, K)

    phi = compute_basis(pieces)
    if basis_mean is not None:
        phi, _, _ = normalize_basis(phi, basis_mean, basis_std)
    else:
        phi, _, _ = normalize_basis(phi)

    log_N = jnp.log(jnp.asarray(pieces, dtype=jnp.float32))
    interaction = jnp.zeros(len(puzzler_idx))
    for k in range(min(K, phi.shape[1])):
        interaction = interaction + (
            a_list[k][puzzler_idx] * b_list[k][puzzle_idx] * phi[:, k]
        )

    source_effect = gamma[source_idx] if source_idx is not None else 0.0
    mean = mu + alpha[puzzler_idx] + beta[puzzle_idx] + c_pieces * log_N + interaction + source_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.Normal(mean, sigma), obs=log_time)


# Non-centered versions for MCMC efficiency
model_0_nc = reparam(model_0, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_1_nc = reparam(model_1, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_1b_nc = reparam(model_1b, config={"alpha": LocScaleReparam(0), "beta": LocScaleReparam(0)})
model_2_nc = reparam(model_2, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "b_factor": LocScaleReparam(0),
})
_factor_reparam = {f"b_factor_{k}": LocScaleReparam(0) for k in range(5)}
model_3_nc = reparam(model_3, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    **_factor_reparam,
})
model_4_nc = reparam(model_4, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    **_factor_reparam,
})
model_5_nc = reparam(model_5, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    **_factor_reparam,
})

MODELS = {
    "model_0": model_0_nc,
    "model_1": model_1_nc,
    "model_1b": model_1b_nc,
    "model_2": model_2_nc,
    "model_3": model_3_nc,
    "model_4": model_4_nc,
    "model_5": model_5_nc,
}
