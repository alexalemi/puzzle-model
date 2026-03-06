"""Basis functions for piece-count scaling dimensions.

DEPRECATED: The production model (model_2c) now uses physical basis functions
defined directly in model.py. This module is retained for backward compatibility
with older scripts but is no longer used by the main model pipeline.
"""

import jax.numpy as jnp
import numpy as np


BASIS_NAMES = ["log_N", "sqrt_N", "N", "N_log_N", "N_sq"]


def compute_basis(pieces: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """Compute raw basis functions phi_k(N) for given piece counts.

    Returns array of shape (len(pieces), 5) with columns:
        [log(N), sqrt(N), N, N*log(N), N^2]
    """
    N = jnp.asarray(pieces, dtype=jnp.float32)
    return jnp.column_stack([
        jnp.log(N),
        jnp.sqrt(N),
        N,
        N * jnp.log(N),
        N ** 2,
    ])


def normalize_basis(
    phi: jnp.ndarray, mean: jnp.ndarray | None = None, std: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Zero-mean, unit-variance normalization of basis functions.

    If mean/std are provided, uses those (for applying train normalization to test).
    Returns (normalized_phi, mean, std).
    """
    if mean is None:
        mean = jnp.mean(phi, axis=0)
    if std is None:
        std = jnp.std(phi, axis=0)
        std = jnp.where(std < 1e-8, 1.0, std)  # avoid division by zero
    return (phi - mean) / std, mean, std
