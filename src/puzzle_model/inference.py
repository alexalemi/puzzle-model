"""MCMC and SVI inference for puzzle models."""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_median, init_to_value
from numpyro.infer.autoguide import AutoNormal


def run_mcmc(
    model,
    data: dict,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    **nuts_kwargs,
) -> MCMC:
    """Run NUTS MCMC sampling."""
    kernel = NUTS(model, init_strategy=init_to_median, **nuts_kwargs)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(seed), **data)
    return mcmc


def run_svi(
    model,
    data: dict,
    num_steps: int = 10_000,
    lr: float = 0.005,
    seed: int = 0,
    init_values: dict | None = None,
) -> tuple[AutoNormal, dict]:
    """Run stochastic variational inference with AutoNormal guide.

    If init_values is provided, those sites are initialized to the given values
    (others fall back to init_to_median). Useful for mixture models where bad
    initialization leads to component collapse.
    """
    if init_values:
        init_fn = init_to_value(values=init_values)
    else:
        init_fn = init_to_median
    guide = AutoNormal(model, init_loc_fn=init_fn)
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, **data)
    return guide, result


def check_diagnostics(mcmc: MCMC, max_rhat: float = 1.05) -> dict:
    """Check MCMC diagnostics: R-hat, ESS, divergences."""
    summary = mcmc.print_summary(exclude_deterministic=False)

    extra_fields = mcmc.get_extra_fields()
    n_divergent = 0
    if "diverging" in extra_fields:
        n_divergent = int(extra_fields["diverging"].sum())

    # Compute per-parameter diagnostics
    samples = mcmc.get_samples()
    diagnostics = {
        "n_divergent": n_divergent,
        "n_params": sum(v.shape[-1] if v.ndim > 1 else 1 for v in samples.values()),
    }
    return diagnostics


def to_arviz(mcmc: MCMC, model=None, data: dict | None = None):
    """Convert MCMC results to ArviZ InferenceData.

    Pass model and data to include log-likelihood (needed for LOO/WAIC comparison).
    """
    import arviz as az
    from numpyro.infer import log_likelihood as compute_ll

    kwargs = {}
    if model is not None and data is not None:
        ll = compute_ll(model, mcmc.get_samples(), **data)
        kwargs["log_likelihood"] = ll
    return az.from_numpyro(mcmc, **kwargs)


def posterior_predictive(mcmc: MCMC, model, data: dict, seed: int = 1):
    """Generate posterior predictive samples."""
    predictive = Predictive(model, mcmc.get_samples())
    return predictive(jax.random.PRNGKey(seed), **{k: v for k, v in data.items() if k != "log_time"})


def save_mcmc(mcmc: MCMC, path: str):
    """Save MCMC samples to disk."""
    import pickle
    samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}
    extra = mcmc.get_extra_fields()
    with open(path, "wb") as f:
        pickle.dump({"samples": samples, "extra_fields": extra}, f)


def load_mcmc_samples(path: str) -> dict:
    """Load saved MCMC samples from disk."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
