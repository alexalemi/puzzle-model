"""NumPyro model definitions for speed puzzling latent factor model.

All parameters are on the milliBel (mB) scale where 1000 mB = 1 Bel = factor of 10 in time.
Response variable: 1000 * log10(time_seconds).

mu is fixed (passed as mu_fixed kwarg) to eliminate the additive identifiability between
mu, alpha, and beta. Default is MU_ONE_HOUR = 1000*log10(3600) ≈ 3556.3 mB, so alpha=0
means "solves like a 1-hour reference" and hierarchical shrinkage works correctly.

Model hierarchy:
  Model 1t: mu_fixed + alpha_i + beta_j + c*log(N)  (Student-t, robust baseline)
  Model 2c: + physical piece-count correction + velocity  (Student-t)
  Model team: + logsumexp team alpha + Amdahl's law  (Student-t)
  Model team_gmm: same as team but contaminated Normal likelihood
  Model team_nst: same as team but Normal + Student-t mixture likelihood

Piece-count correction uses physically-motivated basis functions that sum additively
in time space: time(N) = w0*sqrt(N) + w1*N + w2*N*log(N) + w3*N^2, then converts
to mB centered at N_REF. This gives sensible extrapolation (log-time grows as at
most 2*log(N), not exponentially).
"""

import jax
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


def model_team(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None,
    solve_number=None,
    team_member_idx=None,   # (n_obs, MAX_TEAM_SIZE) int
    team_mask=None,          # (n_obs, MAX_TEAM_SIZE) bool
    team_size=None,          # (n_obs,) int
    **kwargs,
):
    """Joint solo+team model: logsumexp team alpha + Amdahl's law + per-bucket corrections.

    For each team member i, compute time-adjusted alpha:
        alpha_i(t) = alpha_i + (delta_0 + delta_i) * t

    Logsumexp gives the pure-parallel prediction (rates add):
        alpha_parallel = -mB * log(sum_i exp(-alpha_i(t) / mB))

    Amdahl's law corrects for serial fraction s:
        amdahl = mB * log(1 + s*(K-1))     [>= 0, slows teams down]

    Per-bucket residual corrections (K=2, K=3, K>=4):
        eta[bucket(K)]

    For solo (K=1): all corrections vanish, reduces exactly to model_2r.
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

    # Team parameters
    logit_s = numpyro.sample("logit_s", dist.Normal(0, 2.0))  # serial fraction via sigmoid
    eta_team = numpyro.sample("eta_team", dist.Normal(0, 50.0).expand([3]))  # per-bucket corrections
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(500.0))  # separate scale for teams

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    # Physical basis correction
    N = jnp.asarray(pieces, dtype=jnp.float32)
    g = jnp.column_stack([jnp.sqrt(N), N, N * jnp.log(N), N ** 2])
    g_ref = jnp.array([jnp.sqrt(N_REF), N_REF, N_REF * jnp.log(N_REF), N_REF ** 2])
    w = jnp.exp(log_w)
    time_contrib = jnp.dot(g, w)
    time_ref = jnp.dot(g_ref, w)
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Practice
    sn = jnp.asarray(solve_number, dtype=jnp.float32) if solve_number is not None else jnp.ones(len(puzzler_idx))
    repeat_effect = gamma * jnp.log(sn)

    # --- Team alpha via logsumexp with per-member velocity ---
    team_member_idx = jnp.asarray(team_member_idx, dtype=jnp.int32)
    team_mask_arr = jnp.asarray(team_mask, dtype=jnp.float32)
    team_size_arr = jnp.asarray(team_size, dtype=jnp.float32)

    # Per-member velocity-adjusted alpha
    alpha_gathered = alpha[team_member_idx]          # (n_obs, MAX_TEAM_SIZE)
    delta_gathered = delta[team_member_idx]          # (n_obs, MAX_TEAM_SIZE)
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    alpha_adjusted = alpha_gathered + (delta_0 + delta_gathered) * t[:, None]

    # Logsumexp: rates add, alpha_parallel = -mB * log(sum exp(-alpha_adj / mB))
    rates = jnp.where(team_mask_arr, jnp.exp(-alpha_adjusted / mB_scale), 0.0)
    total_rate = jnp.sum(rates, axis=1)
    alpha_parallel = -mB_scale * jnp.log(total_rate)

    # Amdahl's law correction: s = serial fraction
    s = jax.nn.sigmoid(logit_s)
    numpyro.deterministic("s", s)
    amdahl_correction = mB_scale * jnp.log(1 + s * (team_size_arr - 1))

    # Per-bucket residual correction (K=2->0, K=3->1, K>=4->2)
    is_team = team_size_arr > 1
    team_bucket = jnp.clip(team_size_arr.astype(jnp.int32) - 2, 0, 2)
    eta_correction = jnp.where(is_team, eta_team[team_bucket], 0.0)

    alpha_eff = alpha_parallel + amdahl_correction + eta_correction

    # Use sigma_team for team obs, sigma for solo
    sigma_eff = jnp.where(team_size_arr > 1, sigma_team, sigma)

    mean = mu + alpha_eff + beta[puzzle_idx] + piece_correction + repeat_effect
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", dist.StudentT(nu, mean, sigma_eff), obs=log_time)


def model_team_gmm(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None,
    solve_number=None,
    team_member_idx=None,
    team_mask=None,
    team_size=None,
    **kwargs,
):
    """Joint solo+team model with contaminated Normal (Gaussian mixture) likelihood.

    Identical to model_team but replaces StudentT(nu, mean, sigma) with:
        w * Normal(mean, sigma) + (1-w) * Normal(mean, k*sigma)

    where w is the fraction of "clean" observations and k is the scale
    multiplier for the outlier component.  This captures the empirical
    finding that residuals are a sharp core + broad tails, which a
    single Student-t cannot match (Student-t couples peak shape to tail weight).
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    # LogNormal priors on sigma prevent mixture-induced collapse to near-zero.
    # HalfNormal(500) is flat near zero, which is fine for Student-t (where nu
    # controls tails) but causes a GMM narrow component to degenerate to a delta.
    # LogNormal centered at empirical residual width prevents this.
    sigma = numpyro.sample("sigma", dist.LogNormal(jnp.log(60.0), 0.5))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    # Contaminated Normal parameters (replace nu)
    # Constrain w_clean > 0.5 and k_sigma > 1 to break label-switching symmetry:
    # without these constraints, SVI can swap the narrow/wide components.
    w_raw = numpyro.sample("w_raw", dist.Beta(5, 1))  # concentrates near 1
    w_clean = numpyro.deterministic("w_clean", 0.5 + 0.5 * w_raw)  # ∈ [0.5, 1.0]
    k_raw = numpyro.sample("k_raw", dist.LogNormal(1.0, 0.5))  # median ≈ 2.7
    k_sigma = numpyro.deterministic("k_sigma", 1.0 + k_raw)  # > 1 always

    # Physical basis weights
    log_w = numpyro.sample("log_w", dist.Normal(0, 5.0).expand([4]))

    # Velocity
    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    # Practice effect (global)
    gamma = numpyro.sample("gamma", dist.Normal(0, 200.0))

    # Team parameters
    logit_s = numpyro.sample("logit_s", dist.Normal(0, 2.0))
    eta_team = numpyro.sample("eta_team", dist.Normal(0, 50.0).expand([3]))
    sigma_team = numpyro.sample("sigma_team", dist.LogNormal(jnp.log(80.0), 0.5))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    # Physical basis correction
    N = jnp.asarray(pieces, dtype=jnp.float32)
    g = jnp.column_stack([jnp.sqrt(N), N, N * jnp.log(N), N ** 2])
    g_ref = jnp.array([jnp.sqrt(N_REF), N_REF, N_REF * jnp.log(N_REF), N_REF ** 2])
    w = jnp.exp(log_w)
    time_contrib = jnp.dot(g, w)
    time_ref = jnp.dot(g_ref, w)
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Practice
    sn = jnp.asarray(solve_number, dtype=jnp.float32) if solve_number is not None else jnp.ones(len(puzzler_idx))
    repeat_effect = gamma * jnp.log(sn)

    # --- Team alpha (identical to model_team) ---
    team_member_idx = jnp.asarray(team_member_idx, dtype=jnp.int32)
    team_mask_arr = jnp.asarray(team_mask, dtype=jnp.float32)
    team_size_arr = jnp.asarray(team_size, dtype=jnp.float32)

    alpha_gathered = alpha[team_member_idx]
    delta_gathered = delta[team_member_idx]
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    alpha_adjusted = alpha_gathered + (delta_0 + delta_gathered) * t[:, None]

    rates = jnp.where(team_mask_arr, jnp.exp(-alpha_adjusted / mB_scale), 0.0)
    total_rate = jnp.sum(rates, axis=1)
    alpha_parallel = -mB_scale * jnp.log(total_rate)

    s = jax.nn.sigmoid(logit_s)
    numpyro.deterministic("s", s)
    amdahl_correction = mB_scale * jnp.log(1 + s * (team_size_arr - 1))

    is_team = team_size_arr > 1
    team_bucket = jnp.clip(team_size_arr.astype(jnp.int32) - 2, 0, 2)
    eta_correction = jnp.where(is_team, eta_team[team_bucket], 0.0)

    alpha_eff = alpha_parallel + amdahl_correction + eta_correction
    sigma_eff = jnp.where(team_size_arr > 1, sigma_team, sigma)

    mean = mu + alpha_eff + beta[puzzle_idx] + piece_correction + repeat_effect

    # --- Contaminated Normal likelihood ---
    # p(y) = w_clean * N(mean, sigma_eff) + (1 - w_clean) * N(mean, k_sigma * sigma_eff)
    mixing = dist.Categorical(probs=jnp.stack([w_clean, 1 - w_clean]))
    component_locs = jnp.stack([mean, mean], axis=-1)
    component_scales = jnp.stack([sigma_eff, k_sigma * sigma_eff], axis=-1)
    obs_dist = dist.MixtureSameFamily(mixing, dist.Normal(component_locs, component_scales))

    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", obs_dist, obs=log_time)


class NormalStudentTMixture(dist.Distribution):
    """Mixture of Normal and Student-t with shared location.

    p(y) = w * Normal(loc, scale) + (1-w) * StudentT(df, loc, scale_t)

    This decouples peak sharpness (Normal controls the core) from tail weight
    (Student-t controls extreme values), unlike pure Student-t where a single
    nu parameter governs both.
    """
    support = dist.constraints.real
    arg_constraints = {
        "w": dist.constraints.unit_interval,
        "loc": dist.constraints.real,
        "scale": dist.constraints.positive,
        "df": dist.constraints.positive,
        "scale_t": dist.constraints.positive,
    }

    def __init__(self, w, loc, scale, df, scale_t, *, validate_args=None):
        self.w, self.loc, self.scale, self.df, self.scale_t = (
            jnp.asarray(w), jnp.asarray(loc), jnp.asarray(scale),
            jnp.asarray(df), jnp.asarray(scale_t),
        )
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(scale_t),
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        ll_n = jnp.log(self.w) + dist.Normal(self.loc, self.scale).log_prob(value)
        ll_t = jnp.log(1 - self.w) + dist.StudentT(self.df, self.loc, self.scale_t).log_prob(value)
        return jnp.logaddexp(ll_n, ll_t)

    def sample(self, key, sample_shape=()):
        k1, k2, k3 = jax.random.split(key, 3)
        mask = jax.random.bernoulli(k1, self.w, shape=sample_shape + self.batch_shape)
        n_samples = dist.Normal(self.loc, self.scale).sample(k2, sample_shape)
        t_samples = dist.StudentT(self.df, self.loc, self.scale_t).sample(k3, sample_shape)
        return jnp.where(mask, n_samples, t_samples)


def model_team_nst(
    puzzler_idx, puzzle_idx, pieces, n_puzzlers, n_puzzles,
    mu_fixed=None, year=None, log_time=None,
    solve_number=None,
    team_member_idx=None,
    team_mask=None,
    team_size=None,
    **kwargs,
):
    """Joint solo+team model with Normal + wide Student-t mixture likelihood.

    p(y) = w * Normal(mean, sigma) + (1-w) * StudentT(nu, mean, k*sigma)

    The Normal captures the sharp residual core (~95%), while the Student-t
    at wider scale k*sigma handles both the broad shoulder and heavy tails.
    This gives three degrees of freedom for the noise shape:
      sigma — core width, w — mixture fraction, nu — tail decay, k — outlier scale.
    """
    mu = numpyro.deterministic("mu", jnp.float32(mu_fixed))
    # LogNormal prior prevents mixture-induced sigma collapse (same as GMM).
    sigma = numpyro.sample("sigma", dist.LogNormal(jnp.log(60.0), 0.5))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(300.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(300.0))

    # Mixture weight: fraction in Normal component, constrained > 0.5
    w_raw = numpyro.sample("w_raw", dist.Beta(5, 1))
    w_clean = numpyro.deterministic("w_clean", 0.5 + 0.5 * w_raw)
    # Student-t scale multiplier: > 1 so outlier component is wider
    k_raw = numpyro.sample("k_raw", dist.LogNormal(1.0, 0.5))
    k_sigma = numpyro.deterministic("k_sigma", 1.0 + k_raw)

    # Physical basis weights
    log_w = numpyro.sample("log_w", dist.Normal(0, 5.0).expand([4]))

    # Velocity
    delta_0 = numpyro.sample("delta_0", dist.Normal(0, 100.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(100.0))

    # Practice effect (global)
    gamma = numpyro.sample("gamma", dist.Normal(0, 200.0))

    # Team parameters
    logit_s = numpyro.sample("logit_s", dist.Normal(0, 2.0))
    eta_team = numpyro.sample("eta_team", dist.Normal(0, 50.0).expand([3]))
    sigma_team = numpyro.sample("sigma_team", dist.LogNormal(jnp.log(80.0), 0.5))

    with numpyro.plate("puzzlers", n_puzzlers):
        alpha = numpyro.sample("alpha", dist.Normal(0, sigma_alpha))
        delta = numpyro.sample("delta", dist.Normal(0, sigma_delta))

    with numpyro.plate("puzzles", n_puzzles):
        beta = numpyro.sample("beta", dist.Normal(0, sigma_beta))

    # Physical basis correction
    N = jnp.asarray(pieces, dtype=jnp.float32)
    g = jnp.column_stack([jnp.sqrt(N), N, N * jnp.log(N), N ** 2])
    g_ref = jnp.array([jnp.sqrt(N_REF), N_REF, N_REF * jnp.log(N_REF), N_REF ** 2])
    w = jnp.exp(log_w)
    time_contrib = jnp.dot(g, w)
    time_ref = jnp.dot(g_ref, w)
    mB_scale = 1000.0 / jnp.log(10.0)
    piece_correction = mB_scale * (jnp.log(time_contrib) - jnp.log(time_ref))

    # Practice
    sn = jnp.asarray(solve_number, dtype=jnp.float32) if solve_number is not None else jnp.ones(len(puzzler_idx))
    repeat_effect = gamma * jnp.log(sn)

    # --- Team alpha (identical to model_team) ---
    team_member_idx = jnp.asarray(team_member_idx, dtype=jnp.int32)
    team_mask_arr = jnp.asarray(team_mask, dtype=jnp.float32)
    team_size_arr = jnp.asarray(team_size, dtype=jnp.float32)

    alpha_gathered = alpha[team_member_idx]
    delta_gathered = delta[team_member_idx]
    t = jnp.asarray(year, dtype=jnp.float32) - YEAR_CENTER if year is not None else 0.0
    alpha_adjusted = alpha_gathered + (delta_0 + delta_gathered) * t[:, None]

    rates = jnp.where(team_mask_arr, jnp.exp(-alpha_adjusted / mB_scale), 0.0)
    total_rate = jnp.sum(rates, axis=1)
    alpha_parallel = -mB_scale * jnp.log(total_rate)

    s = jax.nn.sigmoid(logit_s)
    numpyro.deterministic("s", s)
    amdahl_correction = mB_scale * jnp.log(1 + s * (team_size_arr - 1))

    is_team = team_size_arr > 1
    team_bucket = jnp.clip(team_size_arr.astype(jnp.int32) - 2, 0, 2)
    eta_correction = jnp.where(is_team, eta_team[team_bucket], 0.0)

    alpha_eff = alpha_parallel + amdahl_correction + eta_correction
    sigma_eff = jnp.where(team_size_arr > 1, sigma_team, sigma)

    mean = mu + alpha_eff + beta[puzzle_idx] + piece_correction + repeat_effect

    # Normal + wide Student-t mixture likelihood
    # Normal(mean, sigma_eff) for the core; StudentT(nu, mean, k*sigma_eff) for tails
    obs_dist = NormalStudentTMixture(w_clean, mean, sigma_eff, nu, k_sigma * sigma_eff)
    with numpyro.plate("obs", len(puzzler_idx)):
        numpyro.sample("log_time", obs_dist, obs=log_time)


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

model_team_nc = reparam(model_team, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

model_team_gmm_nc = reparam(model_team_gmm, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

model_team_nst_nc = reparam(model_team_nst, config={
    "alpha": LocScaleReparam(0), "beta": LocScaleReparam(0),
    "delta": LocScaleReparam(0),
})

MODELS = {
    "model_1t": model_1t_nc,
    "model_2c": model_2c_nc,
    "model_2r": model_2r_nc,
    "model_team": model_team_nc,
    "model_team_gmm": model_team_gmm_nc,
    "model_team_nst": model_team_nst_nc,
}
