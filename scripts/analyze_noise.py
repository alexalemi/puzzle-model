"""Empirical noise structure analysis for the speed puzzling model.

Computes residuals from the fitted model and analyzes:
1. Within-event correlation of residuals (is noise independent per competitor?)
2. Variance decomposition: how much is alpha, beta, epsilon?
3. Residual distribution: Student-t vs Normal fit
4. Team noise: are team member residuals correlated?
5. Repeat-solve noise: is noise consistent across attempts?

Outputs noise_data.json for the noise.html explorer.

Usage: uv run python scripts/analyze_noise.py
"""

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from puzzle_model.data import parse_team_members, _msp_competitor_name

# --- Load model data ---
MODEL_DATA = Path("explorer_team_data.json")
DATA_PATH = Path("data/processed/combined_results.csv")
PLAYER_LINKS_PATH = Path("data/mappings/player_links.csv")
OUTPUT = Path("noise_data.json")

MB_SCALE = 1000.0 / math.log(10.0)


def fit_gaussian_mixture(z, n_restarts=5):
    """Fit a 2-component Gaussian mixture with shared mean=0 to standardized residuals.

    Model: p(z) = w * N(0, s1^2) + (1-w) * N(0, s2^2)
    Fits via EM algorithm. Returns dict with weight, sigma_narrow, sigma_wide.
    """
    best_ll = -np.inf
    best_params = None

    for _ in range(n_restarts):
        # Random init
        w = np.random.uniform(0.5, 0.9)
        s1 = np.random.uniform(0.3, 0.8)
        s2 = np.random.uniform(1.5, 4.0)

        for _ in range(200):
            # E-step: responsibilities
            p1 = w * np.exp(-0.5 * z**2 / s1**2) / (s1 * np.sqrt(2 * np.pi))
            p2 = (1 - w) * np.exp(-0.5 * z**2 / s2**2) / (s2 * np.sqrt(2 * np.pi))
            total = p1 + p2
            total = np.maximum(total, 1e-300)
            r1 = p1 / total  # responsibility for narrow component

            # M-step
            w_new = np.mean(r1)
            s1_new = np.sqrt(np.sum(r1 * z**2) / np.sum(r1))
            s2_new = np.sqrt(np.sum((1 - r1) * z**2) / np.sum(1 - r1))

            w, s1, s2 = w_new, max(s1_new, 1e-6), max(s2_new, 1e-6)

        # Ensure s1 < s2 (narrow < wide)
        if s1 > s2:
            s1, s2 = s2, s1
            w = 1 - w

        ll = np.sum(np.log(np.maximum(
            w * np.exp(-0.5 * z**2 / s1**2) / (s1 * np.sqrt(2 * np.pi)) +
            (1 - w) * np.exp(-0.5 * z**2 / s2**2) / (s2 * np.sqrt(2 * np.pi)),
            1e-300
        )))

        if ll > best_ll:
            best_ll = ll
            best_params = (w, s1, s2)

    w, s1, s2 = best_params
    return {
        "weight_narrow": round(float(w), 3),
        "sigma_narrow": round(float(s1), 3),
        "sigma_wide": round(float(s2), 3),
        "log_likelihood": round(float(best_ll / len(z)), 4),
    }


def fit_normal_studentt_mixture(z, n_restarts=20):
    """Fit a Normal + wide Student-t mixture with shared mean=0.

    Model: p(z) = w * N(0, s1²) + (1-w) * T(nu, 0, s2)
    Constraints: s2 >= s1 (Student-t is wider), w in [0.5, 0.99], nu >= 1.5.

    Uses direct optimization of the negative log-likelihood (more robust than EM
    for this mixture because the Student-t component creates degenerate optima
    that trap EM).
    """
    from scipy.special import gammaln
    from scipy.optimize import minimize

    def t_logpdf(z, nu, s):
        return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi) - np.log(s)
                - (nu + 1) / 2 * np.log(1 + z**2 / (nu * s**2)))

    def n_logpdf(z, s):
        return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * z**2 / s**2

    def neg_ll(params):
        """Negative log-likelihood in unconstrained space."""
        logit_w, log_s1, log_nu, log_k = params
        w = 0.5 + 0.49 / (1 + np.exp(-logit_w))  # w ∈ [0.5, 0.99]
        s1 = np.exp(log_s1)
        nu = 1.5 + np.exp(log_nu)  # nu > 1.5
        s2 = s1 * (1 + np.exp(log_k))  # s2 > s1

        ll = np.sum(np.logaddexp(
            np.log(w) + n_logpdf(z, s1),
            np.log(1 - w) + t_logpdf(z, nu, s2),
        ))
        return -ll / len(z)  # normalize for numerical stability

    best_result = None
    best_val = np.inf

    for _ in range(n_restarts):
        # Random starting points in unconstrained space
        x0 = [
            np.random.uniform(-1, 3),       # logit_w
            np.log(np.random.uniform(0.3, 0.8)),  # log_s1
            np.log(np.random.uniform(0.5, 10)),   # log_nu offset
            np.log(np.random.uniform(0.2, 4)),     # log_k (s2/s1 - 1)
        ]
        res = minimize(neg_ll, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res

    logit_w, log_s1, log_nu, log_k = best_result.x
    w = 0.5 + 0.49 / (1 + np.exp(-logit_w))
    s1 = np.exp(log_s1)
    nu = 1.5 + np.exp(log_nu)
    s2 = s1 * (1 + np.exp(log_k))

    return {
        "w_normal": round(float(w), 3),
        "sigma_normal": round(float(s1), 3),
        "nu": round(float(nu), 2),
        "sigma_t": round(float(s2), 3),
        "log_likelihood": round(float(-best_val), 4),
    }


def load_model():
    """Load model posteriors."""
    return json.load(MODEL_DATA.open())


def load_player_links():
    """Load MSP player_id → SP name mappings."""
    if not PLAYER_LINKS_PATH.exists():
        return {}
    pl = pd.read_csv(PLAYER_LINKS_PATH)
    return dict(zip(pl["msp_player_id"], pl["sp_competitor_name"]))


def load_data():
    """Load combined results."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["completed"] == True].copy()
    df = df[df["time_seconds"] > 0].copy()
    df["log_time"] = 1000 * np.log10(df["time_seconds"])
    df["puzzle_pieces"] = pd.to_numeric(df["puzzle_pieces"], errors="coerce")
    df = df.dropna(subset=["puzzle_pieces"])
    df["puzzle_pieces"] = df["puzzle_pieces"].astype(int)
    df["puzzle_id"] = df["puzzle_name"].astype(str) + "_" + df["puzzle_pieces"].astype(str)
    return df


def compute_piece_correction(N, log_w):
    """Compute piece correction in mB."""
    N_REF = 500.0
    w = [math.exp(lw) for lw in log_w]
    def g(n):
        return [math.sqrt(n), n, n * math.log(n), n * n]
    gN, gR = g(float(N)), g(N_REF)
    dot_N = sum(wi * gi for wi, gi in zip(w, gN))
    dot_R = sum(wi * gi for wi, gi in zip(w, gR))
    return MB_SCALE * (math.log(dot_N) - math.log(dot_R))


def compute_team_alpha(member_alphas, team_size, s, eta_by_k):
    """Compute effective alpha for a team observation using the full team model.

    alpha_parallel = -MB_SCALE * log(sum_i exp(-alpha_i / MB_SCALE))
    amdahl = MB_SCALE * log(1 + s*(K-1))
    eta = eta_by_k[min(K, 4)] for K >= 2, else 0

    Returns alpha_eff = alpha_parallel + amdahl + eta
    """
    K = team_size
    if K <= 1:
        return member_alphas[0]

    # Parallel speedup: logsumexp of rates
    # rate_i = exp(-alpha_i / MB_SCALE), so log(sum rates) = logsumexp(-alpha/MB)
    neg_alpha_scaled = [-a / MB_SCALE for a in member_alphas]
    max_val = max(neg_alpha_scaled)
    log_sum_rates = max_val + math.log(sum(math.exp(v - max_val) for v in neg_alpha_scaled))
    alpha_parallel = -MB_SCALE * log_sum_rates

    # Amdahl's law serial fraction correction (positive = slower)
    amdahl = MB_SCALE * math.log(1 + s * (K - 1))

    # Per-bucket eta correction
    bucket = min(K, 4)  # K=2 → 2, K=3 → 3, K≥4 → 4
    eta = eta_by_k.get(bucket, 0.0)

    return alpha_parallel + amdahl + eta


def resolve_team_members(tm_string, pmap, player_links):
    """Parse team_members string, resolve names, return list of (name, alpha) for matched members."""
    members = parse_team_members(tm_string)
    result = []
    for name, pid in members:
        comp_name = _msp_competitor_name(name, pid, player_links)
        if comp_name in pmap:
            result.append((comp_name, pmap[comp_name]["alpha_proj"]))
        elif name in pmap:
            # SP names in team_members may not have player_id
            result.append((name, pmap[name]["alpha_proj"]))
    return result


def compute_predicted(row, mu, log_w, pmap, puzzle_map, team_ctx):
    """Compute full predicted log_time for any observation (solo or team).

    Returns (predicted, team_size) or (None, None) if observation can't be matched.
    """
    name = row["competitor_name"]
    pid = row.get("puzzle_id")
    if pid not in puzzle_map:
        return None, None

    pz = puzzle_map[pid]
    pc = compute_piece_correction(row["puzzle_pieces"], log_w) if pd.notna(row.get("puzzle_pieces")) else 0
    beta = pz["beta"]

    tm = row.get("team_members", "")
    division = row.get("division", "solo")
    is_team = division in ("duo", "group") and tm and not pd.isna(tm)

    if is_team and team_ctx:
        # Full team prediction
        members = resolve_team_members(tm, team_ctx["pmap"], team_ctx["player_links"])
        if not members:
            return None, None
        member_alphas = [a for _, a in members]
        team_size = len(members)
        alpha_eff = compute_team_alpha(member_alphas, team_size, team_ctx["s"], team_ctx["eta_by_k"])
        return mu + alpha_eff + beta + pc, team_size
    else:
        # Solo prediction
        if name not in pmap:
            return None, None
        alpha = pmap[name]["alpha_proj"]
        return mu + alpha + beta + pc, 1


def analyze_solo_residuals(df, model, player_links):
    """Analyze residuals for all observations (solo + team)."""
    stats = model["stats"]
    mu = stats["mu_fixed"]
    log_w = stats["log_w"]
    year_center = stats["year_center"]
    ranking_year = stats["ranking_year"]

    # Build puzzler and puzzle maps
    pmap = {p["name"]: p for p in model["puzzlers"]}
    puzzle_map = {}
    for p in model["puzzles"]:
        puzzle_map[p["puzzle_id"]] = p

    # Team model parameters
    sp = model["scalar_params"]["joint"]
    s = sp["s"]["mean"]
    eta_by_k = {
        2: sp["eta_2"]["mean"],
        3: sp["eta_3"]["mean"],
        4: sp["eta_4plus"]["mean"],
    }

    team_ctx = {"pmap": pmap, "player_links": player_links, "s": s, "eta_by_k": eta_by_k}

    results = {
        "within_event": analyze_within_event(df, mu, log_w, pmap, puzzle_map, ranking_year),
        "variance_decomp": analyze_variance_decomposition(df, mu, log_w, pmap, puzzle_map, team_ctx),
        "residual_distribution": analyze_residual_distribution(df, mu, log_w, pmap, puzzle_map, team_ctx),
        "repeat_consistency": analyze_repeat_consistency(df, mu, log_w, pmap, puzzle_map),
    }
    return results


def _compute_residuals_for_group(group, mu, log_w, pmap):
    """Compute residuals for a group of observations."""
    residuals = []
    for _, row in group.iterrows():
        name = row["competitor_name"]
        if name not in pmap:
            continue
        p = pmap[name]
        pc = compute_piece_correction(row["puzzle_pieces"], log_w) if pd.notna(row["puzzle_pieces"]) else 0
        expected = mu + p["alpha_proj"] + pc
        residuals.append(row["log_time"] - expected)
    return residuals


def _compute_pairwise_correlation(groups, mu, log_w, pmap, min_size=15, max_pairs=50):
    """Compute pairwise within-group correlation of deviations from group mean."""
    all_devs = []
    within_prods = []

    for _, group in groups:
        resids = _compute_residuals_for_group(group, mu, log_w, pmap)
        if len(resids) < min_size:
            continue
        mean_r = np.mean(resids)
        devs = [r - mean_r for r in resids]
        all_devs.extend(devs)
        for i, j in combinations(range(min(len(devs), max_pairs)), 2):
            within_prods.append(devs[i] * devs[j])

    var_devs = np.var(all_devs) if all_devs else 1.0
    corr = float(np.mean(within_prods) / var_devs) if within_prods else 0.0
    return corr, all_devs


def analyze_within_event(df, mu, log_w, pmap, puzzle_map, ranking_year):
    """Check within-event noise correlation, grouped by event+puzzle and broken down by source."""
    print("  Analyzing within-event correlations...")

    # Create event+puzzle grouping key
    df = df.copy()
    df["event_puzzle"] = (
        df["event_id"].astype(str) + "_" + df["puzzle_name"].astype(str) + "_" + df["puzzle_pieces"].astype(str)
    )

    source_results = {}
    for source in ["speedpuzzling", "myspeedpuzzling", "usajigsaw"]:
        sdf = df[df["source"] == source]
        if len(sdf) < 100:
            continue

        # Group by event+puzzle (each group = same puzzle for all competitors)
        groups = list(sdf.groupby("event_puzzle"))
        n_groups = sum(1 for _, g in groups if len(g) >= 10)

        corr, all_devs = _compute_pairwise_correlation(groups, mu, log_w, pmap)

        # Also compute event-mean std
        event_means = []
        for _, group in groups:
            resids = _compute_residuals_for_group(group, mu, log_w, pmap)
            if len(resids) >= 10:
                event_means.append(np.mean(resids))

        dev_std = float(np.std(all_devs)) if all_devs else 0
        event_std = float(np.std(event_means)) if event_means else 0

        print(f"    {source}: corr={corr:.4f}, n_groups={n_groups}, dev_std={dev_std:.1f}")

        source_results[source] = {
            "n_groups": n_groups,
            "n_deviations": len(all_devs),
            "pairwise_correlation": round(corr, 4),
            "event_mean_std": round(event_std, 1),
            "individual_dev_std": round(dev_std, 1),
            "dev_histogram": make_histogram(all_devs, 60),
        }

    # Also check if correlation depends on number of observations per puzzler
    # (well-measured vs poorly-measured)
    sp = df[df["source"] == "speedpuzzling"].copy()
    sp["n_obs"] = sp["competitor_name"].map(
        lambda n: pmap[n]["n_solo"] if n in pmap else 0
    )
    # Split into well-measured (n >= 5) and poorly-measured (n < 5)
    for label, condition in [("n_solo>=5", sp["n_obs"] >= 5), ("n_solo<5", sp["n_obs"] < 5)]:
        sub = sp[condition]
        groups = list(sub.groupby("event_puzzle"))
        corr, devs = _compute_pairwise_correlation(groups, mu, log_w, pmap, min_size=5)
        print(f"    SP {label}: corr={corr:.4f}, n_devs={len(devs)}")
        source_results[f"sp_{label}"] = {
            "pairwise_correlation": round(corr, 4),
            "n_deviations": len(devs),
        }

    return {
        "by_source": source_results,
        "interpretation": (
            "SP events show ~15% within-event correlation after removing the shared puzzle effect. "
            "MSP shows near-zero (~4%), consistent with independent solves. "
            "The SP correlation likely reflects shared environmental conditions (venue, timing) "
            "that create a small common noise component. For ranking predictions this cancels out "
            "(affects everyone equally), but matters for absolute time predictions."
        ),
    }


def analyze_variance_decomposition(df, mu, log_w, pmap, puzzle_map, team_ctx):
    """Decompose total variance into alpha, beta, and noise components."""
    print("  Analyzing variance decomposition...")

    rows = []
    for _, row in df.iterrows():
        predicted, team_size = compute_predicted(row, mu, log_w, pmap, puzzle_map, team_ctx)
        if predicted is None:
            continue
        name = row["competitor_name"]
        pid = row.get("puzzle_id")
        pz = puzzle_map[pid]
        # For variance decomposition, use the solo alpha (individual component)
        alpha = pmap[name]["alpha_proj"] if name in pmap else 0
        pc = compute_piece_correction(row["puzzle_pieces"], log_w) if pd.notna(row.get("puzzle_pieces")) else 0
        rows.append({
            "log_time": row["log_time"],
            "predicted": predicted,
            "alpha": alpha,
            "beta": pz["beta"],
            "pc": pc,
            "residual": row["log_time"] - predicted,
        })

    if not rows:
        return {"error": "No matched observations"}

    rdf = pd.DataFrame(rows)
    total_var = rdf["log_time"].var()
    alpha_var = rdf["alpha"].var()
    beta_var = rdf["beta"].var()
    pc_var = rdf["pc"].var()
    residual_var = rdf["residual"].var()

    print(f"    Total variance: {total_var:.0f}")
    print(f"    Alpha (puzzler) variance: {alpha_var:.0f} ({100*alpha_var/total_var:.1f}%)")
    print(f"    Beta (puzzle) variance: {beta_var:.0f} ({100*beta_var/total_var:.1f}%)")
    print(f"    Piece count variance: {pc_var:.0f} ({100*pc_var/total_var:.1f}%)")
    print(f"    Residual variance: {residual_var:.0f} ({100*residual_var/total_var:.1f}%)")

    # Per-source breakdown
    source_decomp = {}
    for source in df["source"].unique():
        sdf = df[df["source"] == source]
        s_rows = []
        for _, row in sdf.iterrows():
            predicted, _ = compute_predicted(row, mu, log_w, pmap, puzzle_map, team_ctx)
            if predicted is None:
                continue
            s_rows.append({"residual": row["log_time"] - predicted})
        if s_rows:
            source_decomp[source] = {
                "n": len(s_rows),
                "residual_std": round(float(pd.DataFrame(s_rows)["residual"].std()), 1),
            }

    return {
        "n_obs": len(rdf),
        "total_std": round(float(np.sqrt(total_var)), 1),
        "components": [
            {"name": "Puzzler (alpha)", "variance": round(float(alpha_var), 0), "pct": round(100 * alpha_var / total_var, 1)},
            {"name": "Puzzle (beta)", "variance": round(float(beta_var), 0), "pct": round(100 * beta_var / total_var, 1)},
            {"name": "Piece count", "variance": round(float(pc_var), 0), "pct": round(100 * pc_var / total_var, 1)},
            {"name": "Residual (noise)", "variance": round(float(residual_var), 0), "pct": round(100 * residual_var / total_var, 1)},
        ],
        "by_source": source_decomp,
        "residual_histogram": make_histogram(rdf["residual"].tolist(), 60),
    }


def analyze_residual_distribution(df, mu, log_w, pmap, puzzle_map, team_ctx):
    """Compare residual distribution to Normal and Student-t."""
    print("  Analyzing residual distribution...")

    residuals = []
    sources = []
    puzzle_ids = []
    divisions = []
    puzzler_names = []
    alphas = []
    betas = []
    for _, row in df.iterrows():
        predicted, team_size = compute_predicted(row, mu, log_w, pmap, puzzle_map, team_ctx)
        if predicted is None:
            continue
        residuals.append(row["log_time"] - predicted)
        sources.append(row["source"])
        puzzle_ids.append(row.get("puzzle_id"))
        divisions.append(row.get("division", "unknown"))
        puzzler_names.append(row["competitor_name"])
        # Track alpha and beta for quadrant analysis
        name = row["competitor_name"]
        pid = row.get("puzzle_id")
        alphas.append(pmap[name]["alpha_proj"] if name in pmap else 0)
        betas.append(puzzle_map[pid]["beta"] if pid in puzzle_map else 0)

    if not residuals:
        return {}

    residuals = np.array(residuals)
    sources = np.array(sources)
    puzzle_ids = np.array(puzzle_ids)
    divisions = np.array(divisions)
    puzzler_names = np.array(puzzler_names)
    alphas = np.array(alphas)
    betas = np.array(betas)
    std = np.std(residuals)
    mean = np.mean(residuals)

    # Standardize
    z = (residuals - mean) / std

    # Compute empirical quantiles
    quantiles = [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]
    empirical_q = [float(np.quantile(z, q)) for q in quantiles]

    # Normal theoretical quantiles
    from scipy import stats as sp_stats
    normal_q = [float(sp_stats.norm.ppf(q)) for q in quantiles]

    # Student-t theoretical quantiles (fit nu)
    # Fit Student-t to standardized residuals
    nu_fit, _, scale_fit = sp_stats.t.fit(z)
    t_q = [float(sp_stats.t.ppf(q, nu_fit)) for q in quantiles]

    # Kurtosis (excess)
    kurtosis = float(sp_stats.kurtosis(z))
    # Normal kurtosis = 0, Student-t(3) kurtosis ≈ ∞

    print(f"    Residual std: {std:.1f} mB")
    print(f"    Fitted Student-t df: {nu_fit:.2f}")
    print(f"    Excess kurtosis: {kurtosis:.2f} (Normal=0, t(3)=∞)")

    # Fit Gaussian mixture (shared mean=0)
    gmm = fit_gaussian_mixture(z)
    print(f"    GMM: w_narrow={gmm['weight_narrow']:.2f}, "
          f"σ_narrow={gmm['sigma_narrow']:.2f}, σ_wide={gmm['sigma_wide']:.2f}")
    # Convert back to mB
    gmm_mB = {
        "weight_narrow": gmm["weight_narrow"],
        "sigma_narrow_mB": round(gmm["sigma_narrow"] * std, 1),
        "sigma_wide_mB": round(gmm["sigma_wide"] * std, 1),
        **gmm,
    }

    # Fit Normal + Student-t mixture (shared mean=0)
    nst = fit_normal_studentt_mixture(z)
    print(f"    NST: w_normal={nst['w_normal']:.2f}, σ_normal={nst['sigma_normal']:.2f}, "
          f"ν={nst['nu']:.2f}, σ_t={nst['sigma_t']:.2f}")
    nst_mB = {
        **nst,
        "sigma_normal_mB": round(nst["sigma_normal"] * std, 1),
        "sigma_t_mB": round(nst["sigma_t"] * std, 1),
    }

    # NST theoretical quantiles (numerically invert the mixture CDF)
    from scipy.optimize import brentq
    w_nst, s1_nst, nu_nst, s2_nst = nst["w_normal"], nst["sigma_normal"], nst["nu"], nst["sigma_t"]
    def nst_cdf(z_val):
        return w_nst * sp_stats.norm.cdf(z_val, scale=s1_nst) + (1 - w_nst) * sp_stats.t.cdf(z_val, nu_nst, scale=s2_nst)
    nst_q = []
    for q in quantiles:
        try:
            nst_q.append(float(brentq(lambda z_val: nst_cdf(z_val) - q, -50, 50)))
        except ValueError:
            nst_q.append(float("nan"))
    print(f"    NST quantiles computed: {[round(v, 2) for v in nst_q]}")

    # Tail fractions
    tail_fracs = {}
    for threshold in [2, 3, 4, 5]:
        frac = float(np.mean(np.abs(z) > threshold))
        normal_frac = float(2 * sp_stats.norm.sf(threshold))
        t_frac = float(2 * sp_stats.t.sf(threshold, nu_fit))
        tail_fracs[str(threshold)] = {
            "empirical": round(frac, 6),
            "normal": round(normal_frac, 6),
            "student_t": round(t_frac, 6),
        }

    # Per-source histograms (standardized with global mean/std for comparability)
    by_source = {}
    for src in ["myspeedpuzzling", "speedpuzzling", "usajigsaw"]:
        mask = sources == src
        if mask.sum() < 50:
            continue
        src_resids = residuals[mask]
        src_z = (src_resids - mean) / std
        src_std = float(np.std(src_resids))
        src_mean = float(np.mean(src_resids))
        src_nu, _, _ = sp_stats.t.fit(src_z)
        src_kurtosis = float(sp_stats.kurtosis(src_z))
        print(f"    {src}: n={mask.sum()}, mean={src_mean:.1f}, std={src_std:.1f}, "
              f"fitted_nu={src_nu:.2f}, kurtosis={src_kurtosis:.2f}")
        by_source[src] = {
            "n_obs": int(mask.sum()),
            "mean": round(src_mean, 1),
            "std": round(src_std, 1),
            "fitted_nu": round(src_nu, 2),
            "excess_kurtosis": round(src_kurtosis, 2),
            "histogram": make_histogram(src_z.tolist(), 80, range_limit=6),
        }

    # Per-puzzle observation count breakdown
    # Count how many observations each puzzle has in the matched data
    from collections import Counter
    puzzle_obs_counts = Counter(puzzle_ids)
    obs_per_puzzle = np.array([puzzle_obs_counts[pid] for pid in puzzle_ids])

    by_puzzle_n = {}
    bins_spec = [
        ("n=1", 1, 1),
        ("n=2-5", 2, 5),
        ("n=6-20", 6, 20),
        ("n=21-100", 21, 100),
        ("n>100", 101, None),
    ]
    for label, lo, hi in bins_spec:
        if hi is not None:
            mask = (obs_per_puzzle >= lo) & (obs_per_puzzle <= hi)
        else:
            mask = obs_per_puzzle >= lo
        if mask.sum() < 50:
            continue
        bin_resids = residuals[mask]
        bin_z = (bin_resids - mean) / std
        bin_std = float(np.std(bin_resids))
        bin_mean = float(np.mean(bin_resids))
        n_puzzles = len(set(puzzle_ids[mask]))
        print(f"    {label}: n_obs={mask.sum()}, n_puzzles={n_puzzles}, "
              f"mean={bin_mean:.1f}, std={bin_std:.1f}")
        by_puzzle_n[label] = {
            "n_obs": int(mask.sum()),
            "n_puzzles": n_puzzles,
            "mean": round(bin_mean, 1),
            "std": round(bin_std, 1),
            "histogram": make_histogram(bin_z.tolist(), 80, range_limit=6),
        }

    # By division (solo / duo / group)
    by_division = {}
    for div in ["solo", "duo", "group"]:
        mask = divisions == div
        if mask.sum() < 50:
            continue
        div_resids = residuals[mask]
        div_z = (div_resids - np.mean(div_resids)) / np.std(div_resids)
        div_std = float(np.std(div_resids))
        div_mean = float(np.mean(div_resids))
        div_nu, _, _ = sp_stats.t.fit(div_z)
        div_gmm = fit_gaussian_mixture(div_z)
        print(f"    division={div}: n_obs={mask.sum()}, mean={div_mean:.1f}, "
              f"std={div_std:.1f}, fitted_nu={div_nu:.2f}, "
              f"GMM w={div_gmm['weight_narrow']:.2f} "
              f"σ_n={div_gmm['sigma_narrow']:.2f} σ_w={div_gmm['sigma_wide']:.2f}")
        by_division[div] = {
            "n_obs": int(mask.sum()),
            "mean": round(div_mean, 1),
            "std": round(div_std, 1),
            "fitted_nu": round(div_nu, 2),
            "gmm": {
                **div_gmm,
                "sigma_narrow_mB": round(div_gmm["sigma_narrow"] * div_std, 1),
                "sigma_wide_mB": round(div_gmm["sigma_wide"] * div_std, 1),
            },
            "histogram": make_histogram(div_z.tolist(), 80, range_limit=6),
        }

    # By puzzler observation count
    from collections import Counter
    puzzler_obs_counts = Counter(puzzler_names)
    obs_per_puzzler = np.array([puzzler_obs_counts[n] for n in puzzler_names])

    by_puzzler_n = {}
    puzzler_bins = [
        ("n=1", 1, 1),
        ("n=2-5", 2, 5),
        ("n=6-20", 6, 20),
        ("n=21-100", 21, 100),
        ("n>100", 101, None),
    ]
    for label, lo, hi in puzzler_bins:
        if hi is not None:
            mask = (obs_per_puzzler >= lo) & (obs_per_puzzler <= hi)
        else:
            mask = obs_per_puzzler >= lo
        if mask.sum() < 50:
            continue
        bin_resids = residuals[mask]
        bin_z = (bin_resids - mean) / std
        bin_std = float(np.std(bin_resids))
        bin_mean = float(np.mean(bin_resids))
        n_puzzlers = len(set(puzzler_names[mask]))
        print(f"    puzzler {label}: n_obs={mask.sum()}, n_puzzlers={n_puzzlers}, "
              f"mean={bin_mean:.1f}, std={bin_std:.1f}")
        by_puzzler_n[label] = {
            "n_obs": int(mask.sum()),
            "n_puzzlers": n_puzzlers,
            "mean": round(bin_mean, 1),
            "std": round(bin_std, 1),
            "histogram": make_histogram(bin_z.tolist(), 80, range_limit=6),
        }

    # Quadrant analysis: skill × difficulty interaction
    # Split at median alpha (puzzler skill) and median beta (puzzle difficulty)
    # Lower alpha = faster = "good"; lower beta = easier
    alpha_median = float(np.median(alphas))
    beta_median = float(np.median(betas))
    print(f"    Quadrant split: alpha_median={alpha_median:.1f} mB, beta_median={beta_median:.1f} mB")

    quadrant_labels = {
        "easy_good": "Easy puzzle, Good puzzler",
        "easy_worse": "Easy puzzle, Worse puzzler",
        "hard_good": "Hard puzzle, Good puzzler",
        "hard_worse": "Hard puzzle, Worse puzzler",
    }
    quadrant_masks = {
        "easy_good": (betas <= beta_median) & (alphas <= alpha_median),
        "easy_worse": (betas <= beta_median) & (alphas > alpha_median),
        "hard_good": (betas > beta_median) & (alphas <= alpha_median),
        "hard_worse": (betas > beta_median) & (alphas > alpha_median),
    }

    by_quadrant = {}
    for qkey, qmask in quadrant_masks.items():
        if qmask.sum() < 50:
            continue
        q_resids = residuals[qmask]
        q_z = (q_resids - mean) / std
        q_std = float(np.std(q_resids))
        q_mean = float(np.mean(q_resids))
        q_skew = float(sp_stats.skew(q_resids))
        q_kurtosis = float(sp_stats.kurtosis(q_z))
        n_puzzlers_q = len(set(puzzler_names[qmask]))
        n_puzzles_q = len(set(puzzle_ids[qmask]))
        print(f"    {quadrant_labels[qkey]}: n_obs={qmask.sum()}, "
              f"mean={q_mean:.1f}, std={q_std:.1f}, skew={q_skew:.2f}")
        by_quadrant[qkey] = {
            "label": quadrant_labels[qkey],
            "n_obs": int(qmask.sum()),
            "n_puzzlers": n_puzzlers_q,
            "n_puzzles": n_puzzles_q,
            "mean": round(q_mean, 1),
            "std": round(q_std, 1),
            "skewness": round(q_skew, 3),
            "excess_kurtosis": round(q_kurtosis, 2),
            "histogram": make_histogram(q_z.tolist(), 80, range_limit=6),
        }

    # Also compute a simple 2D scatter summary: mean residual in bins of alpha and beta
    # Use quintiles for a 5x5 grid
    alpha_edges = np.quantile(alphas, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    beta_edges = np.quantile(betas, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    heatmap = []
    for i in range(5):
        for j in range(5):
            mask_ij = (
                (alphas >= alpha_edges[i]) & (alphas < alpha_edges[i + 1] + (1 if i == 4 else 0))
                & (betas >= beta_edges[j]) & (betas < beta_edges[j + 1] + (1 if j == 4 else 0))
            )
            if mask_ij.sum() > 0:
                heatmap.append({
                    "alpha_bin": i,
                    "beta_bin": j,
                    "alpha_center": round(float((alpha_edges[i] + alpha_edges[i + 1]) / 2), 0),
                    "beta_center": round(float((beta_edges[j] + beta_edges[j + 1]) / 2), 0),
                    "mean_residual": round(float(np.mean(residuals[mask_ij])), 1),
                    "std_residual": round(float(np.std(residuals[mask_ij])), 1),
                    "n_obs": int(mask_ij.sum()),
                })

    return {
        "n_obs": len(residuals),
        "mean": round(float(mean), 1),
        "std": round(float(std), 1),
        "fitted_nu": round(nu_fit, 2),
        "excess_kurtosis": round(kurtosis, 2),
        "gmm": gmm_mB,
        "nst": nst_mB,
        "quantiles": quantiles,
        "empirical_quantiles": [round(q, 3) for q in empirical_q],
        "normal_quantiles": [round(q, 3) for q in normal_q],
        "student_t_quantiles": [round(q, 3) for q in t_q],
        "nst_quantiles": [round(q, 3) for q in nst_q],
        "tail_fractions": tail_fracs,
        "histogram": make_histogram(z.tolist(), 80, range_limit=6),
        "by_source": by_source,
        "by_puzzle_n": by_puzzle_n,
        "by_division": by_division,
        "by_puzzler_n": by_puzzler_n,
        "by_quadrant": by_quadrant,
        "quadrant_heatmap": heatmap,
        "alpha_median": round(alpha_median, 1),
        "beta_median": round(beta_median, 1),
    }


def analyze_repeat_consistency(df, mu, log_w, pmap, puzzle_map):
    """Check if the same competitor's residuals are consistent across events."""
    print("  Analyzing repeat-solve consistency...")

    # For competitors who appear in multiple events, compute their residuals
    # and check the correlation between residuals across events
    competitor_residuals = {}

    for _, row in df.iterrows():
        name = row["competitor_name"]
        pid = row.get("puzzle_id")
        if name not in pmap or pid not in puzzle_map:
            continue
        p = pmap[name]
        pz = puzzle_map[pid]
        pc = compute_piece_correction(row["puzzle_pieces"], log_w) if pd.notna(row["puzzle_pieces"]) else 0
        predicted = mu + p["alpha_proj"] + pz["beta"] + pc
        resid = row["log_time"] - predicted

        if name not in competitor_residuals:
            competitor_residuals[name] = []
        competitor_residuals[name].append(resid)

    # For competitors with >= 5 observations, compute within-person variance
    within_vars = []
    between_means = []
    n_obs_list = []

    for name, resids in competitor_residuals.items():
        if len(resids) >= 5:
            within_vars.append(np.var(resids))
            between_means.append(np.mean(resids))
            n_obs_list.append(len(resids))

    if not within_vars:
        return {}

    # ICC (intraclass correlation): how much of residual variance is between-person vs within-person
    mean_within = np.mean(within_vars)
    var_between = np.var(between_means)

    print(f"    Competitors with >=5 obs: {len(within_vars)}")
    print(f"    Mean within-person variance: {mean_within:.0f} (std={np.sqrt(mean_within):.1f} mB)")
    print(f"    Between-person mean variance: {var_between:.0f} (std={np.sqrt(var_between):.1f} mB)")

    # If within-person variance is much less than total, the noise is person-specific
    # (some people are consistently over/under-predicted)
    icc = var_between / (var_between + mean_within) if (var_between + mean_within) > 0 else 0

    print(f"    ICC (residual): {icc:.3f}")
    print(f"    Interpretation: {icc:.0%} of residual variance is stable across solves (person-specific bias)")

    # Distribution of within-person std
    within_stds = [np.sqrt(v) for v in within_vars]

    return {
        "n_competitors": len(within_vars),
        "mean_within_std": round(float(np.sqrt(mean_within)), 1),
        "between_person_std": round(float(np.sqrt(var_between)), 1),
        "icc": round(icc, 3),
        "within_std_histogram": make_histogram(within_stds, 40),
        "interpretation": f"{100*icc:.0f}% of residual variance is person-specific (stable bias). "
                          f"The remaining {100*(1-icc):.0f}% is genuine solve-to-solve noise.",
    }


def make_histogram(values, n_bins, range_limit=None):
    """Create histogram data for D3."""
    values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
    if not values:
        return {"bins": [], "edges": []}
    if range_limit:
        values = [v for v in values if abs(v) <= range_limit]
    counts, edges = np.histogram(values, bins=n_bins)
    total = sum(counts)
    bin_width = edges[1] - edges[0]
    bins = []
    for i in range(len(counts)):
        bins.append({
            "x0": round(float(edges[i]), 2),
            "x1": round(float(edges[i + 1]), 2),
            "count": int(counts[i]),
            "density": round(float(counts[i]) / total / bin_width, 4) if total > 0 else 0,
        })
    return {"bins": bins, "n": len(values)}


def main():
    print("Loading data...")
    model = load_model()
    df = load_data()
    player_links = load_player_links()

    print(f"Loaded {len(df)} observations, {len(player_links)} player links")

    print("Analyzing noise structure (full team model)...")
    results = analyze_solo_residuals(df, model, player_links)

    # Add model params for context
    stats = model["stats"]
    sp = model["scalar_params"]["joint"]
    results["model_params"] = {
        "mu_fixed": stats["mu_fixed"],
        "sigma": sp["sigma"]["mean"],
        "nu": sp["nu"]["mean"],
        "w_clean": sp["w_clean"]["mean"],
        "k_sigma": sp["k_sigma"]["mean"],
        "sigma_alpha": sp["sigma_alpha"]["mean"],
        "sigma_beta": sp["sigma_beta"]["mean"],
        "sigma_team": sp["sigma_team"]["mean"],
    }

    # Sanitize
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return round(obj, 6)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    results = sanitize(results)
    OUTPUT.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
