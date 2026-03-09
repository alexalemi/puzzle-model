"""Monte Carlo simulation of the 2026 USA Jigsaw Nationals.

Loads model posteriors from explorer_team_data.json and tournament
structure from data/tournament/nationals_2026.json, runs N_MC stochastic
simulations, and writes prediction_data.json for the explorer.

Usage: uv run python scripts/simulate_nationals.py
"""

import json
import math
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

# --- Configuration ---
N_MC = 20_000
TOURNAMENT_YEAR = 2026.24  # Late March 2026

# Empirical cold-start prior for nationals competitors, estimated from
# competitors who were unknown at the 2024 and 2025 USA Jigsaw Nationals
# but are now in the model (n=58, pooled across both years).
# Much tighter than the model-wide sigma_alpha=219 because nationals
# self-selects for serious puzzlers.
COLD_START_MEAN = -6.0   # mB (roughly average)
COLD_START_STD = 63.0    # mB (vs model sigma_alpha=219)

# --- File paths ---
MODEL_DATA = Path("explorer_team_data.json")
TOURNAMENT_DATA = Path("data/tournament/nationals_2026.json")
OUTPUT = Path("prediction_data.json")

# --- Constants ---
MB_SCALE = 1000.0 / math.log(10.0)  # ≈ 434.294
N_REF = 500.0


def load_data():
    """Load model and tournament data."""
    model = json.load(MODEL_DATA.open())
    tournament = json.load(TOURNAMENT_DATA.open())
    return model, tournament


def build_puzzler_map(model: dict) -> dict:
    """Build name -> puzzler lookup from model data."""
    pmap = {}
    for p in model["puzzlers"]:
        pmap[p["name"]] = p
        # Also index by lowercase for case-insensitive matching
        pmap[p["name"].lower()] = p
    return pmap


def strip_msp_tag(name: str) -> str:
    """Strip '(msp:...)' suffix from MSP display names."""
    idx = name.find(" (msp:")
    return name[:idx].strip() if idx >= 0 else name


def match_name(name: str, pmap: dict, model_puzzlers: list) -> tuple[str | None, dict | None]:
    """Match a tournament name against the model puzzler map.

    Returns (model_name, puzzler_dict) or (None, None) for cold start.
    """
    # 1. Exact match
    if name in pmap:
        p = pmap[name]
        return p["name"], p
    if name.lower() in pmap:
        p = pmap[name.lower()]
        return p["name"], p

    # 2. Try all splits of "Last, First" name
    if ", " in name:
        last, first = name.split(", ", 1)
        parts = first.split() + [last]
        # Try different splits: i words as first name, rest as last
        for i in range(1, len(parts)):
            attempt = f"{' '.join(parts[i:])}, {' '.join(parts[:i])}"
            if attempt.lower() in pmap:
                p = pmap[attempt.lower()]
                return p["name"], p

    # 3. Fuzzy match — only check "Last, First" format names (SP-style, not MSP display names)
    #    This avoids checking 10K+ MSP names and focuses on the ~2K SP names.
    best_ratio = 0.0
    best_match = None
    name_lower = name.lower()
    for p in model_puzzlers:
        pname = p["name"]
        # Skip MSP display names (no comma = not "Last, First" format)
        if ", " not in pname:
            continue
        ratio = SequenceMatcher(None, name_lower, pname.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = p
    if best_ratio >= 0.85:
        return best_match["name"], best_match

    return None, None


def match_all_names(tournament: dict, pmap: dict, model_puzzlers: list) -> dict:
    """Match all tournament competitor names to model puzzlers.

    Returns {tournament_name: (model_name, puzzler_dict_or_None)}.
    """
    # Collect all unique names
    all_names = set()
    for heat_names in tournament["solo"]["heats"].values():
        all_names.update(heat_names)
    for heat_pairs in tournament["pairs"]["heats"].values():
        for pair in heat_pairs:
            all_names.update(pair)
    for heat_teams in tournament["teams"]["heats"].values():
        for team in heat_teams:
            all_names.update(team["members"])

    matches = {}
    matched = 0
    cold = 0
    for name in sorted(all_names):
        model_name, pdata = match_name(name, pmap, model_puzzlers)
        if pdata is not None:
            matches[name] = (model_name, pdata)
            matched += 1
        else:
            matches[name] = (None, None)
            cold += 1

    print(f"  Name matching: {matched} matched, {cold} cold-start, {matched + cold} total")
    return matches


def get_alpha_params(name: str, matches: dict, stats: dict, year: float) -> tuple[float, float]:
    """Get (alpha_mean, alpha_std) for a competitor at given year.

    For matched competitors with solo data, project alpha to tournament year.
    For anyone with no solo observations (including model matches with only
    team data), use empirical nationals prior: Normal(COLD_START_MEAN, COLD_START_STD).
    """
    model_name, pdata = matches.get(name, (None, None))
    if pdata is not None and pdata.get("n_solo", 0) > 0:
        ranking_year = stats["ranking_year"]
        dt = year - ranking_year
        alpha_mean = pdata["alpha_proj"] - pdata["velocity"] * dt
        alpha_std = math.sqrt(pdata["std"] ** 2 + pdata["velocity_std"] ** 2 * dt ** 2)
        return alpha_mean, alpha_std
    else:
        return COLD_START_MEAN, COLD_START_STD


def compute_piece_correction(pieces: int, log_w: list[float]) -> float:
    """Compute mB piece-count correction relative to N_REF=500.

    g(N) = [sqrt(N), N, N*log(N), N^2]
    correction = mB_scale * (log(dot(g(N), w)) - log(dot(g(N_REF), w)))
    """
    w = [math.exp(lw) for lw in log_w]
    N = float(pieces)

    def g(n):
        return [math.sqrt(n), n, n * math.log(n), n * n]

    gN = g(N)
    gR = g(N_REF)
    dot_N = sum(wi * gi for wi, gi in zip(w, gN))
    dot_R = sum(wi * gi for wi, gi in zip(w, gR))
    return MB_SCALE * (math.log(dot_N) - math.log(dot_R))


def simulate_solo(tournament: dict, matches: dict, params: dict, stats: dict,
                   rng: np.random.Generator, all_person_alphas: np.ndarray,
                   person_to_idx: dict):
    """Simulate solo competition: 4 prelim heats → top 100 each → finals.

    Alpha draws come from the shared all_person_alphas array (drawn once in main).
    Beta (puzzle) and noise are drawn independently per round.
    """
    mu = stats["mu_fixed"]
    sigma = params["sigma"]["mean"]
    nu = params["nu"]["mean"]
    w_clean = params["w_clean"]["mean"]
    k_sigma = params["k_sigma"]["mean"]
    sigma_beta = params["sigma_beta"]["mean"]
    log_w = stats["log_w"]
    pieces = tournament["solo"]["pieces"]
    time_limit = tournament["solo"]["time_limit"]
    advance_n = tournament["solo"]["advance_n"]
    final_time_limit = tournament["solo"]["final_time_limit"]
    final_pieces = tournament["solo"]["final_pieces"]

    pc_corr = compute_piece_correction(pieces, log_w)
    final_pc_corr = compute_piece_correction(final_pieces, log_w)

    # Build heat index mapping into the global person_to_idx
    heat_indices = {}  # heat_id -> list of global person indices
    for heat_id, names in tournament["solo"]["heats"].items():
        heat_indices[heat_id] = [person_to_idx[name] for name in names]

    heat_results = {}
    all_advancers = [[] for _ in range(N_MC)]

    for heat_id, names in tournament["solo"]["heats"].items():
        idx = heat_indices[heat_id]
        n = len(names)

        # Use pre-drawn alphas from global array
        alphas = all_person_alphas[:, idx]  # (N_MC, n)
        betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
        noise = sigma * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, n))

        log_times = mu + alphas + betas + pc_corr + noise
        times_sec = np.power(10, log_times / 1000.0)
        dnf = times_sec > time_limit

        rank_times = np.where(dnf, np.inf, times_sec)
        ranks = rank_times.argsort(axis=1).argsort(axis=1) + 1
        advances = (ranks <= advance_n) & (~dnf)

        heat_result = []
        for i, name in enumerate(names):
            model_name, pdata = matches.get(name, (None, None))
            am, astd = get_alpha_params(name, matches, stats, TOURNAMENT_YEAR)
            adv_prob = advances[:, i].mean()
            dnf_prob = dnf[:, i].mean()
            median_rank = float(np.median(ranks[:, i]))
            heat_result.append({
                "name": name,
                "model_name": model_name,
                "alpha_mean": round(am, 1),
                "alpha_std": round(astd, 1),
                "is_cold_start": pdata is None,
                "n_obs": pdata["n_solo"] if pdata else 0,
                "advance_prob": round(float(adv_prob), 4),
                "dnf_prob": round(float(dnf_prob), 4),
                "median_prelim_rank": round(median_rank, 1),
            })

        for trial in range(N_MC):
            for i, name in enumerate(names):
                if advances[trial, i]:
                    all_advancers[trial].append((name, heat_id))

        heat_results[heat_id] = sorted(heat_result, key=lambda x: -x["advance_prob"])

    # --- Finals (reuse same alpha draws from global array) ---
    all_finalist_names = set()
    for trial_advancers in all_advancers:
        for name, heat in trial_advancers:
            all_finalist_names.add(name)

    finalist_names = sorted(all_finalist_names)
    finalist_global_idx = [person_to_idx[n] for n in finalist_names]
    name_to_fidx = {n: i for i, n in enumerate(finalist_names)}
    n_finalists = len(finalist_names)

    participates = np.zeros((N_MC, n_finalists), dtype=bool)
    for trial, trial_advancers in enumerate(all_advancers):
        for name, heat in trial_advancers:
            participates[trial, name_to_fidx[name]] = True

    # Reuse SAME alpha draws (same person, same tournament, shared across divisions)
    f_alphas = all_person_alphas[:, finalist_global_idx]  # (N_MC, n_finalists)
    # NEW beta (different puzzle) and noise (different day-of performance)
    f_betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
    f_noise = sigma * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, n_finalists))

    f_log_times = mu + f_alphas + f_betas + final_pc_corr + f_noise
    f_times_sec = np.power(10, f_log_times / 1000.0)
    f_dnf = f_times_sec > final_time_limit

    f_rank_times = np.where(participates & ~f_dnf, f_times_sec, np.inf)
    f_ranks = f_rank_times.argsort(axis=1).argsort(axis=1) + 1

    finals_result = []
    winning_times = []

    for i, name in enumerate(finalist_names):
        part_mask = participates[:, i]
        n_part = part_mask.sum()
        if n_part == 0:
            continue

        ranks_when_part = f_ranks[part_mask, i]

        win_prob = (f_ranks[:, i] == 1).sum() / N_MC
        top3_prob = (f_ranks[:, i] <= 3).sum() / N_MC
        top10_prob = (f_ranks[:, i] <= 10).sum() / N_MC
        advance_prob = float(n_part) / N_MC
        dnf_finals_prob = float((part_mask & f_dnf[:, i]).sum()) / N_MC

        p1 = float((f_ranks[:, i] == 1).sum()) / N_MC
        p2 = float((f_ranks[:, i] == 2).sum()) / N_MC
        p3 = float((f_ranks[:, i] == 3).sum()) / N_MC

        median_rank = float(np.median(ranks_when_part))

        heat = None
        for h, hnames in tournament["solo"]["heats"].items():
            if name in hnames:
                heat = h
                break

        model_name, pdata = matches.get(name, (None, None))
        finals_result.append({
            "name": name,
            "model_name": model_name,
            "heat": heat,
            "is_cold_start": pdata is None,
            "n_obs": pdata["n_solo"] if pdata else 0,
            "advance_prob": round(advance_prob, 4),
            "win_prob": round(float(win_prob), 4),
            "top3_prob": round(float(top3_prob), 4),
            "top10_prob": round(float(top10_prob), 4),
            "medal_probs": [round(p1, 4), round(p2, 4), round(p3, 4)],
            "dnf_finals_prob": round(dnf_finals_prob, 4),
            "median_final_rank": round(median_rank, 1),
        })

    winner_mask = f_ranks == 1
    for trial in range(N_MC):
        winner_idx = np.where(winner_mask[trial] & participates[trial])[0]
        if len(winner_idx) > 0:
            winning_times.append(float(f_times_sec[trial, winner_idx[0]]))

    winning_times = np.array(winning_times)
    wt_quantiles = {
        "q05": round(float(np.percentile(winning_times, 5)), 1),
        "q25": round(float(np.percentile(winning_times, 25)), 1),
        "q50": round(float(np.percentile(winning_times, 50)), 1),
        "q75": round(float(np.percentile(winning_times, 75)), 1),
        "q95": round(float(np.percentile(winning_times, 95)), 1),
    }

    finals_result.sort(key=lambda x: -x["win_prob"])

    return {
        "heats": heat_results,
        "finals": finals_result,
        "winning_time_quantiles": wt_quantiles,
    }


def logsumexp_alphas(alphas: np.ndarray) -> np.ndarray:
    """Logsumexp for team parallel alpha: alpha_parallel = -mB * log(sum exp(-alpha / mB)).

    alphas: shape (N_MC, K) where K = team size
    Returns: shape (N_MC,)
    """
    neg_rates = -alphas / MB_SCALE  # (N_MC, K)
    max_nr = neg_rates.max(axis=1, keepdims=True)
    sum_exp = np.exp(neg_rates - max_nr).sum(axis=1)
    return -MB_SCALE * (max_nr[:, 0] + np.log(sum_exp))


def simulate_pairs(tournament: dict, matches: dict, params: dict, stats: dict,
                    rng: np.random.Generator, all_person_alphas: np.ndarray,
                    person_to_idx: dict):
    """Simulate pairs competition: 2 prelim heats → top 100 each → finals.

    Alpha draws come from the shared all_person_alphas array (drawn once in main).
    """
    mu = stats["mu_fixed"]
    sigma_team = params["sigma_team"]["mean"]
    nu = params["nu"]["mean"]
    w_clean = params["w_clean"]["mean"]
    k_sigma = params["k_sigma"]["mean"]
    sigma_beta = params["sigma_beta"]["mean"]
    log_w = stats["log_w"]
    s = params["s"]["mean"]
    eta_2 = params["eta_2"]["mean"]

    pieces = tournament["pairs"]["pieces"]
    time_limit = tournament["pairs"]["time_limit"]
    advance_n = tournament["pairs"]["advance_n"]
    final_time_limit = tournament["pairs"]["final_time_limit"]
    final_pieces = tournament["pairs"]["final_pieces"]

    pc_corr = compute_piece_correction(pieces, log_w)
    final_pc_corr = compute_piece_correction(final_pieces, log_w)
    amdahl_corr = MB_SCALE * math.log(1 + s)  # K=2

    # Build pair member index mapping into global person_to_idx
    pair_member_indices = {}  # (heat, pair_idx) -> (idx1, idx2)
    for heat_id, pairs in tournament["pairs"]["heats"].items():
        for pi, pair in enumerate(pairs):
            pair_member_indices[(heat_id, pi)] = (person_to_idx[pair[0]], person_to_idx[pair[1]])

    heat_results = {}
    all_advancers = [[] for _ in range(N_MC)]

    for heat_id, pairs in tournament["pairs"]["heats"].items():
        n = len(pairs)

        # Build pair alpha_eff from pre-drawn person alphas
        alpha_eff = np.zeros((N_MC, n))
        for pi in range(n):
            i1, i2 = pair_member_indices[(heat_id, pi)]
            pair_a = np.stack([all_person_alphas[:, i1], all_person_alphas[:, i2]], axis=1)  # (N_MC, 2)
            alpha_eff[:, pi] = logsumexp_alphas(pair_a) + amdahl_corr + eta_2

        betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
        noise = sigma_team * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, n))

        log_times = mu + alpha_eff + betas + pc_corr + noise
        times_sec = np.power(10, log_times / 1000.0)
        dnf = times_sec > time_limit

        rank_times = np.where(dnf, np.inf, times_sec)
        ranks = rank_times.argsort(axis=1).argsort(axis=1) + 1
        advances = (ranks <= advance_n) & (~dnf)

        heat_result = []
        for i, pair in enumerate(pairs):
            model_n1, p1 = matches.get(pair[0], (None, None))
            model_n2, p2 = matches.get(pair[1], (None, None))
            i1, i2 = pair_member_indices[(heat_id, i)]
            heat_result.append({
                "members": pair,
                "model_names": [model_n1, model_n2],
                "is_cold_start": [p1 is None, p2 is None],
                "alpha_means": [
                    round(get_alpha_params(pair[0], matches, stats, TOURNAMENT_YEAR)[0], 1),
                    round(get_alpha_params(pair[1], matches, stats, TOURNAMENT_YEAR)[0], 1),
                ],
                "advance_prob": round(float(advances[:, i].mean()), 4),
                "dnf_prob": round(float(dnf[:, i].mean()), 4),
                "median_prelim_rank": round(float(np.median(ranks[:, i])), 1),
            })

        for trial in range(N_MC):
            for i, pair in enumerate(pairs):
                if advances[trial, i]:
                    all_advancers[trial].append((i, pair, heat_id))

        heat_results[heat_id] = sorted(heat_result, key=lambda x: -x["advance_prob"])

    # --- Finals (reuse same person alphas) ---
    all_finalist_pairs = {}
    for trial_advancers in all_advancers:
        for idx, pair, heat in trial_advancers:
            key = (heat, idx)
            if key not in all_finalist_pairs:
                all_finalist_pairs[key] = pair

    finalist_keys = sorted(all_finalist_pairs.keys())
    finalist_pairs = [all_finalist_pairs[k] for k in finalist_keys]
    key_to_fidx = {k: i for i, k in enumerate(finalist_keys)}
    nf = len(finalist_pairs)

    participates = np.zeros((N_MC, nf), dtype=bool)
    for trial, trial_advancers in enumerate(all_advancers):
        for idx, pair, heat in trial_advancers:
            participates[trial, key_to_fidx[(heat, idx)]] = True

    # Reuse SAME person alphas for finals
    f_alpha_eff = np.zeros((N_MC, nf))
    for fi, key in enumerate(finalist_keys):
        i1, i2 = pair_member_indices[key]
        pair_a = np.stack([all_person_alphas[:, i1], all_person_alphas[:, i2]], axis=1)
        f_alpha_eff[:, fi] = logsumexp_alphas(pair_a) + amdahl_corr + eta_2

    f_betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
    f_noise = sigma_team * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, nf))

    f_log_times = mu + f_alpha_eff + f_betas + final_pc_corr + f_noise
    f_times_sec = np.power(10, f_log_times / 1000.0)
    f_dnf = f_times_sec > final_time_limit

    f_rank_times = np.where(participates & ~f_dnf, f_times_sec, np.inf)
    f_ranks = f_rank_times.argsort(axis=1).argsort(axis=1) + 1

    finals_result = []
    winning_times = []

    for i, pair in enumerate(finalist_pairs):
        part_mask = participates[:, i]
        n_part = part_mask.sum()
        if n_part == 0:
            continue

        heat_id = finalist_keys[i][0]
        model_n1, p1 = matches.get(pair[0], (None, None))
        model_n2, p2 = matches.get(pair[1], (None, None))

        win_prob = float((f_ranks[:, i] == 1).sum()) / N_MC
        top3_prob = float((f_ranks[:, i] <= 3).sum()) / N_MC
        top10_prob = float((f_ranks[:, i] <= 10).sum()) / N_MC
        advance_prob = float(n_part) / N_MC

        finals_result.append({
            "members": pair,
            "model_names": [model_n1, model_n2],
            "heat": heat_id,
            "is_cold_start": [p1 is None, p2 is None],
            "advance_prob": round(advance_prob, 4),
            "win_prob": round(win_prob, 4),
            "top3_prob": round(top3_prob, 4),
            "top10_prob": round(top10_prob, 4),
            "medal_probs": [
                round(float((f_ranks[:, i] == 1).sum()) / N_MC, 4),
                round(float((f_ranks[:, i] == 2).sum()) / N_MC, 4),
                round(float((f_ranks[:, i] == 3).sum()) / N_MC, 4),
            ],
            "median_final_rank": round(float(np.median(f_ranks[part_mask, i])), 1),
        })

    winner_mask = f_ranks == 1
    for trial in range(N_MC):
        winner_idx = np.where(winner_mask[trial] & participates[trial])[0]
        if len(winner_idx) > 0:
            winning_times.append(float(f_times_sec[trial, winner_idx[0]]))

    winning_times = np.array(winning_times) if winning_times else np.array([0.0])
    wt_quantiles = {
        "q05": round(float(np.percentile(winning_times, 5)), 1),
        "q50": round(float(np.percentile(winning_times, 50)), 1),
        "q95": round(float(np.percentile(winning_times, 95)), 1),
    }

    finals_result.sort(key=lambda x: -x["win_prob"])

    return {
        "heats": heat_results,
        "finals": finals_result,
        "winning_time_quantiles": wt_quantiles,
    }


def simulate_teams(tournament: dict, matches: dict, params: dict, stats: dict,
                    rng: np.random.Generator, all_person_alphas: np.ndarray,
                    person_to_idx: dict):
    """Simulate teams competition: 2 prelim heats → top 50 each → finals.

    Alpha draws come from the shared all_person_alphas array (drawn once in main).
    Prelim: 500pc + 1000pc consecutive (times summed in seconds).
    Final: two 1000pc puzzles consecutive.
    """
    mu = stats["mu_fixed"]
    sigma_team = params["sigma_team"]["mean"]
    nu = params["nu"]["mean"]
    w_clean = params["w_clean"]["mean"]
    k_sigma = params["k_sigma"]["mean"]
    sigma_beta = params["sigma_beta"]["mean"]
    log_w = stats["log_w"]
    s = params["s"]["mean"]
    eta_4 = params["eta_4plus"]["mean"]

    prelim_puzzles = tournament["teams"]["prelim_puzzles"]
    final_puzzles = tournament["teams"]["final_puzzles"]
    prelim_limit = tournament["teams"]["prelim_time_limit"]
    final_limit = tournament["teams"]["final_time_limit"]
    advance_n = tournament["teams"]["advance_n"]

    amdahl_corr = MB_SCALE * math.log(1 + s * 3)  # K=4
    prelim_pc_corrs = [compute_piece_correction(p["pieces"], log_w) for p in prelim_puzzles]
    final_pc_corrs = [compute_piece_correction(p["pieces"], log_w) for p in final_puzzles]

    # Build team member index mapping into global person_to_idx
    team_member_indices = {}  # (heat, team_idx) -> [idx0, idx1, idx2, idx3]
    for heat_id, teams in tournament["teams"]["heats"].items():
        for ti, team in enumerate(teams):
            team_member_indices[(heat_id, ti)] = [person_to_idx[m] for m in team["members"]]

    heat_results = {}
    all_advancers = [[] for _ in range(N_MC)]

    for heat_id, teams in tournament["teams"]["heats"].items():
        n = len(teams)

        # Build team alpha_eff from pre-drawn person alphas via logsumexp
        alpha_eff = np.zeros((N_MC, n))
        for i in range(n):
            midx = team_member_indices[(heat_id, i)]
            member_alphas = np.stack([all_person_alphas[:, j] for j in midx], axis=1)  # (N_MC, 4)
            alpha_eff[:, i] = logsumexp_alphas(member_alphas) + amdahl_corr + eta_4

        # Simulate consecutive puzzles, sum times in seconds
        total_times = np.zeros((N_MC, n))
        for pz_idx, pc_corr in enumerate(prelim_pc_corrs):
            betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
            noise = sigma_team * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, n))
            log_times = mu + alpha_eff + betas + pc_corr + noise
            times_sec = np.power(10, log_times / 1000.0)
            total_times += times_sec

        dnf = total_times > prelim_limit
        rank_times = np.where(dnf, np.inf, total_times)
        ranks = rank_times.argsort(axis=1).argsort(axis=1) + 1
        advances = (ranks <= advance_n) & (~dnf)

        heat_result = []
        for i, team in enumerate(teams):
            member_info = []
            for j, member in enumerate(team["members"]):
                mn, pd = matches.get(member, (None, None))
                member_info.append({
                    "name": member,
                    "model_name": mn,
                    "is_cold_start": pd is None,
                })
            adv_prob = advances[:, i].mean()
            dnf_prob = dnf[:, i].mean()
            median_rank = float(np.median(ranks[:, i]))
            heat_result.append({
                "team_name": team["name"],
                "members": member_info,
                "advance_prob": round(float(adv_prob), 4),
                "dnf_prob": round(float(dnf_prob), 4),
                "median_prelim_rank": round(median_rank, 1),
            })

        for trial in range(N_MC):
            for i, team in enumerate(teams):
                if advances[trial, i]:
                    all_advancers[trial].append((i, team, heat_id))

        heat_results[heat_id] = sorted(heat_result, key=lambda x: -x["advance_prob"])

    # --- Finals ---
    all_finalist_teams = {}
    for trial_advancers in all_advancers:
        for idx, team, heat in trial_advancers:
            key = (heat, idx)
            if key not in all_finalist_teams:
                all_finalist_teams[key] = team

    finalist_keys = sorted(all_finalist_teams.keys())
    finalist_teams = [all_finalist_teams[k] for k in finalist_keys]
    key_to_idx = {k: i for i, k in enumerate(finalist_keys)}
    nf = len(finalist_teams)

    # Participation mask
    participates = np.zeros((N_MC, nf), dtype=bool)
    for trial, trial_advancers in enumerate(all_advancers):
        for idx, team, heat in trial_advancers:
            fi = key_to_idx[(heat, idx)]
            participates[trial, fi] = True

    # Reuse SAME person alphas for finals (shared across divisions and rounds)
    f_alpha_eff = np.zeros((N_MC, nf))
    for fi, key in enumerate(finalist_keys):
        midx = team_member_indices[key]
        member_alphas = np.stack([all_person_alphas[:, j] for j in midx], axis=1)  # (N_MC, 4)
        f_alpha_eff[:, fi] = logsumexp_alphas(member_alphas) + amdahl_corr + eta_4

    # Two consecutive puzzles
    f_total_times = np.zeros((N_MC, nf))
    for pz_idx, pc_corr in enumerate(final_pc_corrs):
        f_betas = rng.normal(0, sigma_beta, size=(N_MC, 1))
        f_noise = sigma_team * nst_noise(rng, w_clean, nu, k_sigma, size=(N_MC, nf))
        f_log_times = mu + f_alpha_eff + f_betas + pc_corr + f_noise
        f_times_sec = np.power(10, f_log_times / 1000.0)
        f_total_times += f_times_sec

    f_dnf = f_total_times > final_limit
    f_rank_times = np.where(participates & ~f_dnf, f_total_times, np.inf)
    f_ranks = f_rank_times.argsort(axis=1).argsort(axis=1) + 1

    finals_result = []
    winning_times = []

    for i, team in enumerate(finalist_teams):
        part_mask = participates[:, i]
        n_part = part_mask.sum()
        if n_part == 0:
            continue

        heat_id = finalist_keys[i][0]
        member_info = []
        for j, member in enumerate(team["members"]):
            mn, pd = matches.get(member, (None, None))
            member_info.append({
                "name": member,
                "model_name": mn,
                "is_cold_start": pd is None,
            })

        win_prob = float((f_ranks[:, i] == 1).sum()) / N_MC
        top3_prob = float((f_ranks[:, i] <= 3).sum()) / N_MC
        top10_prob = float((f_ranks[:, i] <= 10).sum()) / N_MC
        advance_prob = float(n_part) / N_MC

        finals_result.append({
            "team_name": team["name"],
            "members": member_info,
            "heat": heat_id,
            "advance_prob": round(advance_prob, 4),
            "win_prob": round(win_prob, 4),
            "top3_prob": round(top3_prob, 4),
            "top10_prob": round(top10_prob, 4),
            "medal_probs": [
                round(float((f_ranks[:, i] == 1).sum()) / N_MC, 4),
                round(float((f_ranks[:, i] == 2).sum()) / N_MC, 4),
                round(float((f_ranks[:, i] == 3).sum()) / N_MC, 4),
            ],
            "median_final_rank": round(float(np.median(f_ranks[part_mask, i])), 1),
        })

    # Winning times
    winner_mask = f_ranks == 1
    for trial in range(N_MC):
        winner_idx = np.where(winner_mask[trial] & participates[trial])[0]
        if len(winner_idx) > 0:
            winning_times.append(float(f_total_times[trial, winner_idx[0]]))

    winning_times = np.array(winning_times) if winning_times else np.array([0.0])
    wt_quantiles = {
        "q05": round(float(np.percentile(winning_times, 5)), 1),
        "q50": round(float(np.percentile(winning_times, 50)), 1),
        "q95": round(float(np.percentile(winning_times, 95)), 1),
    }

    finals_result.sort(key=lambda x: -x["win_prob"])

    return {
        "heats": heat_results,
        "finals": finals_result,
        "winning_time_quantiles": wt_quantiles,
    }


def nst_noise(rng: np.random.Generator, w: float, nu: float, k_sigma: float, size) -> np.ndarray:
    """Sample from NST (Normal + Student-t mixture) noise.

    Returns standardized samples: w*Normal(0,1) + (1-w)*StudentT(nu)*k_sigma.
    Caller multiplies by sigma or sigma_team.
    """
    mask = rng.random(size) < w
    normal = rng.normal(0, 1, size=size)
    student = rng.standard_t(nu, size=size) * k_sigma
    return np.where(mask, normal, student)


def main():
    t0 = time.time()
    print("Loading data...")
    model, tournament = load_data()

    stats = model["stats"]
    params = model["scalar_params"]["joint"]
    pmap = build_puzzler_map(model)

    print("Matching names...")
    matches = match_all_names(tournament, pmap, model["puzzlers"])

    # Count cold starts per division
    for div in ["solo", "pairs", "teams"]:
        names = set()
        if div == "solo":
            for h in tournament[div]["heats"].values():
                names.update(h)
        elif div == "pairs":
            for h in tournament[div]["heats"].values():
                for pair in h:
                    names.update(pair)
        else:
            for h in tournament[div]["heats"].values():
                for team in h:
                    names.update(team["members"])
        cold = sum(1 for n in names if matches[n][1] is None)
        print(f"  {div}: {len(names)} unique, {cold} cold-start")

    rng = np.random.default_rng(42)

    # --- Draw ALL person alphas once, shared across solo/pairs/teams ---
    # Collect every unique person name across all divisions
    all_person_names_set = set()
    for h in tournament["solo"]["heats"].values():
        all_person_names_set.update(h)
    for h in tournament["pairs"]["heats"].values():
        for pair in h:
            all_person_names_set.update(pair)
    for h in tournament["teams"]["heats"].values():
        for team in h:
            all_person_names_set.update(team["members"])

    all_person_names = sorted(all_person_names_set)
    person_to_idx = {name: i for i, name in enumerate(all_person_names)}
    n_all = len(all_person_names)
    print(f"  Drawing alphas for {n_all} unique persons across all divisions")

    alpha_means = np.array([get_alpha_params(n, matches, stats, TOURNAMENT_YEAR)[0] for n in all_person_names])
    alpha_stds = np.array([get_alpha_params(n, matches, stats, TOURNAMENT_YEAR)[1] for n in all_person_names])
    all_person_alphas = rng.normal(alpha_means[None, :], alpha_stds[None, :], size=(N_MC, n_all))

    print("Simulating solo...")
    solo_results = simulate_solo(tournament, matches, params, stats, rng, all_person_alphas, person_to_idx)

    print("Simulating pairs...")
    pairs_results = simulate_pairs(tournament, matches, params, stats, rng, all_person_alphas, person_to_idx)

    print("Simulating teams...")
    teams_results = simulate_teams(tournament, matches, params, stats, rng, all_person_alphas, person_to_idx)

    # Name matching summary
    all_names = set()
    for h in tournament["solo"]["heats"].values():
        all_names.update(h)
    for h in tournament["pairs"]["heats"].values():
        for pair in h:
            all_names.update(pair)
    for h in tournament["teams"]["heats"].values():
        for team in h:
            all_names.update(team["members"])
    n_matched = sum(1 for n in all_names if matches[n][1] is not None)
    n_cold = sum(1 for n in all_names if matches[n][1] is None)

    output = {
        "meta": {
            "n_mc": N_MC,
            "date": "2026-03-09",
            "tournament_date": "2026-03-27",
            "tournament_year": TOURNAMENT_YEAR,
        },
        "params": {
            "mu": stats["mu_fixed"],
            "sigma": params["sigma"]["mean"],
            "nu": params["nu"]["mean"],
            "w_clean": params["w_clean"]["mean"],
            "k_sigma": params["k_sigma"]["mean"],
            "sigma_alpha": params["sigma_alpha"]["mean"],
            "sigma_beta": params["sigma_beta"]["mean"],
            "log_w": stats["log_w"],
            "s": params["s"]["mean"],
            "eta_2": params["eta_2"]["mean"],
            "eta_4plus": params["eta_4plus"]["mean"],
            "sigma_team": params["sigma_team"]["mean"],
            "ranking_year": stats["ranking_year"],
            "year_center": stats["year_center"],
        },
        "name_matching": {
            "matched": n_matched,
            "cold_start": n_cold,
            "total": n_matched + n_cold,
        },
        "solo": solo_results,
        "pairs": pairs_results,
        "teams": teams_results,
    }

    # Sanitize for JSON (no inf/nan)
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    output = sanitize(output)
    OUTPUT.write_text(json.dumps(output, indent=1))
    elapsed = time.time() - t0
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB) in {elapsed:.1f}s")

    # Quick sanity checks
    print(f"\nSanity checks:")
    print(f"  Solo finals: {len(solo_results['finals'])} unique finalists")
    print(f"  Solo win probs sum: {sum(f['win_prob'] for f in solo_results['finals']):.3f}")
    print(f"  Solo top winner: {solo_results['finals'][0]['name']} ({solo_results['finals'][0]['win_prob']:.3f})")
    print(f"  Pairs finals: {len(pairs_results['finals'])} unique finalist pairs")
    print(f"  Pairs win probs sum: {sum(f['win_prob'] for f in pairs_results['finals']):.3f}")
    print(f"  Teams finals: {len(teams_results['finals'])} unique finalist teams")
    print(f"  Teams win probs sum: {sum(f['win_prob'] for f in teams_results['finals']):.3f}")
    print(f"  Solo winning time (median): {solo_results['winning_time_quantiles']['q50']:.0f}s")


if __name__ == "__main__":
    main()
