#!/usr/bin/env python3
"""Fit joint solo+team model and compare against solo-only baseline.

Outputs explorer_team_data.json for the team explorer page.
"""

import json
import math
import random
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import Predictive, log_likelihood

from puzzle_model.data import (
    load_completed,
    load_solo_completed,
    create_puzzle_id,
    encode_indices,
    train_test_split,
    prepare_model_data,
    add_repeat_features,
    build_team_arrays,
    parse_team_members,
    to_fractional_year,
    _msp_competitor_name,
    MAX_TEAM_SIZE,
    UNKNOWN_PUZZLER,
)
from puzzle_model.model import MODELS, N_REF, PHYS_BASIS_NAMES, YEAR_CENTER
from puzzle_model.inference import run_svi
from puzzle_model.evaluate import evaluate_predictions


def load_player_links() -> dict[str, str]:
    """Load manual MSP player_id -> SP competitor_name links."""
    links_path = Path(__file__).resolve().parent.parent / "data" / "mappings" / "player_links.csv"
    if not links_path.exists():
        return {}
    links = pd.read_csv(links_path)
    return dict(zip(links["msp_player_id"], links["sp_competitor_name"]))


def compute_waic(pointwise_log_lik: np.ndarray) -> dict:
    n_samples = pointwise_log_lik.shape[0]
    lppd_i = np.logaddexp.reduce(pointwise_log_lik, axis=0) - np.log(n_samples)
    lppd = np.sum(lppd_i)
    p_waic_i = np.var(pointwise_log_lik, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)
    waic = -2 * (lppd - p_waic)
    se = 2 * np.sqrt(len(lppd_i) * np.var(-2 * (lppd_i - p_waic_i)))
    return {"waic": float(waic), "lppd": float(lppd), "p_waic": float(p_waic), "se": float(se)}


def main():
    player_links = load_player_links()
    print(f"Loaded {len(player_links)} player links")

    # ── Load all data (solo + duo + group) ──
    print("\nLoading all data...")
    df_all = load_completed(divisions=("solo", "duo", "group"))
    df_all = create_puzzle_id(df_all)
    df_all = add_repeat_features(df_all)
    df_all, puzzler_lookup, puzzle_lookup = encode_indices(df_all, player_links=player_links)

    n_solo = (df_all["division"] == "solo").sum()
    n_duo = (df_all["division"] == "duo").sum()
    n_group = (df_all["division"] == "group").sum()
    print(f"Total: {len(df_all):,} obs (solo={n_solo:,}, duo={n_duo:,}, group={n_group:,})")
    print(f"Puzzlers: {len(puzzler_lookup):,}, Puzzles: {len(puzzle_lookup):,}")

    # Build team arrays for ALL observations (solo gets trivial arrays)
    team_arrays = build_team_arrays(df_all, puzzler_lookup, player_links=player_links)

    # ── Team data stats ──
    team_sizes = team_arrays["team_size"]
    team_size_dist = {}
    for s in range(1, MAX_TEAM_SIZE + 1):
        c = int((team_sizes == s).sum())
        if c > 0:
            team_size_dist[s] = c

    # Count registered vs unregistered team members
    n_registered = 0
    n_unregistered = 0
    registered_ids = set()
    team_mask = df_all["division"].isin(["duo", "group"])
    for tm_str in df_all.loc[team_mask, "team_members"].dropna():
        for name, pid in parse_team_members(tm_str):
            if pid:
                n_registered += 1
                registered_ids.add(pid)
            else:
                n_unregistered += 1

    # How many registered team members also have solo data?
    solo_names = set(df_all.loc[df_all["division"] == "solo", "competitor_name"].unique())
    registered_with_solo = 0
    for pid in registered_ids:
        comp_name = _msp_competitor_name("", pid, player_links)
        # Try matching by UUID prefix
        matches = [n for n in solo_names if f"msp:{pid[:8]}" in n or n == comp_name]
        if matches:
            registered_with_solo += 1
        elif player_links and pid in player_links and player_links[pid] in solo_names:
            registered_with_solo += 1

    print(f"\nTeam member stats:")
    print(f"  Registered: {len(registered_ids)} unique ({n_registered} total slots)")
    print(f"  Unregistered: {n_unregistered} slots")
    print(f"  Registered with solo data: {registered_with_solo}/{len(registered_ids)}")

    # ── Train/test split on solo data only ──
    solo_df = df_all[df_all["division"] == "solo"].reset_index(drop=True)
    solo_train, solo_test = train_test_split(solo_df)

    # Build model data
    mu_fixed = float(np.mean(solo_train["log_time"]))  # mu from solo training data
    print(f"\nmu_fixed = {mu_fixed:.3f} mB (from solo train)")

    solo_train_data = prepare_model_data(solo_train, mu_fixed=mu_fixed)
    solo_test_data = prepare_model_data(solo_test, mu_fixed=mu_fixed)
    # Ensure plate sizes cover full puzzler/puzzle universe
    n_puzzlers = len(puzzler_lookup)
    n_puzzles = len(puzzle_lookup)
    for d in (solo_train_data, solo_test_data):
        d["n_puzzlers"] = n_puzzlers
        d["n_puzzles"] = n_puzzles

    # Solo train + team arrays (trivial for solo)
    solo_train_team = build_team_arrays(solo_train, puzzler_lookup, player_links=player_links)

    # ── Split team data into train/test (80/20 random) ──
    team_df = df_all[df_all["division"].isin(["duo", "group"])].reset_index(drop=True)
    rng_team = np.random.default_rng(42)
    team_test_mask = rng_team.random(len(team_df)) < 0.2
    team_train = team_df[~team_test_mask].reset_index(drop=True)
    team_test = team_df[team_test_mask].reset_index(drop=True)

    duo_test = team_test[team_test["division"] == "duo"].reset_index(drop=True)
    group_test = team_test[team_test["division"] == "group"].reset_index(drop=True)

    # ── Joint training set: solo train + team train ──
    joint_train_df = pd.concat([solo_train, team_train], ignore_index=True)
    joint_train_data = prepare_model_data(joint_train_df, mu_fixed=mu_fixed)
    joint_train_data["n_puzzlers"] = n_puzzlers
    joint_train_data["n_puzzles"] = n_puzzles

    # Build team arrays for joint training set
    joint_team_arrays = build_team_arrays(joint_train_df, puzzler_lookup, player_links=player_links)
    joint_train_data.update(joint_team_arrays)

    # Solo test team arrays (trivial, for model_team evaluation)
    solo_test_team = build_team_arrays(solo_test, puzzler_lookup, player_links=player_links)

    # Team test data + arrays (for held-out team evaluation)
    team_test_data = prepare_model_data(team_test, mu_fixed=mu_fixed)
    team_test_data["n_puzzlers"] = n_puzzlers
    team_test_data["n_puzzles"] = n_puzzles
    team_test_arrays = build_team_arrays(team_test, puzzler_lookup, player_links=player_links)

    duo_test_data = prepare_model_data(duo_test, mu_fixed=mu_fixed)
    duo_test_data["n_puzzlers"] = n_puzzlers
    duo_test_data["n_puzzles"] = n_puzzles
    duo_test_arrays = build_team_arrays(duo_test, puzzler_lookup, player_links=player_links)

    group_test_data = prepare_model_data(group_test, mu_fixed=mu_fixed)
    group_test_data["n_puzzlers"] = n_puzzlers
    group_test_data["n_puzzles"] = n_puzzles
    group_test_arrays = build_team_arrays(group_test, puzzler_lookup, player_links=player_links)

    print(f"\nSolo train: {len(solo_train):,}, Solo test: {len(solo_test):,}")
    print(f"Team train: {len(team_train):,}, Team test: {len(team_test):,} (duo={len(duo_test):,}, group={len(group_test):,})")
    print(f"Joint train: {len(joint_train_df):,}")

    # ── Fit baseline: model_2r on solo train only ──
    print(f"\n{'='*60}")
    print("Fitting model_2r (solo-only baseline)...")
    model_2r = MODELS["model_2r"]
    guide_solo, result_solo = run_svi(model_2r, solo_train_data, num_steps=5000, lr=0.005, seed=0)

    # Evaluate on solo test
    posterior_solo = guide_solo.sample_posterior(
        jax.random.PRNGKey(1), result_solo.params, sample_shape=(500,)
    )
    te_no_obs = {k: v for k, v in solo_test_data.items() if k != "log_time"}
    pred_solo = Predictive(model_2r, guide=guide_solo, params=result_solo.params, num_samples=200)
    pred_solo_test = pred_solo(jax.random.PRNGKey(1), **te_no_obs)
    metrics_solo = evaluate_predictions(
        np.array(pred_solo_test["log_time"]),
        np.array(solo_test_data["log_time"]),
    )

    # Test log-likelihood
    ll_solo = log_likelihood(model_2r, posterior_solo, **solo_test_data)
    ll_solo_matrix = np.array(ll_solo["log_time"])
    lppd_solo_i = np.logaddexp.reduce(ll_solo_matrix, axis=0) - np.log(ll_solo_matrix.shape[0])
    mean_lpd_solo = float(np.mean(lppd_solo_i))
    print(f"  Solo baseline test mean LPD: {mean_lpd_solo:.4f}")
    print(f"  Metrics: {metrics_solo}")

    # ── Fit joint: model_team on solo train + team ──
    print(f"\n{'='*60}")
    print("Fitting model_team (joint solo+team)...")
    model_team = MODELS["model_team"]
    guide_joint, result_joint = run_svi(model_team, joint_train_data, num_steps=5000, lr=0.005, seed=0)

    # Evaluate on SAME solo test set
    posterior_joint = guide_joint.sample_posterior(
        jax.random.PRNGKey(1), result_joint.params, sample_shape=(500,)
    )
    te_joint = {k: v for k, v in solo_test_data.items() if k != "log_time"}
    te_joint.update(solo_test_team)  # Add trivial team arrays
    pred_joint = Predictive(model_team, guide=guide_joint, params=result_joint.params, num_samples=200)
    pred_joint_test = pred_joint(jax.random.PRNGKey(1), **te_joint)
    metrics_joint = evaluate_predictions(
        np.array(pred_joint_test["log_time"]),
        np.array(solo_test_data["log_time"]),
    )

    # Test log-likelihood for joint model on solo test
    te_joint_with_obs = dict(solo_test_data)
    te_joint_with_obs.update(solo_test_team)
    ll_joint = log_likelihood(model_team, posterior_joint, **te_joint_with_obs)
    ll_joint_matrix = np.array(ll_joint["log_time"])
    lppd_joint_i = np.logaddexp.reduce(ll_joint_matrix, axis=0) - np.log(ll_joint_matrix.shape[0])
    mean_lpd_joint = float(np.mean(lppd_joint_i))
    print(f"  Joint solo test mean LPD: {mean_lpd_joint:.4f}")
    print(f"  Metrics: {metrics_joint}")

    # ── Held-out team evaluation ──
    def eval_team_subset(label, data_dict, arrays_dict):
        """Evaluate joint model on a held-out team subset."""
        te = {k: v for k, v in data_dict.items() if k != "log_time"}
        te.update(arrays_dict)
        pred = Predictive(model_team, guide=guide_joint, params=result_joint.params, num_samples=200)
        pred_samples = pred(jax.random.PRNGKey(3), **te)
        m = evaluate_predictions(np.array(pred_samples["log_time"]), np.array(data_dict["log_time"]))
        # Log-likelihood
        te_obs = dict(data_dict)
        te_obs.update(arrays_dict)
        ll = log_likelihood(model_team, posterior_joint, **te_obs)
        ll_mat = np.array(ll["log_time"])
        lppd_i = np.logaddexp.reduce(ll_mat, axis=0) - np.log(ll_mat.shape[0])
        mlpd = float(np.mean(lppd_i))
        print(f"  {label}: mean LPD={mlpd:.4f}, RMSE={m['rmse_log']:.2f}, MAE={m['mae_log']:.2f}, "
              f"Cov90={m['coverage_90']:.4f}, Cov50={m['coverage_50']:.4f}")
        return mlpd, m

    print(f"\n{'='*60}")
    print("Held-out team evaluation (joint model):")
    mlpd_team_all, metrics_team_all = eval_team_subset("All team", team_test_data, team_test_arrays)
    mlpd_duo, metrics_duo = eval_team_subset("Duo only", duo_test_data, duo_test_arrays)
    if len(group_test) > 0:
        mlpd_group, metrics_group = eval_team_subset("Group only", group_test_data, group_test_arrays)
    else:
        mlpd_group, metrics_group = None, None

    # ── Comparison ──
    print(f"\n{'='*60}")
    print(f"{'Subset':<25} {'N':>7} {'Mean LPD':>10} {'RMSE':>10} {'MAE':>10} {'Cov90':>8} {'Cov50':>8}")
    print("-" * 82)
    rows = [
        ("Solo test (solo model)", len(solo_test), mean_lpd_solo, metrics_solo),
        ("Solo test (joint model)", len(solo_test), mean_lpd_joint, metrics_joint),
        ("Team test (joint model)", len(team_test), mlpd_team_all, metrics_team_all),
        ("Duo test (joint model)", len(duo_test), mlpd_duo, metrics_duo),
    ]
    if mlpd_group is not None:
        rows.append(("Group test (joint model)", len(group_test), mlpd_group, metrics_group))
    for label, n, lpd, m in rows:
        print(f"{label:<25} {n:>7,} {lpd:>10.4f} {m['rmse_log']:>10.2f} {m['mae_log']:>10.2f} "
              f"{m['coverage_90']:>8.4f} {m['coverage_50']:>8.4f}")
    delta_lpd = mean_lpd_joint - mean_lpd_solo
    print(f"\nDelta solo mean LPD (joint - solo): {delta_lpd:+.4f}")

    # ── Team parameters: Amdahl's s + per-bucket eta + sigma_team ──
    s_vals = np.array(posterior_joint["s"])
    eta_vals = np.array(posterior_joint["eta_team"])  # (n_samples, 3)
    sigma_team_vals = np.array(posterior_joint["sigma_team"])
    bucket_labels = ["K=2", "K=3", "K≥4"]

    team_params = {
        "s": {
            "mean": round(float(np.mean(s_vals)), 4),
            "std": round(float(np.std(s_vals)), 4),
            "q05": round(float(np.percentile(s_vals, 5)), 4),
            "q95": round(float(np.percentile(s_vals, 95)), 4),
        },
    }
    for bi, label in enumerate(bucket_labels):
        vals = eta_vals[:, bi]
        team_params[label] = {
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "q05": round(float(np.percentile(vals, 5)), 3),
            "q95": round(float(np.percentile(vals, 95)), 3),
        }

    # sigma_team stats
    team_params["sigma_team"] = {
        "mean": round(float(np.mean(sigma_team_vals)), 3),
        "std": round(float(np.std(sigma_team_vals)), 3),
        "q05": round(float(np.percentile(sigma_team_vals, 5)), 3),
        "q95": round(float(np.percentile(sigma_team_vals, 95)), 3),
    }

    # Implied total team correction (Amdahl + eta) for each K
    mB_scale_val = 1000.0 / np.log(10.0)
    print(f"\nTeam parameters:")
    print(f"  Serial fraction s: {team_params['s']['mean']:.3f} ± {team_params['s']['std']:.3f} "
          f"[{team_params['s']['q05']:.3f}, {team_params['s']['q95']:.3f}]")
    print(f"  sigma_team: {team_params['sigma_team']['mean']:.1f} ± {team_params['sigma_team']['std']:.1f} mB "
          f"[{team_params['sigma_team']['q05']:.1f}, {team_params['sigma_team']['q95']:.1f}]")
    print(f"  Per-bucket eta corrections:")
    for bi, label in enumerate(bucket_labels):
        v = team_params[label]
        print(f"    {label}: {v['mean']:+.1f} ± {v['std']:.1f} mB [{v['q05']:+.1f}, {v['q95']:+.1f}]")
    print(f"  Implied Amdahl correction (mB * log(1 + s*(K-1))):")
    for k in [2, 3, 4, 6, 8]:
        amdahl = mB_scale_val * np.log(1 + np.mean(s_vals) * (k - 1))
        print(f"    K={k}: +{amdahl:.1f} mB")

    # ── Alpha shift: compare solo-only vs joint alphas ──
    alpha_solo = np.mean(np.array(posterior_solo["alpha"]), axis=0)  # (n_puzzlers,)
    alpha_joint = np.mean(np.array(posterior_joint["alpha"]), axis=0)

    inv_puzzler = {v: k for k, v in puzzler_lookup.items()}
    obs_solo = solo_df.groupby("puzzler_idx").size()
    obs_team = team_df.groupby("puzzler_idx").size() if len(team_df) > 0 else pd.Series(dtype=int)

    alpha_shift = []
    for i in range(min(len(alpha_solo), len(alpha_joint))):
        name = inv_puzzler.get(i, "")
        if name == UNKNOWN_PUZZLER:
            continue
        ns = int(obs_solo.get(i, 0))
        nt = int(obs_team.get(i, 0))
        if ns == 0 and nt == 0:
            continue
        alpha_shift.append({
            "name": name,
            "alpha_solo": round(float(alpha_solo[i]), 3),
            "alpha_joint": round(float(alpha_joint[i]), 3),
            "shift": round(float(alpha_joint[i] - alpha_solo[i]), 3),
            "n_solo": ns,
            "n_team": nt,
        })
    # Sort by absolute shift
    alpha_shift.sort(key=lambda x: abs(x["shift"]), reverse=True)

    # ── Scalar params comparison ──
    scalar_names = ["sigma", "nu", "sigma_alpha", "sigma_beta", "delta_0", "sigma_delta", "gamma"]
    scalar_comparison = {"solo_only": {}, "joint": {}}
    for name in scalar_names:
        for label, samples in [("solo_only", posterior_solo), ("joint", posterior_joint)]:
            vals = np.array(samples[name])
            scalar_comparison[label][name] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
    # Add log_w
    for k in range(4):
        for label, samples in [("solo_only", posterior_solo), ("joint", posterior_joint)]:
            vals = np.array(samples["log_w"][:, k])
            scalar_comparison[label][f"log_w_{k}"] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
    # Add team-only params to joint column
    scalar_comparison["joint"]["s"] = {
        "mean": round(float(np.mean(s_vals)), 4),
        "std": round(float(np.std(s_vals)), 4),
    }
    scalar_comparison["joint"]["sigma_team"] = {
        "mean": round(float(np.mean(sigma_team_vals)), 4),
        "std": round(float(np.std(sigma_team_vals)), 4),
    }
    for bi, label in enumerate(["eta_2", "eta_3", "eta_4plus"]):
        vals = np.array(posterior_joint["eta_team"][:, bi])
        scalar_comparison["joint"][label] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }

    # ── Predicted vs observed scatter (solo model, logsumexp with velocity) ──
    alpha_arr = np.mean(np.array(posterior_solo["alpha"]), axis=0)
    beta_arr = np.mean(np.array(posterior_solo["beta"]), axis=0)
    log_w_arr = np.mean(np.array(posterior_solo["log_w"]), axis=0)
    delta_0_val = float(np.mean(np.array(posterior_solo["delta_0"])))
    delta_arr = np.mean(np.array(posterior_solo["delta"]), axis=0)
    gamma_val = float(np.mean(np.array(posterior_solo["gamma"])))

    pieces_all = df_all["puzzle_pieces"].values.astype(float)
    N_arr = pieces_all
    g_all = np.column_stack([np.sqrt(N_arr), N_arr, N_arr * np.log(N_arr), N_arr ** 2])
    g_ref = np.array([np.sqrt(N_REF), N_REF, N_REF * np.log(N_REF), N_REF ** 2])
    w_arr = np.exp(log_w_arr)
    mB_scale_np = 1000.0 / np.log(10.0)
    pc_all = mB_scale_np * (np.log(g_all @ w_arr) - np.log(g_ref @ w_arr))

    t_all = df_all["year_frac"].values.astype(float) - YEAR_CENTER
    sn_all = df_all["solve_number"].values.astype(float)
    rep_all = gamma_val * np.log(sn_all)

    ta = build_team_arrays(df_all, puzzler_lookup, player_links=player_links)
    alpha_gathered = alpha_arr[ta["team_member_idx"]]  # (n_obs, MAX_TEAM_SIZE)
    delta_gathered = delta_arr[ta["team_member_idx"]]  # (n_obs, MAX_TEAM_SIZE)

    # Per-member velocity-adjusted alpha, then logsumexp
    alpha_adjusted = alpha_gathered + (delta_0_val + delta_gathered) * t_all[:, None]
    rates = np.where(ta["team_mask"], np.exp(-alpha_adjusted / mB_scale_np), 0.0)
    total_rate = np.sum(rates, axis=1)
    alpha_parallel_all = -mB_scale_np * np.log(total_rate)

    pred_all = mu_fixed + alpha_parallel_all + beta_arr[df_all["puzzle_idx"].values] + pc_all + rep_all
    obs_all = df_all["log_time"].values
    ts_all = ta["team_size"]

    # Build scatter data: all team obs + subsampled solo
    rng_scatter = np.random.default_rng(99)
    scatter_points = []
    for k in range(2, MAX_TEAM_SIZE + 1):
        sel = ts_all == k
        for p, o in zip(pred_all[sel], obs_all[sel]):
            scatter_points.append({"p": round(float(p), 1), "o": round(float(o), 1), "k": int(k)})
    solo_mask = ts_all == 1
    solo_idx = np.where(solo_mask)[0]
    solo_sample = rng_scatter.choice(solo_idx, size=min(5000, len(solo_idx)), replace=False)
    for i in solo_sample:
        scatter_points.append({"p": round(float(pred_all[i]), 1), "o": round(float(obs_all[i]), 1), "k": 1})

    # Per-K residual summaries
    residual_by_k = {}
    for k in sorted(set(ts_all)):
        sel = ts_all == k
        resid = obs_all[sel] - pred_all[sel]
        residual_by_k[str(k)] = {
            "mean": round(float(np.mean(resid)), 1),
            "std": round(float(np.std(resid)), 1),
            "n": int(sel.sum()),
        }
    print(f"\nSolo-model residuals by team size:")
    for k, v in residual_by_k.items():
        print(f"  K={k}: {v['mean']:+.1f} ± {v['std']:.1f} mB (n={v['n']:,})")

    # ── Build MSP URL lookups ──
    root = Path(__file__).resolve().parent.parent
    msp_dir = root / "data" / "raw" / "myspeedpuzzling"
    MSP_BASE = "https://myspeedpuzzling.com/en"

    msp_player_url = {}
    players_csv = msp_dir / "players.csv"
    if players_csv.exists():
        players_df = pd.read_csv(players_csv, usecols=["player_id", "name"])
        for _, row in players_df.iterrows():
            pid = row["player_id"]
            display = f"{row['name']} (msp:{pid[:8]})"
            msp_player_url[display] = f"{MSP_BASE}/player-profile/{pid}"
        links_csv = root / "data" / "mappings" / "player_links.csv"
        if links_csv.exists():
            links_df = pd.read_csv(links_csv, usecols=["msp_player_id", "sp_competitor_name"])
            for _, row in links_df.iterrows():
                msp_player_url[row["sp_competitor_name"]] = f"{MSP_BASE}/player-profile/{row['msp_player_id']}"

    msp_puzzle_url = {}
    puzzles_csv = msp_dir / "puzzles.csv"
    if puzzles_csv.exists():
        puzzles_df = pd.read_csv(puzzles_csv, usecols=["puzzle_id"])
        prefix_to_puzzle_uuid = {pid[:8]: pid for pid in puzzles_df["puzzle_id"]}
        msp_rows = df_all[df_all["source"] == "myspeedpuzzling"]
        for _, row in msp_rows.drop_duplicates("puzzle_id").iterrows():
            eid = str(row.get("event_id", ""))
            m = re.match(r"msp_([0-9a-f]+)", eid)
            if m and m.group(1) in prefix_to_puzzle_uuid:
                msp_puzzle_url[row["puzzle_id"]] = f"{MSP_BASE}/puzzle/{prefix_to_puzzle_uuid[m.group(1)]}"
    print(f"MSP links: {len(msp_player_url)} puzzlers, {len(msp_puzzle_url)} puzzles")

    # ── Build image lookup (use MSP image URLs directly) ──
    # Iterate all rows (not drop_duplicates) so cross-source puzzles find their MSP image
    img_lookup = {}
    if puzzles_csv.exists():
        img_df = pd.read_csv(puzzles_csv, usecols=["puzzle_id", "image_url"])
        prefix_to_img_url = {pid[:8]: url for pid, url in zip(img_df["puzzle_id"], img_df["image_url"]) if pd.notna(url)}
        for _, row in df_all.iterrows():
            pid = row["puzzle_id"]
            if pid in img_lookup:
                continue
            eid = str(row.get("event_id", ""))
            m = re.match(r"msp_([0-9a-f]+)", eid)
            if m and m.group(1) in prefix_to_img_url:
                img_lookup[pid] = prefix_to_img_url[m.group(1)]

    # ── Rankings from joint model posterior ──
    from datetime import date, datetime
    from scipy import stats as sp_stats

    params_1 = posterior_joint  # Use joint model for rankings
    RANKING_YEAR = to_fractional_year(datetime.now())
    ELO_SCALE = 1

    alpha_samples = np.array(params_1["alpha"])
    delta_samples = np.array(params_1["delta"])
    delta_0_samples = np.array(params_1["delta_0"])
    beta_samples_raw = np.array(params_1["beta"])
    log_w_samples = np.array(params_1["log_w"])
    log_w_mean = np.mean(log_w_samples, axis=0)
    log_w_vals = [round(float(v), 4) for v in log_w_mean]

    sigma_val = round(float(np.mean(np.array(params_1["sigma"]))), 3)
    delta_0_mean = round(float(np.mean(delta_0_samples)), 4)
    sigma_delta_val = round(float(np.mean(np.array(params_1["sigma_delta"]))), 4)

    log_times = np.array(df_all["log_time"])
    piece_counts = sorted(df_all["puzzle_pieces"].unique())

    # Piece-count stats (solo only for EDA)
    solo_data = df_all[df_all["division"] == "solo"]

    stats = {
        "n_records": len(df_all),
        "n_solo_total": int(n_solo),
        "n_solo_train": len(solo_train),
        "n_solo_test": len(solo_test),
        "n_duo": int(n_duo),
        "n_group": int(n_group),
        "n_team_total": int(n_duo + n_group),
        "n_puzzlers": n_puzzlers,
        "n_puzzles": n_puzzles,
        "n_registered_team_members": len(registered_ids),
        "n_unregistered_slots": n_unregistered,
        "n_registered_with_solo": registered_with_solo,
        "pct_registered_with_solo": round(100 * registered_with_solo / max(len(registered_ids), 1), 1),
        "piece_counts": [int(p) for p in piece_counts],
        "log_time_mean": round(float(np.mean(log_times)), 3),
        "log_time_std": round(float(np.std(log_times)), 3),
        "mu": round(mu_fixed, 3),
        "mu_fixed": round(mu_fixed, 3),
        "log_w": log_w_vals,
        "N_REF": N_REF,
        "phys_basis_names": PHYS_BASIS_NAMES,
        "sigma": sigma_val,
        "delta_0": delta_0_mean,
        "sigma_delta": sigma_delta_val,
        "year_center": YEAR_CENTER,
        "ranking_year": round(RANKING_YEAR, 4),
        "ranking_date": date.today().isoformat(),
        "sigma_alpha": round(float(np.mean(np.array(params_1["sigma_alpha"]))), 3),
        "sigma_beta": round(float(np.mean(np.array(params_1["sigma_beta"]))), 3),
        "nu": round(float(np.mean(np.array(params_1["nu"]))), 3),
    }

    # Puzzler rankings (projected to RANKING_YEAR)
    dt = RANKING_YEAR - YEAR_CENTER
    projected = alpha_samples + (delta_0_samples[:, None] + delta_samples) * dt
    proj_mean = np.mean(projected, axis=0)
    proj_std = np.std(projected, axis=0)
    proj_upper = proj_mean + 3 * proj_std
    ELO_CENTER = round(1500 + float(np.median(proj_mean)))
    stats["elo_center"] = ELO_CENTER

    alpha_mean_joint = np.mean(alpha_samples, axis=0)
    velocity_samples = -(delta_0_samples[:, None] + delta_samples)
    velocity_mean = np.mean(velocity_samples, axis=0)
    velocity_std = np.std(velocity_samples, axis=0)

    obs_counts_all = df_all.groupby("puzzler_idx").size()
    obs_counts_solo_all = solo_df.groupby("puzzler_idx").size()
    obs_counts_team_all = team_df.groupby("puzzler_idx").size() if len(team_df) > 0 else pd.Series(dtype=int)
    year_range = df_all.groupby("puzzler_idx")["year"].agg(["min", "max"])
    puzzlers_list = []
    for i in np.argsort(proj_upper):
        name = inv_puzzler[i]
        if name == UNKNOWN_PUZZLER:
            continue
        elo = ELO_CENTER - ELO_SCALE * float(proj_mean[i])
        elo_wilson = ELO_CENTER - ELO_SCALE * float(proj_upper[i])
        entry = {
            "name": name,
            "alpha": round(float(alpha_mean_joint[i]), 3),
            "alpha_proj": round(float(proj_mean[i]), 3),
            "std": round(float(proj_std[i]), 3),
            "velocity": round(float(velocity_mean[i]), 4),
            "velocity_std": round(float(velocity_std[i]), 4),
            "elo": round(elo),
            "elo_wilson": round(elo_wilson),
            "n": int(obs_counts_all.get(i, 0)),
            "n_solo": int(obs_counts_solo_all.get(i, 0)),
            "n_team": int(obs_counts_team_all.get(i, 0)),
            "yr_min": int(year_range.loc[i, "min"]) if i in year_range.index and pd.notna(year_range.loc[i, "min"]) else 0,
            "yr_max": int(year_range.loc[i, "max"]) if i in year_range.index and pd.notna(year_range.loc[i, "max"]) else 0,
        }
        if name in msp_player_url:
            entry["msp_url"] = msp_player_url[name]
        puzzlers_list.append(entry)

    # Puzzle rankings
    beta_mean = np.mean(beta_samples_raw, axis=0)
    beta_std = np.std(beta_samples_raw, axis=0)
    beta_lower = beta_mean - 3 * beta_std
    beta_upper = beta_mean + 3 * beta_std
    inv_puzzle = {v: k for k, v in puzzle_lookup.items()}
    puzzle_obs = df_all.groupby("puzzle_idx").size()
    puzzle_pieces_map = df_all.groupby("puzzle_idx")["puzzle_pieces"].first()
    puzzle_sources = df_all.groupby("puzzle_idx")["source"].apply(lambda s: sorted(s.unique().tolist())).to_dict()

    # Piece-count correction for difficulty
    n_puzzles_total = len(beta_mean)
    pc_array = np.array([float(puzzle_pieces_map.get(i, N_REF)) for i in range(n_puzzles_total)])
    g_pz = np.column_stack([np.sqrt(pc_array), pc_array, pc_array * np.log(pc_array), pc_array ** 2])
    g_ref_pz = np.array([np.sqrt(N_REF), N_REF, N_REF * np.log(N_REF), N_REF ** 2])
    w_samp = np.exp(log_w_samples)
    time_pz = g_pz @ w_samp.T
    time_ref_pz = g_ref_pz @ w_samp.T
    mB_scale = 1000.0 / np.log(10.0)
    pc_correction_pz = mB_scale * (np.log(time_pz) - np.log(time_ref_pz[None, :]))
    total_diff_samples = beta_samples_raw + pc_correction_pz.T
    difficulty_mean = np.mean(total_diff_samples, axis=0)
    difficulty_std = np.std(total_diff_samples, axis=0)

    puzzles_list = []
    for i in np.argsort(-beta_lower):
        raw_name = str(inv_puzzle[i])
        display_name = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        p_elo = ELO_CENTER + ELO_SCALE * float(beta_mean[i])
        p_elo_hard = ELO_CENTER + ELO_SCALE * float(beta_lower[i])
        p_elo_easy = ELO_CENTER + ELO_SCALE * float(beta_upper[i])
        entry = {
            "name": display_name,
            "puzzle_id": inv_puzzle[i],
            "beta": round(float(beta_mean[i]), 3),
            "std": round(float(beta_std[i]), 3),
            "difficulty": round(float(difficulty_mean[i]), 3),
            "difficulty_std": round(float(difficulty_std[i]), 3),
            "wilson_hard": round(float(beta_lower[i]), 3),
            "wilson_easy": round(float(beta_upper[i]), 3),
            "elo": round(p_elo),
            "elo_hard": round(p_elo_hard),
            "elo_easy": round(p_elo_easy),
            "pc": int(puzzle_pieces_map.get(i, 0)),
            "n": int(puzzle_obs.get(i, 0)),
            "sources": puzzle_sources.get(i, []),
        }
        if inv_puzzle[i] in img_lookup:
            entry["img"] = img_lookup[inv_puzzle[i]]
        if inv_puzzle[i] in msp_puzzle_url:
            entry["msp_url"] = msp_puzzle_url[inv_puzzle[i]]
        puzzles_list.append(entry)

    # ── EDA distributions ──
    # Scatter (subsample)
    eda_scatter_idx = random.Random(42).sample(range(len(solo_data)), min(1500, len(solo_data)))
    eda_scatter = [
        {"p": int(solo_data.iloc[i]["puzzle_pieces"]),
         "t": round(float(solo_data.iloc[i]["time_seconds"]), 1),
         "lt": round(float(solo_data.iloc[i]["log_time"]), 2)}
        for i in sorted(eda_scatter_idx)
    ]

    # Piece distribution
    pc_counts = solo_data["puzzle_pieces"].value_counts().sort_index()
    piece_dist = [{"pieces": int(p), "count": int(c)} for p, c in pc_counts.items()]

    # Histograms for common piece counts
    common_pcs = [p for p, c in pc_counts.items() if c >= 100][:9]
    histograms = {}
    for pc in common_pcs:
        vals = solo_data[solo_data["puzzle_pieces"] == pc]["log_time"].values
        counts, edges = np.histogram(vals, bins=30)
        histograms[str(int(pc))] = {
            "n": len(vals),
            "counts": counts.tolist(),
            "edges": [round(float(e), 3) for e in edges],
        }

    # Puzzler frequency
    obs_per_puzzler = df_all.groupby("puzzler_idx").size().values
    freq_counts, freq_edges = np.histogram(obs_per_puzzler, bins=50)
    puzzler_freq = {
        "counts": freq_counts.tolist(),
        "edges": [round(float(e), 1) for e in freq_edges],
    }

    # Puzzle difficulty distribution
    puzzle_mB = ELO_CENTER + beta_mean
    beta_hist_counts, beta_hist_edges = np.histogram(puzzle_mB, bins=50)
    puzzle_beta_dist = {
        "counts": beta_hist_counts.tolist(),
        "edges": [round(float(e), 3) for e in beta_hist_edges],
    }
    for label, nmin, nmax in [("n1", 1, 1), ("n2_5", 2, 5), ("n6plus", 6, 999999)]:
        mask = np.array([(nmin <= int(puzzle_obs.get(i, 0)) <= nmax) for i in range(len(beta_mean))])
        if mask.any():
            c, _ = np.histogram(puzzle_mB[mask], bins=beta_hist_edges)
            puzzle_beta_dist[label] = c.tolist()

    # ── Model team deep dive data ──
    # Scalar params with mean/std (from joint model)
    scalar_detail_names = ["mu", "sigma", "nu", "sigma_alpha", "sigma_beta", "delta_0", "sigma_delta", "gamma"]
    scalar_params_detail = {}
    for name in scalar_detail_names:
        if name == "mu":
            scalar_params_detail[name] = {"mean": round(mu_fixed, 4), "std": 0.0}
            continue
        vals = np.array(params_1[name])
        scalar_params_detail[name] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    for k in range(4):
        scalar_params_detail[f"log_w_{k}"] = {
            "mean": round(float(np.mean(log_w_samples[:, k])), 4),
            "std": round(float(np.std(log_w_samples[:, k])), 4),
        }
        w_samples_k = np.exp(log_w_samples[:, k])
        scalar_params_detail[f"w_{k}"] = {
            "mean": round(float(np.mean(w_samples_k)), 4),
            "std": round(float(np.std(w_samples_k)), 4),
        }
    # Team params
    scalar_params_detail["s"] = {
        "mean": round(float(np.mean(s_vals)), 4),
        "std": round(float(np.std(s_vals)), 4),
    }
    scalar_params_detail["sigma_team"] = {
        "mean": round(float(np.mean(sigma_team_vals)), 4),
        "std": round(float(np.std(sigma_team_vals)), 4),
    }
    for bi, elabel in enumerate(["eta_2", "eta_3", "eta_4plus"]):
        vals = np.array(posterior_joint["eta_team"][:, bi])
        scalar_params_detail[elabel] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }

    # Student-t vs Normal comparison
    nu_mean = float(np.mean(np.array(params_1["nu"])))
    sigma_mean = float(np.mean(np.array(params_1["sigma"])))
    x_range = np.linspace(-4 * sigma_mean, 4 * sigma_mean, 200)
    student_pdf = sp_stats.t.pdf(x_range, df=nu_mean, scale=sigma_mean)
    normal_pdf = sp_stats.norm.pdf(x_range, scale=sigma_mean)
    student_t_comparison = {
        "nu": round(nu_mean, 2),
        "sigma": round(sigma_mean, 2),
        "x": [round(float(v), 2) for v in x_range],
        "student_pdf": [round(float(v), 6) for v in student_pdf],
        "normal_pdf": [round(float(v), 6) for v in normal_pdf],
    }

    # Basis correction curve
    pieces_range = np.array([50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 13500])
    w_samples = np.exp(log_w_samples)
    g_curve = np.column_stack([np.sqrt(pieces_range), pieces_range, pieces_range * np.log(pieces_range), pieces_range ** 2])
    g_ref_curve = np.array([np.sqrt(N_REF), N_REF, N_REF * np.log(N_REF), N_REF ** 2])
    time_curve = g_curve @ w_samples.T
    time_ref_curve = g_ref_curve @ w_samples.T
    correction_samples = mB_scale * (np.log(time_curve) - np.log(time_ref_curve[None, :]))
    correction_samples = correction_samples.T
    basis_mean_curve = np.mean(correction_samples, axis=0)
    deciles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    quantile_data = {}
    for q in deciles:
        quantile_data[f"p{q}"] = [round(float(v), 2) for v in np.percentile(correction_samples, q, axis=0)]
    basis_correction = {
        "pieces": pieces_range.tolist(),
        "mean": [round(float(v), 2) for v in basis_mean_curve],
        "quantiles": quantile_data,
    }

    # Per-component mB contribution curves
    basis_components = {}
    for k, bname in enumerate(PHYS_BASIS_NAMES):
        w_k_mean = float(np.mean(w_samples[:, k]))
        g_k = g_curve[:, k]
        g_k_ref = g_ref_curve[k]
        component_mB = mB_scale * (np.log(w_k_mean * g_k) - np.log(w_k_mean * g_k_ref))
        basis_components[bname] = [round(float(v), 2) for v in component_mB]
    basis_components["pieces"] = pieces_range.tolist()

    # Basis fraction
    component_times = w_samples[:, None, :] * g_curve[None, :, :]
    total_time = component_times.sum(axis=2, keepdims=True)
    fractions = component_times / total_time
    basis_fractions = {"pieces": pieces_range.tolist()}
    for k, bname in enumerate(PHYS_BASIS_NAMES):
        frac_k = fractions[:, :, k]
        basis_fractions[bname] = {
            "mean": [round(float(v), 4) for v in np.mean(frac_k, axis=0)],
            "p10": [round(float(v), 4) for v in np.percentile(frac_k, 10, axis=0)],
            "p90": [round(float(v), 4) for v in np.percentile(frac_k, 90, axis=0)],
        }

    # Practice curve
    gamma_samples = np.array(params_1["gamma"])
    solve_numbers = list(range(1, 11))
    log_n = np.log(np.array(solve_numbers))
    practice_curves = gamma_samples[:, None] * log_n[None, :]
    practice_mean = np.mean(practice_curves, axis=0)
    practice_q10 = np.percentile(practice_curves, 10, axis=0)
    practice_q90 = np.percentile(practice_curves, 90, axis=0)
    practice_pct = [round(100 * (10 ** (v / 1000) - 1), 2) for v in practice_mean]
    practice_curve = {
        "solve_numbers": solve_numbers,
        "mean_mB": [round(float(v), 2) for v in practice_mean],
        "q10_mB": [round(float(v), 2) for v in practice_q10],
        "q90_mB": [round(float(v), 2) for v in practice_q90],
        "mean_pct": practice_pct,
    }

    # Repeat statistics
    n_first = df_all["first_attempt"].sum() if "first_attempt" in df_all.columns else len(df_all)
    repeat_stats = {
        "n_total": len(df_all),
        "n_first_attempt": int(n_first),
        "n_repeat": int(len(df_all) - n_first),
        "n_puzzlers_with_repeats": int(df_all[df_all["solve_number"] > 1]["puzzler_idx"].nunique()) if "solve_number" in df_all.columns else 0,
        "max_solve_number": int(df_all["solve_number"].max()) if "solve_number" in df_all.columns else 1,
        "median_solve_number_repeats": int(df_all.loc[df_all["solve_number"] > 1, "solve_number"].median()) if "solve_number" in df_all.columns and (df_all["solve_number"] > 1).any() else None,
    }

    model_team_detail = {
        "scalar_params": scalar_params_detail,
        "student_t_comparison": student_t_comparison,
        "basis_correction": basis_correction,
        "basis_components": basis_components,
        "basis_fractions": basis_fractions,
        "practice_curve": practice_curve,
        "repeat_stats": repeat_stats,
    }

    # ── Build explorer_team_data.json ──
    explorer_data = {
        "stats": stats,
        "comparison": {
            "solo_only": {
                "test_mean_lpd": round(mean_lpd_solo, 4),
                **{k: round(v, 4) for k, v in metrics_solo.items()},
            },
            "joint_solo": {
                "test_mean_lpd": round(mean_lpd_joint, 4),
                **{k: round(v, 4) for k, v in metrics_joint.items()},
            },
            "delta_mean_lpd": round(delta_lpd, 4),
            "team_all": {
                "n": len(team_test),
                "test_mean_lpd": round(mlpd_team_all, 4),
                **{k: round(v, 4) for k, v in metrics_team_all.items()},
            },
            "duo": {
                "n": len(duo_test),
                "test_mean_lpd": round(mlpd_duo, 4),
                **{k: round(v, 4) for k, v in metrics_duo.items()},
            },
            "group": {
                "n": len(group_test),
                "test_mean_lpd": round(mlpd_group, 4) if mlpd_group is not None else None,
                **({k: round(v, 4) for k, v in metrics_group.items()} if metrics_group else {}),
            },
        },
        "team_params": team_params,
        "alpha_shift": alpha_shift[:500],
        "team_size_dist": [{"size": s, "count": c} for s, c in sorted(team_size_dist.items()) if s > 1],
        "scalar_params": scalar_comparison,
        "scatter": scatter_points,
        "residual_by_k": residual_by_k,
        "loss_curves": {
            "solo_only": [round(float(v), 1) for v in result_solo.losses[::10]],
            "joint": [round(float(v), 1) for v in result_joint.losses[::10]],
        },
        "puzzlers": puzzlers_list,
        "puzzles": puzzles_list,
        "model_team_detail": model_team_detail,
        "eda_scatter": eda_scatter,
        "piece_dist": piece_dist,
        "histograms": histograms,
        "puzzler_freq": puzzler_freq,
        "puzzle_beta_dist": puzzle_beta_dist,
    }

    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    explorer_data = sanitize(explorer_data)
    out_path = root / "explorer_team_data.json"
    output = json.dumps(explorer_data, separators=(",", ":"))
    out_path.write_text(output)
    print(f"\nWrote {out_path} ({len(output):,} bytes)")

    # ── Write per-observation file for puzzler deep dive ──
    print("Computing posterior predictive percentiles...")
    full_data = prepare_model_data(df_all, mu_fixed=mu_fixed)
    full_data["n_puzzlers"] = n_puzzlers
    full_data["n_puzzles"] = n_puzzles
    full_team = build_team_arrays(df_all, puzzler_lookup, player_links=player_links)
    full_no_obs = {k: v for k, v in full_data.items() if k != "log_time"}
    full_no_obs.update(full_team)
    pp_predictive = Predictive(model_team, guide=guide_joint, params=result_joint.params, num_samples=500)
    pp_samples = np.array(pp_predictive(jax.random.PRNGKey(42), **full_no_obs)["log_time"])
    actual = np.array(full_data["log_time"])
    pp_pred_mean = np.mean(pp_samples, axis=0)
    percentiles = np.mean(pp_samples <= actual[None, :], axis=0)
    print(f"  Mean percentile: {np.mean(percentiles):.3f} (should be ~0.50)")

    source_abbrev = {"speedpuzzling": "s", "myspeedpuzzling": "m", "mallory": "a", "usajigsaw": "u"}
    obs_by_puzzler = {}
    has_finished_date = "finished_date" in df_all.columns
    for pos, (i, row) in enumerate(df_all.iterrows()):
        name = row["competitor_name"]
        if has_finished_date and pd.notna(row["finished_date"]):
            date_str = row["finished_date"].strftime("%Y-%m-%d")
        else:
            date_str = str(int(row["year"])) if "year" in row.index else None
        div = row.get("division", "solo")
        obs = [
            row["puzzle_id"],
            round(float(row["log_time"])),
            round(float(pp_pred_mean[pos])),
            round(float(percentiles[pos]), 2),
            int(row["puzzle_pieces"]),
            date_str,
            int(row.get("solve_number", 1)),
            source_abbrev.get(row.get("source", ""), "?"),
        ]
        obs_by_puzzler.setdefault(name, []).append(obs)

    obs_data = {
        "_columns": ["puzzle_id", "log_time", "pred_mean", "percentile", "pieces", "date", "solve_number", "source"],
        "_images": img_lookup,
        **obs_by_puzzler,
    }
    obs_data = sanitize(obs_data)
    obs_output = json.dumps(obs_data, separators=(",", ":"))
    obs_path = root / "explorer_team_puzzler_obs.json"
    obs_path.write_text(obs_output)
    print(f"Wrote {obs_path} ({len(obs_output):,} bytes, {len(obs_by_puzzler):,} puzzlers)")


if __name__ == "__main__":
    main()
