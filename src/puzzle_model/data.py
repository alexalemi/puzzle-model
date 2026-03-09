"""Data loading, cleaning, and encoding for puzzle model."""

import calendar
import re

import numpy as np
import pandas as pd

MAX_TEAM_SIZE = 8

# COVID-era virtual events where participants had days/weeks to finish and
# reported total elapsed time, not focused solving time.  These are not
# comparable to timed speed-puzzling results and heavily distort the model.
EXCLUDED_EVENTS = {"sp_1", "sp_3", "sp_5"}

# Fixed mu: 1 hour in milliBels. Using a physical constant rather than the
# empirical mean makes parameters stable across refits and interpretable as
# "offset from a 1-hour solve".
MU_ONE_HOUR = 1000.0 * np.log10(3600)  # ≈ 3556.3 mB


def to_fractional_year(dt) -> float:
    """Convert a datetime/Timestamp to year + (day_of_year - 1) / days_in_year."""
    dt = pd.Timestamp(dt)
    days_in_year = 366 if calendar.isleap(dt.year) else 365
    return dt.year + (dt.day_of_year - 1) / days_in_year


def load_solo_completed(
    path: str = "data/processed/combined_results.csv",
    source: str | None = None,
) -> pd.DataFrame:
    """Load and filter to solo, completed results with valid times and piece counts.

    If source is given, filter to that source only.
    Preserves finished_date and first_attempt columns.
    """
    df = pd.read_csv(path, low_memory=False)
    mask = (
        (df["division"] == "solo")
        & (df["completed"] == True)
        & df["time_seconds"].notna()
        & df["puzzle_pieces"].notna()
        & df["competitor_name"].notna()
        & (df["time_seconds"] > 0)
        & (df["puzzle_pieces"] > 0)
    )
    if source is not None:
        mask = mask & (df["source"] == source)
    mask = mask & ~df["event_id"].isin(EXCLUDED_EVENTS)
    df = df[mask].copy()
    df["log_time"] = 1000.0 * np.log10(df["time_seconds"])
    df["puzzle_pieces"] = df["puzzle_pieces"].astype(int)
    if "finished_date" in df.columns:
        df["finished_date"] = pd.to_datetime(df["finished_date"], format="mixed", errors="coerce")
    if "first_attempt" in df.columns:
        df["first_attempt"] = df["first_attempt"].fillna(True).astype(bool)
    # Compute fractional year: use finished_date where available, else year + 0.5
    has_date = df.get("finished_date") is not None and df["finished_date"].notna().any()
    if has_date:
        df["year_frac"] = df.apply(
            lambda r: to_fractional_year(r["finished_date"])
            if pd.notna(r.get("finished_date"))
            else r["year"] + 0.5,
            axis=1,
        )
    else:
        df["year_frac"] = df["year"].astype(float) + 0.5
    return df.reset_index(drop=True)


def load_completed(
    path: str = "data/processed/combined_results.csv",
    divisions: tuple[str, ...] = ("solo",),
    source: str | None = None,
) -> pd.DataFrame:
    """Load and filter to completed results with valid times and piece counts.

    Args:
        divisions: Tuple of divisions to include (e.g. ("solo",) or ("solo", "duo", "group")).
        source: If given, filter to that source only.
    """
    df = pd.read_csv(path, low_memory=False)
    mask = (
        df["division"].isin(divisions)
        & (df["completed"] == True)
        & df["time_seconds"].notna()
        & df["puzzle_pieces"].notna()
        & df["competitor_name"].notna()
        & (df["time_seconds"] > 0)
        & (df["puzzle_pieces"] > 0)
    )
    if source is not None:
        mask = mask & (df["source"] == source)
    mask = mask & ~df["event_id"].isin(EXCLUDED_EVENTS)
    df = df[mask].copy()
    df["log_time"] = 1000.0 * np.log10(df["time_seconds"])
    df["puzzle_pieces"] = df["puzzle_pieces"].astype(int)
    if "finished_date" in df.columns:
        df["finished_date"] = pd.to_datetime(df["finished_date"], format="mixed", errors="coerce")
    if "first_attempt" in df.columns:
        df["first_attempt"] = df["first_attempt"].fillna(True).astype(bool)
    # Compute fractional year
    has_date = df.get("finished_date") is not None and df["finished_date"].notna().any()
    if has_date:
        df["year_frac"] = df.apply(
            lambda r: to_fractional_year(r["finished_date"])
            if pd.notna(r.get("finished_date"))
            else r["year"] + 0.5,
            axis=1,
        )
    else:
        df["year_frac"] = df["year"].astype(float) + 0.5
    # Drop teams larger than MAX_TEAM_SIZE
    if "team_members" in df.columns:
        too_large = df["team_members"].fillna("").apply(
            lambda tm: len([m for m in tm.split(";") if m.strip()]) > MAX_TEAM_SIZE if tm else False
        )
        n_dropped = too_large.sum()
        if n_dropped:
            print(f"  Dropped {n_dropped} obs with team_size > {MAX_TEAM_SIZE}")
        df = df[~too_large]
    return df.reset_index(drop=True)


def parse_team_members(tm_string: str) -> list[tuple[str, str]]:
    """Parse team_members string into list of (name, player_id) tuples.

    Format: "Name1:uuid1;Name2:;Name3:uuid3"
    Returns: [("Name1", "uuid1"), ("Name2", ""), ("Name3", "uuid3")]
    """
    if not tm_string or pd.isna(tm_string):
        return []
    members = []
    for part in tm_string.split(";"):
        part = part.strip()
        if not part:
            continue
        # Split on last colon (name may contain colons, though unlikely)
        idx = part.rfind(":")
        if idx == -1:
            members.append((part, ""))
        else:
            name = part[:idx]
            pid = part[idx + 1:]
            members.append((name, pid))
    return members


def _msp_competitor_name(display_name: str, player_id: str, player_links: dict[str, str] | None = None) -> str:
    """Build the canonical competitor_name for an MSP player.

    Uses the same format as combine.py: "Display Name (msp:short_id)".
    If the player_id is in player_links, uses the linked SP name instead.
    """
    if player_links and player_id in player_links:
        return player_links[player_id]
    if player_id:
        return f"{display_name} (msp:{player_id[:8]})"
    # SP "Last, First" names have no player_id — return as-is for puzzler matching
    if "," in display_name:
        return display_name.strip()
    return ""


UNKNOWN_PUZZLER = "__unknown__"


def encode_indices(
    df: pd.DataFrame,
    player_links: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Map puzzler names and puzzle IDs to contiguous integer indices.

    If the DataFrame contains team_members, also registers all registered
    team members in the puzzler lookup (plus __unknown__ for unregistered).

    Returns (df_with_indices, puzzler_lookup, puzzle_lookup) where lookups
    map names/IDs to integer indices.
    """
    df = df.copy()

    # Start with all competitor_names from the data
    all_names = set(df["competitor_name"].unique())

    # Also register team members so they share the same index space
    if "team_members" in df.columns:
        for tm in df["team_members"].dropna():
            for name, pid in parse_team_members(tm):
                comp_name = _msp_competitor_name(name, pid, player_links)
                if comp_name:
                    all_names.add(comp_name)
        all_names.add(UNKNOWN_PUZZLER)

    puzzlers = sorted(all_names)
    puzzler_lookup = {name: i for i, name in enumerate(puzzlers)}
    df["puzzler_idx"] = df["competitor_name"].map(puzzler_lookup)

    puzzles = df["puzzle_id"].unique()
    puzzle_lookup = {pid: i for i, pid in enumerate(puzzles)}
    df["puzzle_idx"] = df["puzzle_id"].map(puzzle_lookup)

    return df, puzzler_lookup, puzzle_lookup


def build_team_arrays(
    df: pd.DataFrame,
    puzzler_lookup: dict[str, int],
    player_links: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Build padded team arrays for the joint solo+team model.

    For solo obs: team_member_idx = [puzzler_idx, 0, ...], mask = [True, False, ...], size = 1
    For duo/group: parse team_members, look up each member's index, pad, mask.

    Returns dict with:
        team_member_idx: (n_obs, MAX_TEAM_SIZE) int array
        team_mask: (n_obs, MAX_TEAM_SIZE) bool array
        team_size: (n_obs,) int array
    """
    n = len(df)
    member_idx = np.zeros((n, MAX_TEAM_SIZE), dtype=np.int32)
    mask = np.zeros((n, MAX_TEAM_SIZE), dtype=bool)
    sizes = np.ones(n, dtype=np.int32)  # default solo = 1

    unknown_idx = puzzler_lookup.get(UNKNOWN_PUZZLER, 0)
    has_tm = "team_members" in df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        tm = row.get("team_members", "") if has_tm else ""
        if not tm or pd.isna(tm):
            # Solo observation
            member_idx[i, 0] = row["puzzler_idx"]
            mask[i, 0] = True
        else:
            members = parse_team_members(tm)
            size = min(len(members), MAX_TEAM_SIZE)
            sizes[i] = size
            for j, (name, pid) in enumerate(members[:MAX_TEAM_SIZE]):
                comp_name = _msp_competitor_name(name, pid, player_links)
                idx = puzzler_lookup.get(comp_name, unknown_idx)
                member_idx[i, j] = idx
                mask[i, j] = True

    return {
        "team_member_idx": member_idx,
        "team_mask": mask,
        "team_size": sizes,
    }


def create_puzzle_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique puzzle identifier from name and piece count.

    Same puzzle (name + pieces) across different events/sources shares one ID,
    enabling cross-source calibration through shared puzzles.
    """
    df = df.copy()
    df["puzzle_id"] = (
        df["puzzle_name"].astype(str)
        + "_"
        + df["puzzle_pieces"].astype(str)
    )
    return df


def train_test_split(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split, stratified by puzzler to ensure representation."""
    rng = np.random.default_rng(seed)

    test_mask = np.zeros(len(df), dtype=bool)
    for _, group in df.groupby("puzzler_idx"):
        n_test = max(1, int(len(group) * test_frac)) if len(group) >= 3 else 0
        if n_test > 0:
            test_indices = rng.choice(group.index, size=n_test, replace=False)
            test_mask[test_indices] = True

    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def add_repeat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add repeat-solve features for each (competitor, puzzle) group.

    Adds columns:
      solve_number: 1-indexed (1 = first attempt)
      days_since_last: days since previous solve of same puzzle (0 for first)
      days_since_first: days since first solve of same puzzle (0 for first)

    Rows without finished_date get solve_number=1, days=0.
    """
    df = df.copy()
    df["solve_number"] = 1
    df["days_since_last"] = 0.0
    df["days_since_first"] = 0.0

    has_date = df["finished_date"].notna() if "finished_date" in df.columns else pd.Series(False, index=df.index)
    if not has_date.any():
        return df

    # Only compute for rows with dates, grouped by (competitor, puzzle)
    dated = df[has_date].copy()
    for (comp, puz), group in dated.groupby(["competitor_name", "puzzle_id"]):
        if len(group) <= 1:
            continue
        sorted_idx = group.sort_values("finished_date").index
        dates = df.loc[sorted_idx, "finished_date"]
        df.loc[sorted_idx, "solve_number"] = range(1, len(sorted_idx) + 1)
        first_date = dates.iloc[0]
        df.loc[sorted_idx, "days_since_first"] = (dates - first_date).dt.days.astype(float)
        days_diff = dates.diff().dt.days.fillna(0).astype(float)
        df.loc[sorted_idx, "days_since_last"] = days_diff.values

    # Reconcile with first_attempt flag: if MSP says it's a repeat but we
    # only see one solve, bump to solve_number=2 (they solved it at least once before).
    # Our observed sequence takes priority when we have multiple solves.
    if "first_attempt" in df.columns:
        unseen_repeat = (~df["first_attempt"]) & (df["solve_number"] == 1)
        df.loc[unseen_repeat, "solve_number"] = 2

    return df


def prepare_model_data(df: pd.DataFrame, mu_fixed: float | None = None) -> dict:
    """Convert a DataFrame into the dict of arrays needed by NumPyro models.

    If mu_fixed is None, uses MU_ONE_HOUR (1 hour in mB) as a stable,
    interpretable reference point. Pass mu_fixed explicitly to override.
    """
    log_time = np.array(df["log_time"])
    if mu_fixed is None:
        mu_fixed = MU_ONE_HOUR
    data = {
        "puzzler_idx": np.array(df["puzzler_idx"]),
        "puzzle_idx": np.array(df["puzzle_idx"]),
        "log_time": log_time,
        "pieces": np.array(df["puzzle_pieces"]),
        "n_puzzlers": df["puzzler_idx"].max() + 1,
        "n_puzzles": df["puzzle_idx"].max() + 1,
        "mu_fixed": mu_fixed,
    }
    if "year_frac" in df.columns:
        data["year"] = np.array(df["year_frac"], dtype=np.float32)
    elif "year" in df.columns:
        data["year"] = np.array(df["year"], dtype=np.float32)
    if "first_attempt" in df.columns:
        data["first_attempt"] = np.array(df["first_attempt"], dtype=bool)
    if "solve_number" in df.columns:
        data["solve_number"] = np.array(df["solve_number"], dtype=np.float32)
    if "days_since_last" in df.columns:
        data["days_since_last"] = np.array(df["days_since_last"], dtype=np.float32)
    if "days_since_first" in df.columns:
        data["days_since_first"] = np.array(df["days_since_first"], dtype=np.float32)
    return data
