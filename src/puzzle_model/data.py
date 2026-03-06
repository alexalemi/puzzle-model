"""Data loading, cleaning, and encoding for puzzle model."""

import calendar

import numpy as np
import pandas as pd


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


def encode_indices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map puzzler names and puzzle IDs to contiguous integer indices.

    Returns (df_with_indices, puzzler_lookup, puzzle_lookup) where lookups
    map names/IDs to integer indices.
    """
    df = df.copy()

    puzzlers = df["competitor_name"].unique()
    puzzler_lookup = {name: i for i, name in enumerate(puzzlers)}
    df["puzzler_idx"] = df["competitor_name"].map(puzzler_lookup)

    puzzles = df["puzzle_id"].unique()
    puzzle_lookup = {pid: i for i, pid in enumerate(puzzles)}
    df["puzzle_idx"] = df["puzzle_id"].map(puzzle_lookup)

    return df, puzzler_lookup, puzzle_lookup


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

    If mu_fixed is None, computes it as the mean of log_time (use for training).
    Pass the training mu_fixed explicitly for test data to keep them aligned.
    """
    log_time = np.array(df["log_time"])
    if mu_fixed is None:
        mu_fixed = float(np.mean(log_time))
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
