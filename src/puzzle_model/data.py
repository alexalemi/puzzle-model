"""Data loading, cleaning, and encoding for puzzle model."""

import numpy as np
import pandas as pd


def load_solo_completed(
    path: str = "data/processed/combined_results.csv",
    source: str | None = None,
) -> pd.DataFrame:
    """Load and filter to solo, completed results with valid times and piece counts.

    If source is given, filter to that source only.
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
    if "year" in df.columns:
        data["year"] = np.array(df["year"], dtype=np.float32)
    return data
