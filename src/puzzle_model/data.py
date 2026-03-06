"""Data loading, cleaning, and encoding for puzzle model."""

import numpy as np
import pandas as pd


def load_solo_completed(path: str = "data/processed/combined_results.csv") -> pd.DataFrame:
    """Load and filter to solo, completed results with valid times and piece counts."""
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
    df = df[mask].copy()
    df["log_time"] = np.log(df["time_seconds"])
    df["puzzle_pieces"] = df["puzzle_pieces"].astype(int)
    return df.reset_index(drop=True)


def create_puzzle_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique puzzle identifier from event, name, and piece count."""
    df = df.copy()
    df["puzzle_id"] = (
        df["event_id"].astype(str)
        + "_"
        + df["puzzle_name"].astype(str)
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

    if "source" in df.columns:
        sources = sorted(df["source"].unique())
        source_lookup = {s: i for i, s in enumerate(sources)}
        df["source_idx"] = df["source"].map(source_lookup)

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


def prepare_model_data(df: pd.DataFrame) -> dict:
    """Convert a DataFrame into the dict of arrays needed by NumPyro models."""
    data = {
        "puzzler_idx": np.array(df["puzzler_idx"]),
        "puzzle_idx": np.array(df["puzzle_idx"]),
        "log_time": np.array(df["log_time"]),
        "pieces": np.array(df["puzzle_pieces"]),
        "n_puzzlers": df["puzzler_idx"].max() + 1,
        "n_puzzles": df["puzzle_idx"].max() + 1,
    }
    if "source_idx" in df.columns:
        data["source_idx"] = np.array(df["source_idx"])
        data["n_sources"] = df["source_idx"].max() + 1
    return data
