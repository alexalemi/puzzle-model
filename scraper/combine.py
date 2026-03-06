"""Combine results from all sources into a single CSV.

Sources:
1. speedpuzzling.com - US competition results (scraped from PDFs)
2. usajigsaw - USA Jigsaw Puzzle Association results
3. myspeedpuzzling.com - global self-reported solving times

Output schema:
    source, event_id, year, division, round, rank, competitor_name, origin,
    time_seconds, completed, pieces_completed, puzzle_pieces, puzzle_brand,
    puzzle_name, time_limit_seconds, first_attempt
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SP_PATH = PROJECT_ROOT / "data" / "processed" / "speedpuzzling_results.csv"
UJ_PATH = PROJECT_ROOT / "data" / "processed" / "usajigsaw_results.csv"
MSP_PUZZLES_PATH = Path.home() / "build" / "myspeedpuzzling.com" / "scraper_output" / "puzzles.csv"
MSP_TIMES_PATH = Path.home() / "build" / "myspeedpuzzling.com" / "scraper_output" / "solving_times.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "combined_results.csv"


def load_myspeedpuzzling() -> pd.DataFrame:
    """Load and normalize myspeedpuzzling.com data to the shared schema.

    Joins solving_times with puzzles catalog to get puzzle metadata,
    then maps columns to match the combined schema.
    """
    puzzles = pd.read_csv(MSP_PUZZLES_PATH)
    times = pd.read_csv(MSP_TIMES_PATH)

    # Join to get puzzle metadata
    df = times.merge(
        puzzles[["puzzle_id", "name", "manufacturer", "pieces_count"]],
        on="puzzle_id",
        how="left",
    )

    # Build a readable but unique competitor name: "Display Name (msp:short_id)"
    # Use first 8 chars of player_id UUID for disambiguation
    df["competitor_name"] = (
        df["player_name"] + " (msp:" + df["player_id"].str[:8] + ")"
    )

    # Extract year from finished_date
    df["year"] = pd.to_datetime(df["finished_date"], errors="coerce").dt.year

    # Use the MSP puzzle_id as the event_id (each puzzle is its own "event")
    # Prefix with msp_ to avoid collisions with SP event IDs
    df["event_id"] = "msp_" + df["puzzle_id"].str[:8]

    return pd.DataFrame({
        "source": "myspeedpuzzling",
        "event_id": df["event_id"],
        "year": df["year"],
        "division": df["category"],
        "round": None,
        "rank": None,
        "competitor_name": df["competitor_name"],
        "origin": df["player_country"],
        "time_seconds": df["time_seconds"],
        "completed": True,
        "pieces_completed": None,
        "puzzle_pieces": df["pieces_count"],
        "puzzle_brand": df["manufacturer"],
        "puzzle_name": df["name"],
        "time_limit_seconds": None,
        "first_attempt": df["first_attempt"],
    })


def combine() -> pd.DataFrame:
    """Load all sources and combine into a single DataFrame."""
    frames = []

    # speedpuzzling.com
    if SP_PATH.exists():
        sp = pd.read_csv(SP_PATH)
        sp["first_attempt"] = True  # competitions are always first-attempt
        frames.append(sp)
        print(f"speedpuzzling: {len(sp)} rows")
    else:
        print(f"WARNING: {SP_PATH} not found, skipping")

    # usajigsaw
    if UJ_PATH.exists():
        uj = pd.read_csv(UJ_PATH)
        uj["first_attempt"] = True
        frames.append(uj)
        print(f"usajigsaw: {len(uj)} rows")
    else:
        print(f"WARNING: {UJ_PATH} not found, skipping")

    # myspeedpuzzling.com
    if MSP_TIMES_PATH.exists() and MSP_PUZZLES_PATH.exists():
        msp = load_myspeedpuzzling()
        frames.append(msp)
        print(f"myspeedpuzzling: {len(msp)} rows")
    else:
        print(f"WARNING: myspeedpuzzling data not found, skipping")

    if not frames:
        raise RuntimeError("No data sources found")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\ncombined: {len(combined)} rows")
    return combined


if __name__ == "__main__":
    combined = combine()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\nBy source:")
    print(combined.groupby("source").size())
    print(f"\nUnique puzzlers: {combined['competitor_name'].nunique()}")
    print(f"Unique puzzle names: {combined['puzzle_name'].nunique()}")
    print(f"Rows with first_attempt=True: {combined['first_attempt'].sum()}")
