"""Combine results from all sources into a single CSV.

Sources:
1. speedpuzzling.com - US competition results (scraped from PDFs)
2. usajigsaw - USA Jigsaw Puzzle Association results
3. myspeedpuzzling.com - global self-reported solving times
4. mallory - personal solve log (MalloryPuzzleData.csv)

Output schema:
    source, event_id, year, division, round, rank, competitor_name, origin,
    time_seconds, completed, pieces_completed, puzzle_pieces, puzzle_brand,
    puzzle_name, time_limit_seconds, first_attempt
"""

import hashlib
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SP_PATH = PROJECT_ROOT / "data" / "processed" / "speedpuzzling_results.csv"
UJ_PATH = PROJECT_ROOT / "data" / "processed" / "usajigsaw_results.csv"
MSP_PUZZLES_PATH = Path.home() / "build" / "myspeedpuzzling.com" / "scraper_output" / "puzzles.csv"
MSP_TIMES_PATH = Path.home() / "build" / "myspeedpuzzling.com" / "scraper_output" / "solving_times.csv"
MALLORY_PATH = PROJECT_ROOT / "data" / "MalloryPuzzleData.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "combined_results.csv"


# Manual overrides for Mallory's puzzle names -> canonical existing names.
# Maps (mallory_name, pieces) -> existing puzzle name (or None to force no match).
MALLORY_OVERRIDES: dict[tuple[str, int], str | None] = {
    ("Circle of Colors: Insects", 500): None,  # Different puzzle from "Donuts"
    ("Humming Bird", 500): None,  # Not the same as "Hummingbirds"
    ("Sonic", 500): "Classic Sonic",  # Same Ravensburger puzzle
    ("Wild Ones", 500): None,  # Not the same as "Wild Wonders"
    ("The Rivera", 500): None,  # Not the same as "The Riverbank"
    ("Sacred Lake Tahoe", 1000): "Lake Tahoe",  # Same Ravensburger puzzle
    ("Lake Como", 500): "Lake Como, Italy",  # Same Ravensburger puzzle
}


def _normalize_name(name: str) -> str:
    """Normalize puzzle name for matching: lowercase, strip, remove punctuation."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def _match_puzzle(
    name: str,
    pieces: int,
    existing: dict[tuple[str, int], tuple[str, str]],
    threshold: float = 0.85,
) -> tuple[str, str, str] | None:
    """Try to match a puzzle against existing (norm_name, pieces) -> (event_id, original_name).

    Returns (event_id, matched_name, match_type) or None.
    """
    norm = _normalize_name(name)

    # Exact normalized match
    if (norm, pieces) in existing:
        eid, orig = existing[(norm, pieces)]
        return eid, orig, "exact"

    # Fuzzy match against same piece count
    best_score, best_key = 0.0, None
    for (enorm, epieces), (eid, orig) in existing.items():
        if epieces != pieces:
            continue
        score = SequenceMatcher(None, norm, enorm).ratio()
        if score > best_score:
            best_score, best_key = score, (enorm, epieces)

    if best_score >= threshold and best_key is not None:
        eid, orig = existing[best_key]
        return eid, orig, f"fuzzy({best_score:.2f})"

    return None


def load_mallory(existing_puzzles: pd.DataFrame) -> pd.DataFrame:
    """Load Mallory's personal solve log and match puzzles to existing data.

    Args:
        existing_puzzles: DataFrame with columns [event_id, puzzle_name, puzzle_pieces]
            from already-loaded sources, used for puzzle matching.
    """
    df = pd.read_csv(MALLORY_PATH)

    # Filter to first attempts only
    df = df[df["Attempt"] == 1].copy()

    # Compute time in seconds from Hour, Min, Sec columns
    df["time_seconds"] = df["Hour"] * 3600 + df["Min"] * 60 + df["Sec"]

    # Parse year from Date column
    df["year"] = pd.to_datetime(df["Date"], format="mixed").dt.year

    # Build lookup of existing puzzles: (normalized_name, pieces) -> (event_id, original_name)
    existing = {}
    for _, row in existing_puzzles.drop_duplicates(
        subset=["puzzle_name", "puzzle_pieces"]
    ).iterrows():
        if pd.isna(row["puzzle_name"]):
            continue
        norm = _normalize_name(str(row["puzzle_name"]))
        key = (norm, int(row["puzzle_pieces"]))
        if key not in existing:
            existing[key] = (row["event_id"], row["puzzle_name"])

    # Match each puzzle and assign event_id
    event_ids = []
    puzzle_names = []
    for _, row in df.iterrows():
        key = (row["Puzzle"], row["# Pieces"])

        # Check manual overrides first
        if key in MALLORY_OVERRIDES:
            override = MALLORY_OVERRIDES[key]
            if override is None:
                # Forced no-match
                match = None
            else:
                # Look up the override name in existing puzzles
                norm = _normalize_name(override)
                if (norm, row["# Pieces"]) in existing:
                    eid, orig = existing[(norm, row["# Pieces"])]
                    match = (eid, orig, "override")
                else:
                    match = None
        else:
            match = _match_puzzle(row["Puzzle"], row["# Pieces"], existing)

        if match:
            eid, matched_name, match_type = match
            event_ids.append(eid)
            puzzle_names.append(matched_name)  # Use canonical name from existing data
            print(f"  MATCH ({match_type}): \"{row['Puzzle']}\" -> \"{matched_name}\" [{eid}]")
        else:
            # Generate a stable event_id from puzzle name + pieces
            h = hashlib.md5(f"{row['Puzzle']}_{row['# Pieces']}".encode()).hexdigest()[:8]
            eid = f"mal_{h}"
            event_ids.append(eid)
            puzzle_names.append(row["Puzzle"])
            print(f"  NEW: \"{row['Puzzle']}\" {row['# Pieces']}pc -> {eid}")

    df["event_id"] = event_ids
    df["matched_puzzle_name"] = puzzle_names

    return pd.DataFrame({
        "source": "mallory",
        "event_id": df["event_id"],
        "year": df["year"],
        "division": "solo",
        "round": None,
        "rank": None,
        "competitor_name": "Mallory Alemi",
        "origin": "US",
        "time_seconds": df["time_seconds"],
        "completed": True,
        "pieces_completed": None,
        "puzzle_pieces": df["# Pieces"],
        "puzzle_brand": df["Manufactorer"],
        "puzzle_name": df["matched_puzzle_name"],
        "time_limit_seconds": None,
        "first_attempt": True,
    })


def load_myspeedpuzzling() -> pd.DataFrame:
    """Load and normalize myspeedpuzzling.com data to the shared schema.

    Joins solving_times with puzzles catalog to get puzzle metadata,
    then maps columns to match the combined schema.
    """
    puzzles = pd.read_csv(MSP_PUZZLES_PATH)
    times = pd.read_csv(MSP_TIMES_PATH)

    # Only keep first-attempt solves (repeat solves would bias the model)
    times = times[times["first_attempt"] == True]

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

    # mallory (personal solve log, matched against existing puzzles)
    if MALLORY_PATH.exists():
        existing = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(f"\nMatching Mallory's puzzles against {existing['puzzle_name'].nunique()} existing puzzles...")
        mal = load_mallory(existing)
        frames.append(mal)
        print(f"mallory: {len(mal)} rows ({mal['event_id'].str.startswith('mal_').sum()} new puzzles)")
    else:
        print(f"WARNING: {MALLORY_PATH} not found, skipping")

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
