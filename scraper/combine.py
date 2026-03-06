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


# Manual overrides for SP puzzle names -> canonical MSP names.
# Used when the naming difference can't be resolved by automatic normalization
# (case, unicode apostrophes, "(exclusive)" suffix stripping).
# Maps (sp_name, pieces) -> msp_name.
SP_TO_MSP_OVERRIDES: dict[tuple[str, int], str] = {
    # Series prefixes in MSP
    ("Dutch Welfare State", 500): "Velvet Soft Touch: Dutch Welfare State",
    ("Fresh Pie Tonight", 300): "Green Acres: Fresh Pie Tonight",
    ("In Search of the Child", 500): 'Star Wars: The Mandalorian "In Search of The Child"',
    ("Picnic Raiders", 300): "Adorable Animals Picnic Raiders",
    ("Quarantine Moods", 500): "Velvet Soft Touch: Quarantine Moods",
    ("The Wild North", 500): "Amazing Nature: The Wild North",
    ("Trading Cards", 500): "Star Wars Trading Cards",
    ("Campside - Oceanside Camping", 300): "Oceanside camping",
    ("Coca Cola Barn Dance", 300): "Barn Dance",
    # Renames / typos
    ("Beechcraft Scatterwing", 500): "Beechcraft Staggerwing",
    ("Ellen Shershow Photography", 500): "Dogs by Ellen Shershow",
    # Comma difference
    ("My Hair, My Crown", 300): "My Hair My crown",
}


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

    # Compute time in seconds from Hour, Min, Sec columns
    df["time_seconds"] = df["Hour"] * 3600 + df["Min"] * 60 + df["Sec"]

    # Parse date for year and finished_date (needed for repeat-solve features)
    df["parsed_date"] = pd.to_datetime(df["Date"], format="mixed")
    df["year"] = df["parsed_date"].dt.year
    df["finished_date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")

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
        "first_attempt": df["Attempt"] == 1,
        "finished_date": df["finished_date"],
    })


def _normalize_for_matching(name: str) -> str:
    """Normalize a puzzle name for cross-source matching.

    Lowercases, strips whitespace, normalizes unicode, and maps
    curly quotes/apostrophes to ASCII equivalents.
    """
    import unicodedata

    name = unicodedata.normalize("NFKC", name).lower().strip()
    # Map curly single quotes to ASCII apostrophe
    name = name.replace("\u2018", "'").replace("\u2019", "'")
    # Map curly double quotes to ASCII
    name = name.replace("\u201c", '"').replace("\u201d", '"')
    return name


def normalize_sp_to_msp(sp: pd.DataFrame, msp: pd.DataFrame) -> pd.DataFrame:
    """Rename SP puzzle names to match MSP canonical names where possible.

    Handles: case differences, unicode apostrophes, "(exclusive)" suffix,
    and manual overrides for series prefixes / renames.
    """
    sp = sp.copy()

    # Build MSP lookup: (normalized_name, pieces) -> original_name
    msp_lookup: dict[tuple[str, int], str] = {}
    for _, row in (
        msp[["puzzle_name", "puzzle_pieces"]]
        .dropna(subset=["puzzle_name"])
        .drop_duplicates()
        .iterrows()
    ):
        name = str(row["puzzle_name"])
        pieces = int(row["puzzle_pieces"])
        norm = _normalize_for_matching(name)
        msp_lookup[(norm, pieces)] = name

    renamed = 0
    for idx, row in sp.iterrows():
        if pd.isna(row.get("puzzle_name")) or pd.isna(row.get("puzzle_pieces")):
            continue
        sp_name = str(row["puzzle_name"])
        pieces = int(row["puzzle_pieces"])

        # 1. Check manual overrides first
        if (sp_name, pieces) in SP_TO_MSP_OVERRIDES:
            sp.at[idx, "puzzle_name"] = SP_TO_MSP_OVERRIDES[(sp_name, pieces)]
            renamed += 1
            continue

        # 2. Normalize and try exact match (handles case + unicode apostrophes)
        norm = _normalize_for_matching(sp_name)
        if (norm, pieces) in msp_lookup:
            msp_name = msp_lookup[(norm, pieces)]
            if msp_name != sp_name:
                sp.at[idx, "puzzle_name"] = msp_name
                renamed += 1
            continue

        # 3. Strip "(exclusive)" suffix and try again
        if "(exclusive)" in sp_name.lower():
            stripped = sp_name.replace(" (exclusive)", "").replace(" (Exclusive)", "")
            norm_stripped = _normalize_for_matching(stripped)
            if (norm_stripped, pieces) in msp_lookup:
                sp.at[idx, "puzzle_name"] = msp_lookup[(norm_stripped, pieces)]
                renamed += 1
                continue

    print(f"  SP->MSP name normalization: {renamed} rows renamed")
    return sp


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

    # Parse finished_date and extract year
    df["finished_date_parsed"] = pd.to_datetime(df["finished_date"], errors="coerce")
    df["year"] = df["finished_date_parsed"].dt.year

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
        "finished_date": df["finished_date_parsed"],
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

    # Normalize SP puzzle names toward MSP canonical names
    if len(frames) >= 2:
        # frames[0] is SP, find MSP frame
        sp_idx = next((i for i, f in enumerate(frames) if f["source"].iloc[0] == "speedpuzzling"), None)
        msp_idx = next((i for i, f in enumerate(frames) if f["source"].iloc[0] == "myspeedpuzzling"), None)
        if sp_idx is not None and msp_idx is not None:
            frames[sp_idx] = normalize_sp_to_msp(frames[sp_idx], frames[msp_idx])

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
    print(f"Rows with first_attempt=False: {(~combined['first_attempt']).sum()}")
    print(f"Rows with finished_date: {combined['finished_date'].notna().sum()}")
