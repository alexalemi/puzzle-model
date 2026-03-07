"""Suggest potential MSP-to-SP player links for manual review.

Finds MSP players whose display name plausibly matches an SP real name.
Outputs candidates sorted by confidence for human review.

Usage:
    uv run python -m scripts.suggest_player_links
"""

import re

import pandas as pd

SP_PATH = "data/processed/speedpuzzling_results.csv"
MSP_TIMES_PATH = "data/raw/myspeedpuzzling/solving_times.csv"
LINKS_PATH = "data/mappings/player_links.csv"


def normalize(name: str) -> str:
    """Lowercase, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def main():
    sp = pd.read_csv(SP_PATH)
    msp_times = pd.read_csv(MSP_TIMES_PATH)

    # Load existing links to exclude already-linked players
    try:
        existing = pd.read_csv(LINKS_PATH)
        linked_ids = set(existing["msp_player_id"])
    except (FileNotFoundError, KeyError):
        linked_ids = set()

    # Build SP name lookup: normalized -> original "Last, First"
    sp_names = sp["competitor_name"].unique()
    sp_lookup: dict[str, str] = {}
    for name in sp_names:
        # "Last, First" -> normalize both parts
        if ", " in name:
            last, first = name.split(", ", 1)
            # Index by "first last" normalized
            key = normalize(f"{first} {last}")
            sp_lookup[key] = name
            # Also index by last name alone
            sp_lookup[normalize(last)] = name

    # Get unique MSP players
    msp_players = (
        msp_times.groupby("player_id")
        .agg({"player_name": "first", "player_country": "first", "time_seconds": "count"})
        .rename(columns={"time_seconds": "n_solves"})
        .reset_index()
    )

    candidates = []
    for _, row in msp_players.iterrows():
        if row["player_id"] in linked_ids:
            continue
        display = row["player_name"]
        if not isinstance(display, str):
            continue
        norm = normalize(display)

        # Only match multi-word display names (single words are too ambiguous)
        if len(norm.split()) < 2:
            continue

        # Try exact match on normalized "first last"
        if norm in sp_lookup:
            candidates.append({
                "msp_player_id": row["player_id"],
                "msp_display_name": display,
                "sp_name": sp_lookup[norm],
                "msp_country": row["player_country"],
                "msp_solves": row["n_solves"],
                "match_type": "exact",
            })

    if not candidates:
        print("No candidates found.")
        return

    df = pd.DataFrame(candidates).sort_values(
        ["match_type", "msp_solves"], ascending=[True, False]
    )
    print(f"Found {len(df)} candidate links for review:\n")
    for _, row in df.iterrows():
        print(
            f"{row['msp_display_name']!r} -> {row['sp_name']!r}"
            f"  ({row['match_type']}, {row['msp_country']}, {row['msp_solves']} solves)"
        )
        print(
            f'{row["msp_player_id"]},"{row["sp_name"]}",confirmed,'
            f'{row["msp_display_name"]} on MSP'
        )
        print()


if __name__ == "__main__":
    main()
