"""Suggest potential MSP-to-SP player links for manual review.

Finds MSP players whose display name plausibly matches an SP real name.
Two matching strategies:
  1. Exact "First Last" match (original)
  2. Fuzzy "First L" initial match with alpha/location disambiguation

Outputs candidates sorted by confidence tier for human review.

Usage:
    uv run python -m scripts.suggest_player_links
"""

import json
import math
import re
from collections import defaultdict

import pandas as pd

SP_PATH = "data/processed/speedpuzzling_results.csv"
MSP_TIMES_PATH = "data/raw/myspeedpuzzling/solving_times.csv"
MSP_PLAYERS_PATH = "data/raw/myspeedpuzzling/players.csv"
LINKS_PATH = "data/mappings/player_links.csv"
EXPLORER_PATH = "explorer_data.json"

# US state name -> abbreviation and vice versa for location matching
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}
ABBREV_TO_STATE = {v: k for k, v in US_STATES.items()}

FIRST_L_RE = re.compile(r"^(\S+) ([A-Z])\.?$")


def normalize(name: str) -> str:
    """Lowercase, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def extract_state(text: str | None) -> str | None:
    """Best-effort US state extraction from freeform location text."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    low = text.lower()

    # Direct state name match (SP origin is typically a full state name)
    if low in US_STATES:
        return low

    # Check for state abbreviation at end: "Houston, TX" or "Fort Knox, KY"
    m = re.search(r"\b([A-Z]{2})\b", text)
    if m and m.group(1) in ABBREV_TO_STATE:
        return ABBREV_TO_STATE[m.group(1)]

    # Check for state name substring
    for state_name in US_STATES:
        if state_name in low:
            return state_name

    return None


def load_alphas(explorer_path: str) -> dict[str, tuple[float, float]]:
    """Load alpha (mean, std) per puzzler name from explorer_data.json."""
    with open(explorer_path) as f:
        data = json.load(f)
    alphas = {}
    for p in data.get("puzzlers", []):
        name = p["name"]
        alpha = p.get("alpha")
        std = p.get("std")
        if alpha is not None and std is not None:
            alphas[name] = (alpha, std)
    return alphas


def alpha_z_score(
    alpha1: tuple[float, float] | None,
    alpha2: tuple[float, float] | None,
) -> float | None:
    """Compute z-score distance between two alpha estimates."""
    if alpha1 is None or alpha2 is None:
        return None
    mu1, s1 = alpha1
    mu2, s2 = alpha2
    denom = math.sqrt(s1**2 + s2**2)
    if denom < 1e-6:
        return None
    return abs(mu1 - mu2) / denom


def tier_label(match_type: str, n_candidates: int, z: float | None, loc_match: str | None) -> tuple[int, str]:
    """Assign confidence tier based on signals."""
    if z is not None and z > 3.0:
        return 5, "UNLIKELY"
    if match_type == "exact":
        return 1, "HIGH"
    # "First L" matches
    if n_candidates == 1:
        if z is not None and z < 2.0:
            return 2, "GOOD"
        if z is None:
            return 2, "GOOD"  # unique candidate, no alpha to contradict
        return 3, "MAYBE"  # z between 2-3
    # Multiple candidates
    if z is not None and z < 2.0 and loc_match == "match":
        return 3, "MAYBE"
    if z is not None and z < 2.0:
        return 3, "MAYBE"
    return 4, "WEAK"


def main():
    sp = pd.read_csv(SP_PATH)
    msp_times = pd.read_csv(MSP_TIMES_PATH)
    msp_players_df = pd.read_csv(MSP_PLAYERS_PATH)

    # Load existing links to exclude already-linked players
    try:
        existing = pd.read_csv(LINKS_PATH)
        linked_ids = set(existing["msp_player_id"])
    except (FileNotFoundError, KeyError):
        linked_ids = set()

    # Load alphas from explorer data
    alphas = load_alphas(EXPLORER_PATH)

    # Build alpha lookup helpers
    # MSP players appear as "DisplayName (msp:XXXXXXXX)" in explorer
    msp_alpha_by_prefix: dict[str, tuple[float, float]] = {}
    sp_alpha_by_name: dict[str, tuple[float, float]] = {}
    for name, alpha_val in alphas.items():
        m = re.search(r"\(msp:([0-9a-f]{8})\)", name)
        if m:
            msp_alpha_by_prefix[m.group(1)] = alpha_val
        elif ", " in name and "(msp:" not in name:
            sp_alpha_by_name[name] = alpha_val

    # Build SP name lookups
    sp_names = sp["competitor_name"].unique()

    # 1) Exact normalized lookup: "first last" -> "Last, First"
    sp_exact_lookup: dict[str, str] = {}
    # 2) Initial lookup: (first_lower, initial_upper) -> list of "Last, First"
    sp_initial_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for name in sp_names:
        if ", " not in name:
            continue
        last, first = name.split(", ", 1)
        key = normalize(f"{first} {last}")
        sp_exact_lookup[key] = name
        # Index by (first_name_lower, last_initial)
        first_lower = first.strip().lower()
        last_initial = last.strip()[0].upper() if last.strip() else ""
        if first_lower and last_initial:
            sp_initial_index[(first_lower, last_initial)].append(name)

    # Build SP player state from origin (most common per player)
    sp_player_state: dict[str, str | None] = {}
    for name, group in sp.groupby("competitor_name"):
        origins = group["origin"].dropna()
        if len(origins) > 0:
            sp_player_state[name] = extract_state(origins.mode().iloc[0])
        else:
            sp_player_state[name] = None

    # Build MSP player info from players.csv
    msp_player_info: dict[str, dict] = {}
    for _, row in msp_players_df.iterrows():
        pid = row["player_id"]
        msp_player_info[pid] = {
            "country": row.get("country"),
            "city": row.get("city"),
        }

    # Get unique MSP players with solve counts
    msp_players = (
        msp_times.groupby("player_id")
        .agg({"player_name": "first", "player_country": "first", "time_seconds": "count"})
        .rename(columns={"time_seconds": "n_solves"})
        .reset_index()
    )

    # Collect all candidates grouped by MSP player
    # Each candidate: dict with msp info + sp_name + signals
    candidates_by_msp: dict[str, list[dict]] = defaultdict(list)

    for _, row in msp_players.iterrows():
        pid = row["player_id"]
        if pid in linked_ids:
            continue
        display = row["player_name"]
        if not isinstance(display, str):
            continue

        norm = normalize(display)
        words = norm.split()
        if len(words) < 2:
            continue

        # Get MSP player info
        info = msp_player_info.get(pid, {})
        msp_country = info.get("country") or row.get("player_country")
        msp_city = info.get("city")
        msp_state = extract_state(msp_city)
        msp_alpha = msp_alpha_by_prefix.get(pid[:8])
        n_solves = row["n_solves"]

        # Strategy 1: Exact "First Last" match
        if norm in sp_exact_lookup:
            sp_name = sp_exact_lookup[norm]
            sp_alpha = sp_alpha_by_name.get(sp_name)
            z = alpha_z_score(msp_alpha, sp_alpha)
            sp_state = sp_player_state.get(sp_name)
            loc = _location_match(msp_state, sp_state)
            tier_num, tier_str = tier_label("exact", 1, z, loc)
            candidates_by_msp[pid].append({
                "msp_player_id": pid,
                "msp_display_name": display,
                "msp_country": msp_country,
                "msp_city": msp_city,
                "msp_solves": n_solves,
                "sp_name": sp_name,
                "sp_state": sp_state,
                "match_type": "exact",
                "alpha_z": z,
                "loc_match": loc,
                "tier_num": tier_num,
                "tier_str": tier_str,
            })

        # Strategy 2: "First L" initial match (only for US MSP players)
        m = FIRST_L_RE.match(display)
        if m and msp_country == "US":
            first_name = m.group(1).lower()
            last_initial = m.group(2)
            sp_candidates = sp_initial_index.get((first_name, last_initial), [])

            # Skip if this MSP player already had an exact match to the same SP name
            exact_matches = {c["sp_name"] for c in candidates_by_msp.get(pid, [])}

            for sp_name in sp_candidates:
                if sp_name in exact_matches:
                    continue
                sp_alpha = sp_alpha_by_name.get(sp_name)
                z = alpha_z_score(msp_alpha, sp_alpha)
                sp_state = sp_player_state.get(sp_name)
                loc = _location_match(msp_state, sp_state)
                tier_num, tier_str = tier_label("initial", len(sp_candidates), z, loc)
                candidates_by_msp[pid].append({
                    "msp_player_id": pid,
                    "msp_display_name": display,
                    "msp_country": msp_country,
                    "msp_city": msp_city,
                    "msp_solves": n_solves,
                    "sp_name": sp_name,
                    "sp_state": sp_state,
                    "match_type": "initial",
                    "alpha_z": z,
                    "loc_match": loc,
                    "tier_num": tier_num,
                    "tier_str": tier_str,
                })

    if not candidates_by_msp:
        print("No candidates found.")
        return

    # Sort MSP players by best tier (ascending), then by solve count (descending)
    sorted_pids = sorted(
        candidates_by_msp.keys(),
        key=lambda pid: (
            min(c["tier_num"] for c in candidates_by_msp[pid]),
            -max(c["msp_solves"] for c in candidates_by_msp[pid]),
        ),
    )

    # Print summary
    total_candidates = sum(len(cs) for cs in candidates_by_msp.values())
    print(f"Found {total_candidates} candidate links for {len(candidates_by_msp)} MSP players:\n")

    for pid in sorted_pids:
        cs = candidates_by_msp[pid]
        # Sort candidates within group: tier ascending, alpha_z ascending
        cs.sort(key=lambda c: (c["tier_num"], c["alpha_z"] if c["alpha_z"] is not None else 999))
        first = cs[0]
        city_str = f", {first['msp_city']}" if first["msp_city"] and isinstance(first["msp_city"], str) else ""
        country_str = first["msp_country"] or "?"
        print(
            f"--- {first['msp_display_name']} (msp:{pid[:8]}) "
            f"| {country_str}{city_str} | {first['msp_solves']} solves ---"
        )

        for c in cs:
            z_str = f"alpha_z={c['alpha_z']:.2f}" if c["alpha_z"] is not None else "alpha_z=N/A"
            loc_str = f"  location={c['loc_match']}" if c["loc_match"] else ""
            sp_state_str = f", {c['sp_state'].title()}" if c["sp_state"] else ""
            print(
                f"  -> \"{c['sp_name']}\" (SP{sp_state_str})  "
                f"{z_str}{loc_str}  {c['tier_str']}"
            )
            print(
                f'{c["msp_player_id"]},"{c["sp_name"]}",confirmed,'
                f'{c["msp_display_name"]} on MSP'
            )
        print()


def _location_match(msp_state: str | None, sp_state: str | None) -> str | None:
    """Compare extracted states. Returns 'match', 'mismatch', or None."""
    if msp_state is None or sp_state is None:
        return None
    if msp_state == sp_state:
        return "match"
    return "mismatch"


if __name__ == "__main__":
    main()
