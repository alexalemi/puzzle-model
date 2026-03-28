"""
Scraper for World Jigsaw Puzzle Federation championship results.

Scrapes WJPC (World Jigsaw Puzzle Championship) results from
worldjigsawpuzzle.org. Uses Playwright for JavaScript-rendered pages
(same infrastructure as the USAJPA scraper).

Only scrapes the General classification — age subcategories (Junior18,
Senior45-60, etc.) are subsets and would double-count competitors.
"""

import re
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

from scraper.wjpf_common import extract_results_from_table, fetch_page_with_playwright

BASE_URL = "https://www.worldjigsawpuzzle.org"

# Division → piece count (matches USAJPA conventions)
PUZZLE_SIZES = {
    "individual": 500,
    "pairs": 1000,
    "teams": 2000,
}

# Division name normalization for the combined schema
DIVISION_MAP = {
    "individual": "solo",
    "pairs": "duo",
    "teams": "group",
}

# Known WJPC events: (year, wjpf_division, round, url_path)
#
# The scraper tries each URL and gracefully skips pages that don't load
# a results table within the Playwright timeout. This means we can be
# generous with plausible URLs — non-existent rounds just get skipped.
#
# Structure by year:
#   2019: single round per division (1st WJPC, no qualifying groups)
#   2022: A/B qualifying + final (2nd WJPC, post-COVID return)
#   2023: A/B/C qualifying + S1/S2 semifinals + final
#   2024-2025: A-F qualifying + S1/S2 semifinals + final (expanded field)
WJPC_EVENTS: list[tuple[int, str, str, str]] = [
    # 2019 — 1st WJPC, single round, 2h individual time limit
    (2019, "individual", "final", "/wjpc/2019/individual"),
    (2019, "pairs", "final", "/wjpc/2019/pairs"),
    (2019, "teams", "final", "/wjpc/2019/teams"),
    # 2022 — qualifying rounds + final
    *[(2022, "individual", r, f"/wjpc/2022/individual/{r}")
      for r in ("A", "B", "C", "final")],
    *[(2022, "pairs", r, f"/wjpc/2022/pairs/{r}")
      for r in ("A", "B", "final")],
    *[(2022, "teams", r, f"/wjpc/2022/teams/{r}")
      for r in ("A", "final")],
    # 2023
    *[(2023, "individual", r, f"/wjpc/2023/individual/{r}")
      for r in ("A", "B", "C", "S1", "S2", "final")],
    *[(2023, "pairs", r, f"/wjpc/2023/pairs/{r}")
      for r in ("A", "B", "final")],
    *[(2023, "teams", r, f"/wjpc/2023/teams/{r}")
      for r in ("A", "final")],
    # 2024 — expanded field
    *[(2024, "individual", r, f"/wjpc/2024/individual/{r}")
      for r in ("A", "B", "C", "D", "E", "F", "S1", "S2", "final")],
    *[(2024, "pairs", r, f"/wjpc/2024/pairs/{r}")
      for r in ("A", "B", "C", "S1", "final")],
    *[(2024, "teams", r, f"/wjpc/2024/teams/{r}")
      for r in ("A", "B", "final")],
    # 2025
    *[(2025, "individual", r, f"/wjpc/2025/individual/{r}")
      for r in ("A", "B", "C", "D", "E", "F", "S1", "S2", "final")],
    *[(2025, "pairs", r, f"/wjpc/2025/pairs/{r}")
      for r in ("A", "B", "C", "S1", "final")],
    *[(2025, "teams", r, f"/wjpc/2025/teams/{r}")
      for r in ("A", "B", "final")],
]


def _extract_page_metadata(soup) -> dict:
    """Extract time limit and event date from page JavaScript.

    WJPF pages embed competition metadata in <script> tags:
        var maximosegundos = 4500;
        var horainicioUTC = new Date("2024-09-21T15:00:00Z");
    """
    metadata: dict = {"time_limit_seconds": None, "finished_date": None}

    for script in soup.find_all("script"):
        text = script.string or ""

        # Time limit in seconds
        m = re.search(r"maximosegundos\s*=\s*(\d+)", text)
        if m:
            metadata["time_limit_seconds"] = int(m.group(1))

        # Event date from ISO timestamp
        m = re.search(r'horainicioUTC.*?new Date\("([^"]+)"\)', text)
        if m:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
                metadata["finished_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Alternative: DD/MM/YYYY format
        if not metadata["finished_date"]:
            m = re.search(r"(\d{2}/\d{2}/\d{4})\s*,?\s*\d{2}:\d{2}", text)
            if m:
                from datetime import datetime

                try:
                    dt = datetime.strptime(m.group(1), "%d/%m/%Y")
                    metadata["finished_date"] = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

    return metadata


def _parse_team_members(name: str) -> str | None:
    """Extract individual member names from a pair/team entry.

    Pairs: "Name1 & Name2" or "Name1 / Name2"
    Teams: may have 3+ members separated by & or /

    Returns semicolon-separated member names, or None for solo.
    """
    for sep in (" & ", " / "):
        if sep in name:
            members = [m.strip() for m in name.split(sep) if m.strip()]
            if len(members) >= 2:
                return ";".join(members)
    return None


def scrape_event(page, year: int, division: str, round_: str,
                 url_path: str) -> pd.DataFrame:
    """Scrape a single WJPC event page.

    Returns empty DataFrame if the page doesn't load or has no results.
    """
    url = BASE_URL + url_path
    print(f"  Fetching {year} {division} {round_} from {url}")

    try:
        soup = fetch_page_with_playwright(page, url)
    except Exception as e:
        print(f"    Skipped (page didn't load): {e}")
        return pd.DataFrame()

    results = extract_results_from_table(soup, division)
    if not results:
        print(f"    Skipped (no results in table)")
        return pd.DataFrame()

    # Extract metadata from page JS
    meta = _extract_page_metadata(soup)
    print(f"    Found {len(results)} results"
          f" (time_limit={meta['time_limit_seconds']}s,"
          f" date={meta['finished_date']})")

    norm_div = DIVISION_MAP[division]
    event_id = f"wjpc_{year}_{norm_div}"
    is_team = division in ("pairs", "teams")

    rows = []
    for r in results:
        origin = r.origin or ""
        if r.country and r.country not in origin:
            origin = f"{origin}, {r.country}" if origin else r.country

        row = {
            "source": "wjpf",
            "event_id": event_id,
            "year": year,
            "division": norm_div,
            "round": round_,
            "rank": r.rank,
            "competitor_name": r.name,
            "origin": origin or None,
            "time_seconds": r.time_seconds,
            "completed": r.completed,
            "pieces_completed": r.pieces_completed,
            "puzzle_pieces": PUZZLE_SIZES.get(division),
            "puzzle_brand": None,
            "puzzle_name": None,  # Secret competition puzzles; filled in combine.py
            "finished_date": meta["finished_date"],
            "first_attempt": True,
            "time_limit_seconds": meta["time_limit_seconds"],
            "team_members": _parse_team_members(r.name) if is_team else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def scrape_all() -> pd.DataFrame:
    """Scrape all known WJPC championship events."""
    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for year, division, round_, url_path in WJPC_EVENTS:
            df = scrape_event(page, year, division, round_, url_path)
            if not df.empty:
                all_results.append(df)

        browser.close()

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Deduplicate (same competitor shouldn't appear twice in same round)
    n_before = len(combined)
    combined = combined.drop_duplicates(
        subset=["event_id", "round", "competitor_name"]
    )
    n_dupes = n_before - len(combined)
    if n_dupes:
        print(f"\nDedup: dropped {n_dupes} duplicate rows")

    return combined


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scraping WJPC championship results...")
    print(f"({len(WJPC_EVENTS)} event pages to try)\n")
    df = scrape_all()

    if df.empty:
        print("No data scraped!")
        sys.exit(1)

    output_path = output_dir / "wjpf_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to {output_path}")

    # Summary
    print("\nBy year and division:")
    print(df.groupby(["year", "division"]).size().unstack(fill_value=0))
    print(f"\nBy year and round:")
    print(df.groupby(["year", "round"]).size().unstack(fill_value=0))
    print(f"\nUnique competitors: {df['competitor_name'].nunique()}")
