"""
Scraper for USA Jigsaw Puzzle Association nationals results.

Data is hosted on worldjigsawpuzzle.org but linked from usajigsaw.org.
Results are loaded dynamically via JavaScript, so we use Playwright.
"""

import pandas as pd
from playwright.sync_api import sync_playwright

from scraper.wjpf_common import extract_results_from_table, fetch_page_with_playwright

# Competition URLs - data hosted on worldjigsawpuzzle.org
COMPETITIONS = {
    # 2026 Nationals (Atlanta, March 27-29)
    ("2026", "individual", "A"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/individual/A",
    ("2026", "individual", "B"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/individual/B",
    ("2026", "individual", "C"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/individual/C",
    ("2026", "individual", "D"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/individual/D",
    ("2026", "individual", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/individual/final",
    ("2026", "pairs", "A"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/pairs/A",
    ("2026", "pairs", "B"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/pairs/B",
    ("2026", "pairs", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/pairs/final",
    ("2026", "teams", "A"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/teams/A",
    ("2026", "teams", "B"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/teams/B",
    ("2026", "teams", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2026/teams/final",
    # 2025 Nationals
    ("2025", "individual", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2025",
    ("2025", "pairs", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2025/pairs/final",
    ("2025", "teams", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2025/teams/final",
    # 2024 Nationals
    ("2024", "individual", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2024",
    ("2024", "pairs", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2024/pairs/final",
    ("2024", "teams", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2024/teams/final",
    # 2022 Nationals
    ("2022", "individual", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2022/individual",
    ("2022", "pairs", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2022/pairs",
    ("2022", "teams", "final"): "https://www.worldjigsawpuzzle.org/usajpa/nationals2022/teams/final",
}

# Known puzzle sizes by competition type
PUZZLE_SIZES = {
    "individual": 500,
    "pairs": 1000,
    "teams": 2000,
}

# Time limits in seconds
TIME_LIMITS = {
    "individual": 5400,   # 90 minutes
    "pairs": 7200,        # 2 hours
    "teams": 10800,       # 3 hours
}



def scrape_competition(page, year: str, division: str, round_: str = "final") -> pd.DataFrame:
    """
    Scrape a single competition's results.

    Args:
        page: Playwright page object
        year: Competition year (2022, 2024, 2025)
        division: "individual", "pairs", or "teams"
        round_: Round name (usually "final")

    Returns:
        DataFrame with competition results
    """
    key = (year, division, round_)
    if key not in COMPETITIONS:
        raise ValueError(f"Unknown competition: {key}")

    url = COMPETITIONS[key]
    print(f"Fetching {year} {division} {round_} from {url}")

    soup = fetch_page_with_playwright(page, url)
    results = extract_results_from_table(soup, division)

    if not results:
        print(f"  Warning: No results found for {key}")
        return pd.DataFrame()

    print(f"  Found {len(results)} results")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "source": "usajigsaw",
            "event_id": f"usajpa_{year}_{division}",
            "year": int(year),
            "division": division,
            "round": round_,
            "puzzle_pieces": PUZZLE_SIZES.get(division),
            "time_limit_seconds": TIME_LIMITS.get(division),
            "rank": r.rank,
            "competitor_name": r.name,
            "origin": r.origin,
            "time_seconds": r.time_seconds,
            "completed": r.completed,
            "pieces_completed": r.pieces_completed,
        }
        for r in results
    ])

    return df


def scrape_all() -> pd.DataFrame:
    """Scrape all known USA Jigsaw competitions using Playwright."""
    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for (year, division, round_) in COMPETITIONS.keys():
            try:
                df = scrape_competition(page, year, division, round_)
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                print(f"  Error scraping {year} {division}: {e}")

        browser.close()

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scraping USA Jigsaw Nationals results...")
    df = scrape_all()

    if df.empty:
        print("No data scraped!")
        sys.exit(1)

    output_path = output_dir / "usajigsaw_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to {output_path}")

    # Print summary
    print("\nSummary by year and division:")
    print(df.groupby(["year", "division"]).size().unstack(fill_value=0))
