"""
Scraper for USA Jigsaw Puzzle Association nationals results.

Data is hosted on worldjigsawpuzzle.org but linked from usajigsaw.org.
Results are loaded dynamically via JavaScript, so we use Playwright.
"""

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# Competition URLs - data hosted on worldjigsawpuzzle.org
COMPETITIONS = {
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


@dataclass
class CompetitorResult:
    """A single competitor/team result."""
    rank: int
    name: str
    origin: Optional[str]
    time_seconds: Optional[int]  # None if DNF
    completed: bool
    pieces_completed: Optional[int]  # For DNFs


def parse_time_or_pieces(time_str: str) -> tuple[Optional[int], bool, Optional[int]]:
    """
    Parse a time string that could be either a completion time or a piece count (DNF).

    Args:
        time_str: Either "HH:MM:SS" format or "N Pieces" / "N,NNN Pieces" format

    Returns:
        Tuple of (time_seconds, completed, pieces_completed)
        - For completions: (seconds, True, None)
        - For DNFs: (None, False, piece_count)

    Examples:
        "00:40:34" -> (2434, True, None)
        "1,977 Pieces" -> (None, False, 1977)
        "1.977 Pieces" -> (None, False, 1977)  # European decimal format
    """
    time_str = time_str.strip()

    # Check if it's a piece count (DNF)
    if "piece" in time_str.lower():
        # Extract number, handling both "1,977" and "1.977" formats
        num_str = re.sub(r"[^\d]", "", time_str.split()[0].replace(".", "").replace(",", ""))
        pieces = int(num_str) if num_str else None
        return (None, False, pieces)

    # Parse HH:MM:SS time format
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return (h * 3600 + m * 60 + s, True, None)
    elif len(parts) == 2:
        m, s = map(int, parts)
        return (m * 60 + s, True, None)

    # Unknown format
    return (None, False, None)


def fetch_page_with_playwright(page, url: str) -> BeautifulSoup:
    """
    Fetch a page using Playwright, wait for JS to load data, return BeautifulSoup.

    Args:
        page: Playwright page object (reused for efficiency)
        url: URL to fetch
    """
    page.goto(url)
    # Wait for the results table to be populated
    # The table has id="participantes" and rows are added by JS
    page.wait_for_selector("#participantes tr", timeout=15000)
    # Give a moment for all rows to load
    page.wait_for_timeout(1000)

    html = page.content()
    return BeautifulSoup(html, "lxml")


def find_time_in_cells(cell_texts: list[str]) -> str:
    """
    Find the cell containing time or pieces data.

    Looks for patterns like "HH:MM:SS" or "N Pieces" in any cell.
    Returns the first matching cell content.
    """
    time_pattern = re.compile(r"^\d{1,2}:\d{2}:\d{2}")
    pieces_pattern = re.compile(r"^\d[\d,.\s]*piece", re.IGNORECASE)

    for cell in cell_texts[3:]:  # Skip rank, name, origin
        cell = cell.strip()
        if not cell or cell == "-":
            continue
        if time_pattern.match(cell) or pieces_pattern.match(cell):
            return cell
    return ""


def extract_results_from_table(soup: BeautifulSoup, division: str) -> list[CompetitorResult]:
    """
    Extract results from the competition table.

    Table structure varies by division:
    - Individual: rank, name+origin, origin, time, gap1, gap2
    - Teams: rank, teamname+members, city, state?, time, gap1, gap2

    We detect the time column by looking for HH:MM:SS or "N Pieces" pattern.
    """
    results = []
    table = soup.find("table", {"id": "participantes"})

    if not table:
        table = soup.find("table", {"id": "tabla_clasificacion"})

    if not table:
        return results

    rows = table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Extract text from cells
        cell_texts = []
        for cell in cells:
            text = cell.get_text(separator=" ", strip=True)
            cell_texts.append(text)

        # Skip header-like rows
        rank_text = cell_texts[0].strip()
        if not rank_text or not rank_text.isdigit():
            continue

        rank = int(rank_text)

        # Cell 1 contains name (possibly with origin on newline)
        name_cell = cell_texts[1] if len(cell_texts) > 1 else ""
        name = name_cell.split("\n")[0].strip()

        # Cell 2 contains origin (city, possibly state in next cell)
        origin = cell_texts[2].strip() if len(cell_texts) > 2 else None

        # Find time column dynamically (structure varies by division)
        time_cell = find_time_in_cells(cell_texts)

        # Extract just the time/pieces part (before any gap info)
        time_str = time_cell.split()[0] if time_cell else ""

        # For pieces, we need to rejoin "1,977 Pieces"
        if "piece" in time_cell.lower():
            time_str = time_cell

        time_seconds, completed, pieces = parse_time_or_pieces(time_str)

        results.append(CompetitorResult(
            rank=rank,
            name=name,
            origin=origin,
            time_seconds=time_seconds,
            completed=completed,
            pieces_completed=pieces,
        ))

    return results


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
