"""
Shared utilities for scrapers that use worldjigsawpuzzle.org.

Both usajigsaw.py and wjpf.py scrape from the same website with
identical table structures (#participantes). This module provides
the common Playwright-based page fetching and result extraction.
"""

import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class CompetitorResult:
    """A single competitor/team result."""
    rank: int
    name: str
    origin: Optional[str]
    time_seconds: Optional[int]  # None if DNF
    completed: bool
    pieces_completed: Optional[int]  # For DNFs
    country: Optional[str] = None  # ISO country code from flag image


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


def _extract_country_from_row(row) -> Optional[str]:
    """Extract country code from flag <img> tag in a table row.

    WJPF uses flag images with paths like:
        /img/flags/shiny/64/ES.png
        img/flags/24/NO.png
    """
    imgs = row.find_all("img")
    for img in imgs:
        src = img.get("src", "")
        if "flag" in src.lower():
            # Extract the country code from the filename
            match = re.search(r"/([A-Z]{2})\.png", src, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # Also check title/alt attributes
        title = img.get("title", "") or img.get("alt", "")
        if title and "flag" not in title.lower():
            # Some pages use the country name as the title
            pass
    return None


def _extract_name_origin_country(cells, cell_texts, row) -> tuple[str, str | None, str | None]:
    """Extract competitor name, origin, and country from table cells.

    Handles two table layouts found on worldjigsawpuzzle.org:

    WJPC layout (8 cells):
      cell 0: rank
      cell 1: flag image only (empty text)
      cell 2: compound cell with <div>s for country, name, origin
      cell 3: origin text
      cell 4: country name
      cell 5+: time, gaps

    USAJPA layout (6 cells):
      cell 0: rank
      cell 1: name + origin concatenated
      cell 2: origin text
      cell 3+: time, gaps
    """
    # Detect WJPC layout: cell 1 is empty (flag-only) and cell 2 has name divs
    is_wjpc = len(cells) >= 8 and not cell_texts[1].strip()

    if is_wjpc:
        # Extract name from the plain <div> in cell 2 (not ver_movil/pais_movil)
        name = ""
        for div in cells[2].find_all("div", recursive=False):
            classes = div.get("class") or []
            if "ver_movil" not in classes and "pais_movil" not in classes:
                # This div contains the competitor name(s)
                # For pairs/teams, members are separated by <br/> tags
                name = div.get_text(separator="\n", strip=True)
                # Normalize internal whitespace but keep \n for member separation
                name = "\n".join(part.strip() for part in name.split("\n") if part.strip())
                # Join multi-line names with " & " for pairs/teams readability
                parts = [p for p in name.split("\n") if p]
                name = " & ".join(parts) if len(parts) > 1 else name.replace("\n", " ")
                break

        origin = cell_texts[3].strip() if len(cell_texts) > 3 else None
        country = cell_texts[4].strip() if len(cell_texts) > 4 else None

        # Fallback: get country from flag image if not in cell 4
        if not country:
            country = _extract_country_from_row(row)
    else:
        # USAJPA layout: name (+ origin) in cell 1
        name_cell = cell_texts[1] if len(cell_texts) > 1 else ""
        name = name_cell.split("\n")[0].strip()
        origin = cell_texts[2].strip() if len(cell_texts) > 2 else None
        country = _extract_country_from_row(row)

    return name, origin or None, country or None


def extract_results_from_table(soup: BeautifulSoup, division: str) -> list[CompetitorResult]:
    """
    Extract results from the competition table.

    Handles both WJPC (8-cell) and USAJPA (6-cell) table layouts.
    Time column is detected dynamically by pattern matching.
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

        # Extract name, origin, country (handles both table layouts)
        name, origin, country = _extract_name_origin_country(cells, cell_texts, row)

        # Find time column dynamically (searches from cell 3 onward)
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
            country=country,
        ))

    return results
