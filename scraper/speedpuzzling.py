"""
Scraper for speedpuzzling.com competition results.

PDFs are linked from results pages and contain Excel-generated tables.
This module handles:
1. Discovering PDF links from results pages
2. Downloading PDFs
3. Extracting tabular data using pdfplumber
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

# Results pages to scrape (newest to oldest)
RESULTS_PAGES = [
    "https://www.speedpuzzling.com/results.html",
    "https://www.speedpuzzling.com/results2025.html",
    "https://www.speedpuzzling.com/results2024.html",
    "https://www.speedpuzzling.com/results-2023.html",
    "https://www.speedpuzzling.com/results-2022.html",
    "https://www.speedpuzzling.com/results-2021.html",
    "https://www.speedpuzzling.com/results-2020.html",
]

BASE_URL = "https://www.speedpuzzling.com"


@dataclass
class PDFInfo:
    """Metadata about a discovered PDF."""
    url: str
    filename: str
    event_id: str  # e.g., "sp_456" or "ca2024"
    division: Optional[str] = None  # solo, pair, team, etc.
    round: Optional[str] = None  # quarterfinal, semifinal, final, overall
    year: Optional[int] = None
    event_name: Optional[str] = None
    local_path: Optional[Path] = None


def parse_pdf_filename(filename: str) -> dict:
    """
    Extract metadata from PDF filename.

    Examples:
        sp_456q_results.pdf -> event=456, round=quarterfinal
        sp_456s_results.pdf -> event=456, division=solo
        ca2024_solo_results.pdf -> event=ca2024, division=solo
        speed-puzzling-012-individual_2.pdf -> event=12, division=individual
    """
    info = {"event_id": None, "division": None, "round": None, "year": None}

    # Remove extension and _results suffix
    name = filename.lower().replace(".pdf", "").replace("_results", "").replace("-results", "")

    # Try to extract year
    year_match = re.search(r"20(2[0-5])", name)
    if year_match:
        info["year"] = int("20" + year_match.group(1))

    # Pattern 1: sp_NNN[suffix] (modern format)
    match = re.match(r"sp[_-]?(\d+)([a-z])?", name)
    if match:
        info["event_id"] = f"sp_{match.group(1)}"
        suffix = match.group(2)
        if suffix:
            suffix_map = {
                "s": "solo", "i": "solo",
                "p": "pair",
                "t": "team", "q": "team",  # 'q' can mean quad/team
                "j": "junior",
                "o": "overall",
                "f": "final",
            }
            # Check if it's a round indicator
            if suffix in ("q", "s", "f", "o") and "quarterfinal" in name or "semifinal" in name:
                round_map = {"q": "quarterfinal", "s": "semifinal", "f": "final", "o": "overall"}
                info["round"] = round_map.get(suffix)
            else:
                info["division"] = suffix_map.get(suffix)
        return info

    # Pattern 2: state championship (ca2024, tx2023, etc.)
    match = re.match(r"([a-z]{2})(20\d{2})", name)
    if match:
        info["event_id"] = f"{match.group(1)}{match.group(2)}"
        info["year"] = int(match.group(2))
        # Look for division in rest of filename
        if "solo" in name or "individual" in name:
            info["division"] = "solo"
        elif "pair" in name:
            info["division"] = "pair"
        elif "team" in name:
            info["division"] = "team"
        return info

    # Pattern 3: older format (speed-puzzling-NNN-category)
    match = re.search(r"speed[_-]?puzzling[_-]?(\d+)", name)
    if match:
        info["event_id"] = f"sp_{match.group(1)}"
        if "individual" in name or "solo" in name:
            info["division"] = "solo"
        elif "pair" in name:
            info["division"] = "pair"
        elif "team" in name or "quad" in name:
            info["division"] = "team"
        return info

    # Pattern 4: special events (cruise, ssmt, etc.)
    for prefix in ["cruise", "ssmt", "ner", "ser", "wc"]:
        if name.startswith(prefix):
            info["event_id"] = name.split("_")[0].split("-")[0]
            if "solo" in name:
                info["division"] = "solo"
            elif "pair" in name:
                info["division"] = "pair"
            elif "team" in name:
                info["division"] = "team"
            return info

    # Fallback: use filename as event_id
    info["event_id"] = name.split("_")[0].split("-")[0]
    return info


def infer_year_from_url(url: str) -> Optional[int]:
    """
    Infer year from results page URL.

    Examples:
        results.html -> 2025 (current/default)
        results2024.html -> 2024
        results-2023.html -> 2023
        results-2020.html -> 2020
    """
    # Try to find year in URL
    match = re.search(r"results[_-]?(20\d{2})", url)
    if match:
        return int(match.group(1))

    # Default: results.html is current year (2025)
    if url.endswith("results.html"):
        return 2025

    return None


def discover_pdfs(url: str, delay: float = 1.0) -> list[PDFInfo]:
    """
    Discover all PDF links on a results page.

    Returns list of PDFInfo objects with metadata extracted from
    filenames and surrounding page context. Year is inferred from
    the results page URL if not present in filename.
    """
    time.sleep(delay)
    headers = {"User-Agent": "Mozilla/5.0 (puzzle-model research project)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    pdfs = []

    # Infer year from the results page URL
    page_year = infer_year_from_url(url)

    # Find all links to PDFs
    for link in soup.find_all("a", href=re.compile(r"\.pdf$", re.I)):
        href = link.get("href", "")
        if not href:
            continue

        # Build full URL
        pdf_url = urljoin(BASE_URL, href)
        filename = Path(href).name

        # Parse filename for metadata
        meta = parse_pdf_filename(filename)

        # Use year from filename if available, otherwise from page URL
        year = meta.get("year") or page_year

        pdfs.append(PDFInfo(
            url=pdf_url,
            filename=filename,
            event_id=meta.get("event_id", filename),
            division=meta.get("division"),
            round=meta.get("round"),
            year=year,
        ))

    return pdfs


def discover_all_pdfs() -> list[PDFInfo]:
    """Discover PDFs from all results pages."""
    all_pdfs = []
    seen_urls = set()

    for page_url in RESULTS_PAGES:
        print(f"Scanning {page_url}...")
        try:
            pdfs = discover_pdfs(page_url)
            # Deduplicate by URL
            for pdf in pdfs:
                if pdf.url not in seen_urls:
                    seen_urls.add(pdf.url)
                    all_pdfs.append(pdf)
            print(f"  Found {len(pdfs)} PDFs ({len(all_pdfs)} total unique)")
        except Exception as e:
            print(f"  Error: {e}")

    return all_pdfs


def download_pdf(pdf: PDFInfo, output_dir: Path, delay: float = 0.5) -> bool:
    """
    Download a PDF to the output directory.

    Returns True if successful, False otherwise.
    """
    output_path = output_dir / pdf.filename

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        pdf.local_path = output_path
        return True

    time.sleep(delay)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (puzzle-model research project)"}
        response = requests.get(pdf.url, headers=headers, timeout=60)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        pdf.local_path = output_path
        return True
    except Exception as e:
        print(f"  Error downloading {pdf.filename}: {e}")
        return False


def download_all_pdfs(pdfs: list[PDFInfo], output_dir: Path) -> int:
    """
    Download all PDFs to output directory.

    Returns count of successfully downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for i, pdf in enumerate(pdfs):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(pdfs)}")
        if download_pdf(pdf, output_dir):
            success_count += 1

    return success_count


@dataclass
class PDFExtractResult:
    """Result of PDF extraction including metadata from text."""
    df: Optional[pd.DataFrame]
    puzzle_pieces: Optional[int] = None
    puzzle_name: Optional[str] = None
    puzzle_brand: Optional[str] = None
    event_number: Optional[int] = None
    division: Optional[str] = None  # solo, pair, team, junior
    event_date: Optional[str] = None  # YYYY-MM-DD from PDF creation date


def _extract_puzzle_title(next_line: str) -> Optional[str]:
    """Extract puzzle title from the line following the 'Brand NNN piece' line.

    The PDF header has the brand+piece count on one line, with the actual puzzle
    title either on its own line or appended to a disclaimer/note line.

    Patterns:
    1. Clean title: "Backyard Heroes" -> "Backyard Heroes"
    2. After disclaimer: "...ineligible for prizes. The Zoo" -> "The Zoo"
    3. After note: "**...automated.** Cabin in the Mountains" -> "Cabin in the Mountains"
    4. Table header: "Rank %ile Points..." -> None
    """
    if not next_line:
        return None

    next_line = next_line.strip()

    # Check if it's a table header line — skip these
    if re.match(r"^(Rank\s|%ile\s|This Contest)", next_line):
        return None

    # Check if it's a disclaimer/note with a title appended at the end.
    # These lines contain "Italicized", "Racers in italics", or "NOTE" and end
    # with a sentence followed by the puzzle title.
    disclaimer_patterns = [
        r"(?:Italicized|italicized|Racers in italics)",
        r"\*{2,3}NOTE:",
        r"\*{2,3}NOTE\b",
    ]
    for pattern in disclaimer_patterns:
        if re.search(pattern, next_line):
            # Title appears after the last period (and optional closing **)
            # e.g. "...ineligible for prizes. The Zoo"
            # e.g. "**...automated.** Cabin in the Mountains"
            # Strip trailing/leading markdown bold markers
            cleaned = re.sub(r"\*{2,3}", "", next_line).strip()
            # Find the last sentence-ending period and take everything after it
            last_period = cleaned.rfind(".")
            if last_period >= 0 and last_period < len(cleaned) - 1:
                title = cleaned[last_period + 1:].strip()
                if title:
                    return title
            return None

    # Clean title on its own line — but skip if it looks like a URL or is too long
    if "http" in next_line.lower():
        return None
    if len(next_line) > 100:
        return None

    return next_line


def extract_table_from_text(pdf_path: Path) -> Optional[PDFExtractResult]:
    """
    Extract data from PDF using raw text parsing.

    This is a fallback method when table extraction fails. It works by:
    1. Extracting all text from the PDF
    2. Parsing the header for event number and puzzle info
    3. Parsing each data line using regex

    Text format example:
        Speed Puzzling #263 Results
        Boardwalk 500 piece
        Backyard Heroes
        ...
        1 100 1000 159,134 5 Black Kautz, Lauren 0:41:11 12.14 Minnesota 1
        2 100 999 18,773 213 Black DeLaat, Conner 0:42:24 11.79 Texas 1
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract PDF creation date from metadata
            event_date = None
            raw_date = (pdf.metadata or {}).get("CreationDate", "")
            if raw_date:
                # Format: D:YYYYMMDDHHMMSS+TZ'TZ' or D:YYYYMMDDHHMMSS
                date_match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(raw_date))
                if date_match:
                    event_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text() or ""
                all_text += "\n"

            lines = all_text.strip().split("\n")
            if not lines:
                return None

            # Extract metadata from header lines
            puzzle_pieces = None
            puzzle_brand = None
            puzzle_name = None
            event_number = None
            division = None

            for idx, line in enumerate(lines[:15]):  # Check first 15 lines for header info
                # Event number: "Speed Puzzling #263 Results"
                event_match = re.search(r"Speed Puzzling #(\d+)", line, re.IGNORECASE)
                if event_match:
                    event_number = int(event_match.group(1))

                # Division: "(Zoom Jigsaw Puzzle Contest--Solo Division)" or "Solo Puzzle Pyramid"
                if division is None:
                    div_match = re.search(r"\b(Solo|Pair|Team|Junior|Individual|Pairs|Teams)\b", line, re.IGNORECASE)
                    if div_match:
                        div_raw = div_match.group(1).lower()
                        # Normalize: individual→solo, pairs→pair, teams→team
                        division = {
                            "individual": "solo",
                            "pairs": "pair",
                            "teams": "team",
                        }.get(div_raw, div_raw)

                # Strip URL prefix if present - puzzle info comes AFTER the URL on same line
                # Example: "https://www.facebook.com/groups/SpeedPuzzling/ Boardwalk 500 piece"
                search_line = line
                if "http" in line.lower() or "www." in line.lower():
                    # Find the last URL-like segment and take everything after it
                    # URLs end with a space before the puzzle info
                    url_match = re.search(r"https?://\S+\s+(.+)", line)
                    if url_match:
                        search_line = url_match.group(1)
                    else:
                        continue  # Line is just a URL with nothing after

                # Puzzle info patterns:
                # - "Boardwalk 500 piece" or "Something 1000 Piece"
                # - "MasterPieces 300p" (shorthand at end of line)
                # - "Ravensburger 500pc"
                # - "Boardwalk 210p Haunted House" (relay: title on same line)
                # Must NOT match header lines like "Cruise25 Solo 300p Division"
                # or "February 500 piece Solo Results"
                if re.search(r"\bDivision\b|Speed Puzzling\s*#|\bResults\b", search_line):
                    continue
                piece_match = re.search(
                    r"(.+?)\s+(\d+)\s*(?:[Pp]ieces?\b|[Pp]c\b|[Pp])(?:\s+(.+))?$",
                    search_line,
                )
                if piece_match and puzzle_pieces is None:  # Only take first match
                    puzzle_brand = piece_match.group(1).strip()
                    puzzle_pieces = int(piece_match.group(2))

                    # Some formats have the title on the same line after piece count
                    # e.g. "Boardwalk 210p Haunted House" (relay events)
                    inline_title = piece_match.group(3)
                    if inline_title:
                        puzzle_name = inline_title.strip()
                        break

                    # Look at following lines for the actual puzzle title.
                    # The title is usually on the very next line, but in older PDFs
                    # there may be a URL-only line in between, or the NOTE text may
                    # be split across lines with the title sandwiched inside.
                    for lookahead in range(1, 5):
                        if idx + lookahead >= len(lines):
                            break
                        candidate = lines[idx + lookahead].strip()
                        # Skip blank lines and URL-only lines
                        if not candidate:
                            continue
                        if re.match(r"^https?://", candidate) and len(candidate.split()) <= 1:
                            continue
                        # Stop if we hit a data/header line (table start)
                        if re.match(r"^(Rank\s|%ile\s|This Contest|\d+\s+\d)", candidate):
                            break
                        # Skip continuation of a broken NOTE (e.g. "automated.***")
                        if re.match(r"^(automated|reported)\b", candidate, re.IGNORECASE):
                            continue
                        puzzle_name = _extract_puzzle_title(candidate)
                        if puzzle_name:
                            break

            # Parse data lines
            # Pattern matches: rank numbers... [belt color] Name, First time...
            # Example: "1 100 1000 159,134 5 Black Kautz, Lauren 0:41:11 12.14 Minnesota 1"
            results = []
            belt_colors = r"(?:Black|Blue|Green|Red|White|Yellow|Orange|Purple|Pink|Gray|Silver|Gold|Brown)"

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Line must start with rank number and contain time
                if not re.match(r"^\d+\s", line):
                    continue

                time_match = re.search(r"(\d+:\d{2}:\d{2})", line)
                if not time_match:
                    continue

                # Extract rank (first number)
                rank_match = re.match(r"^(\d+)", line)
                if not rank_match:
                    continue
                rank = int(rank_match.group(1))

                time_str = time_match.group(1)

                # Get the text between rank stuff and time for name extraction
                # This part has: numbers, belt color, name
                pre_time = line[:time_match.start()].strip()

                # Find name (Last, First pattern) - may have belt color prefix
                name_pattern = rf"(?:{belt_colors}\s+)?([A-Za-z][A-Za-z'\-\s]+,\s*[A-Za-z][A-Za-z'\-\s\.]*)"
                name_match = re.search(name_pattern, pre_time)

                if name_match:
                    name = name_match.group(1).strip()
                else:
                    # Fallback: look for comma-separated name anywhere
                    fallback_match = re.search(r"([A-Za-z][A-Za-z'\-]+,\s*[A-Za-z][A-Za-z'\-\.]+)", pre_time)
                    if fallback_match:
                        name = fallback_match.group(1).strip()
                    else:
                        continue  # Can't find a valid name

                # Try to extract location from after time
                post_time = line[time_match.end():].strip()
                # Location is typically after the speed score: "12.14 Minnesota 1"
                loc_match = re.search(r"[\d\.]+\s+([A-Za-z][A-Za-z\s]+?)(?:\s+\d+)?$", post_time)
                location = loc_match.group(1).strip() if loc_match else None

                results.append({
                    "rank": rank,
                    "name": name,
                    "time_str": time_str,
                    "location": location,
                })

            if not results:
                return None

            return PDFExtractResult(
                df=pd.DataFrame(results),
                puzzle_pieces=puzzle_pieces,
                puzzle_name=puzzle_name,
                puzzle_brand=puzzle_brand,
                event_number=event_number,
                division=division,
                event_date=event_date,
            )

    except Exception as e:
        print(f"  Text extraction error for {pdf_path.name}: {e}")
        return None


def extract_table_from_pdf(pdf_path: Path) -> Optional[PDFExtractResult]:
    """
    Extract tabular data from a PDF using hybrid approach.

    1. First tries text extraction (more reliable, provides metadata)
    2. Falls back to table extraction if text fails

    Text extraction is preferred because:
    - It reliably captures all rows across all pages
    - It can extract metadata (event number, puzzle name/pieces) from headers
    - pdfplumber's table detection sometimes misses data on multi-page PDFs
    """
    # Try text extraction first (more reliable)
    text_result = extract_table_from_text(pdf_path)
    if text_result is not None and text_result.df is not None and len(text_result.df) > 0:
        return text_result

    # Fall back to table extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract PDF creation date from metadata
            event_date = None
            raw_date = (pdf.metadata or {}).get("CreationDate", "")
            if raw_date:
                date_match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(raw_date))
                if date_match:
                    event_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

            all_results = []

            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Look for the main results table (has rank/name columns)
                    header_row = table[0]
                    if not header_row or len(header_row) < 2:
                        continue

                    # Check if this looks like a results table
                    header_text = " ".join(str(h or "") for h in header_row).lower()
                    if "name" not in header_text and "time" not in header_text:
                        continue

                    # Data row contains newline-separated entries
                    data_row = table[1] if len(table) > 1 else None
                    if not data_row:
                        continue

                    # Split each column by newlines
                    columns = []
                    for cell in data_row:
                        if cell:
                            lines = [l.strip() for l in str(cell).split("\n") if l.strip()]
                            columns.append(lines)
                        else:
                            columns.append([])

                    if len(columns) < 2:
                        continue

                    # Find the column with names/times (look for the one with time patterns)
                    name_col_idx = None
                    for idx, col in enumerate(columns):
                        if col and any(re.search(r"\d+:\d{2}:\d{2}", line) for line in col[:5]):
                            name_col_idx = idx
                            break

                    if name_col_idx is None:
                        continue

                    rank_col_idx = 0
                    # Location is typically last or after name
                    loc_col_idx = name_col_idx + 1 if name_col_idx + 1 < len(columns) else None

                    # Align rows - use the name column as reference
                    num_rows = len(columns[name_col_idx])

                    for i in range(num_rows):
                        try:
                            # Parse rank column: "1 100 400" -> rank=1
                            rank_parts = columns[rank_col_idx][i].split() if i < len(columns[rank_col_idx]) else []
                            rank = int(rank_parts[0]) if rank_parts and rank_parts[0].isdigit() else i + 1

                            # Parse name/time column
                            # Formats vary:
                            # - "Richardson, Leslie 0:13:55 7.18" (simple)
                            # - "15,938 103 Green Wood, Rachel 0:13:55 7.18" (with all-star rank)
                            name_time = columns[name_col_idx][i] if i < len(columns[name_col_idx]) else ""

                            # Extract time first (always H:M:S format at end)
                            time_match = re.search(r"(\d+:\d{2}:\d{2})", name_time)
                            if not time_match:
                                continue
                            time_str = time_match.group(1)

                            # Get the part before the time
                            pre_time = name_time[:time_match.start()].strip()

                            # Name is typically "Last, First" - find it by looking for comma
                            # Strip out leading numbers, belt colors (Green, Blue, Black, Red, etc.)
                            # Pattern: optional [numbers] [numbers] [Color] Name, First
                            name_match = re.search(
                                r"(?:[\d,]+\s+)?(?:\d+\s+)?(?:Green|Blue|Black|Red|White|Yellow|Orange|Purple|Pink|Gray|Silver|Gold|Brown)?\s*"
                                r"([A-Za-z][A-Za-z'\-\s]+,\s*[A-Za-z][A-Za-z'\-\s\.]*)",
                                pre_time
                            )
                            if name_match:
                                name = name_match.group(1).strip()
                            else:
                                # Fallback: take the last part that looks like a name
                                # (letters with possible comma)
                                parts = pre_time.split()
                                name_parts = []
                                for part in reversed(parts):
                                    if re.match(r"^[A-Za-z'\-,\.]+$", part):
                                        name_parts.insert(0, part)
                                    elif name_parts:
                                        break
                                name = " ".join(name_parts) if name_parts else pre_time

                            # Parse location if available
                            location = None
                            if loc_col_idx is not None and i < len(columns[loc_col_idx]):
                                loc_parts = columns[loc_col_idx][i].rsplit(None, 1)
                                location = loc_parts[0] if loc_parts else None

                            all_results.append({
                                "rank": rank,
                                "name": name,
                                "time_str": time_str,
                                "location": location,
                            })

                        except (ValueError, IndexError):
                            continue

            if not all_results:
                return None

            return PDFExtractResult(df=pd.DataFrame(all_results), event_date=event_date)

    except Exception as e:
        print(f"  Table extraction error for {pdf_path.name}: {e}")
        return None


def normalize_results(extract_result: PDFExtractResult, pdf_info: PDFInfo) -> pd.DataFrame:
    """
    Normalize extracted table data to standard schema.

    Input PDFExtractResult contains:
    - df: DataFrame with columns: rank, name, time_str, location
    - puzzle_pieces, puzzle_name, event_number (from text extraction)
    Output schema matches usajigsaw format.
    """
    if extract_result is None or extract_result.df is None or extract_result.df.empty:
        return pd.DataFrame()

    df = extract_result.df

    # Use extracted metadata if available, fall back to pdf_info
    puzzle_pieces = extract_result.puzzle_pieces
    puzzle_brand = extract_result.puzzle_brand
    puzzle_name = extract_result.puzzle_name
    event_id = pdf_info.event_id

    # If we extracted event number from PDF text, use it
    if extract_result.event_number:
        event_id = f"sp_{extract_result.event_number}"

    # Use extracted division, fall back to pdf_info, normalize values
    division = extract_result.division or pdf_info.division
    if division:
        # Normalize division values
        division = {
            "individual": "solo",
            "pairs": "pair",
            "teams": "team",
        }.get(division, division)

    results = []
    for _, row in df.iterrows():
        try:
            rank = int(row.get("rank", 0))
            name = str(row.get("name", "")).strip()
            time_str = str(row.get("time_str", ""))

            if not name or rank == 0:
                continue

            time_seconds, completed, pieces = parse_time(time_str)

            results.append({
                "source": "speedpuzzling",
                "event_id": event_id,
                "year": pdf_info.year,
                "division": division,
                "round": pdf_info.round,
                "rank": rank,
                "competitor_name": name,
                "origin": row.get("location"),
                "time_seconds": time_seconds,
                "completed": completed,
                "pieces_completed": pieces,
                "puzzle_pieces": puzzle_pieces,
                "puzzle_brand": puzzle_brand,
                "puzzle_name": puzzle_name,
                "finished_date": extract_result.event_date,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def parse_time(time_str: str) -> tuple[Optional[int], bool, Optional[int]]:
    """Parse time string to seconds."""
    time_str = str(time_str).strip().upper()

    # DNF/incomplete
    if "DNF" in time_str or "DID NOT" in time_str:
        return (None, False, None)

    # Try HH:MM:SS or MM:SS format
    match = re.match(r"(\d+):(\d{2}):(\d{2})", time_str)
    if match:
        h, m, s = map(int, match.groups())
        return (h * 3600 + m * 60 + s, True, None)

    match = re.match(r"(\d+):(\d{2})", time_str)
    if match:
        m, s = map(int, match.groups())
        return (m * 60 + s, True, None)

    return (None, False, None)


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "data" / "raw" / "pdfs"
    output_dir = project_root / "data" / "processed"

    # Step 1: Discover PDFs
    print("Step 1: Discovering PDFs...")
    pdfs = discover_all_pdfs()
    print(f"\nFound {len(pdfs)} unique PDFs\n")

    if not pdfs:
        print("No PDFs found!")
        sys.exit(1)

    # Step 2: Download PDFs
    print("Step 2: Downloading PDFs...")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded = download_all_pdfs(pdfs, pdf_dir)
    print(f"\nDownloaded {downloaded}/{len(pdfs)} PDFs\n")

    # Step 3: Extract data from PDFs
    print("Step 3: Extracting data from PDFs...")
    all_results = []
    success_count = 0

    for i, pdf in enumerate(pdfs):
        if pdf.local_path and pdf.local_path.exists():
            extract_result = extract_table_from_pdf(pdf.local_path)
            if extract_result is not None and extract_result.df is not None and not extract_result.df.empty:
                normalized = normalize_results(extract_result, pdf)
                if not normalized.empty:
                    all_results.append(normalized)
                    success_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(pdfs)} ({success_count} successful)")

    print(f"\nSuccessfully extracted data from {success_count} PDFs")

    if not all_results:
        print("No data extracted!")
        sys.exit(1)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "speedpuzzling_results.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} results to {output_path}")
    print("\nSummary by year:")
    print(combined.groupby("year").size())
