#!/usr/bin/env python3
"""
Scraper for myspeedpuzzling.com
Extracts puzzles, solving times, players, and competitions from public HTML pages.
Also downloads puzzle images for later analysis.

Usage:
    uv run python -m scraper.myspeedpuzzling                  # scrape data + download images
    uv run python -m scraper.myspeedpuzzling --no-images      # scrape data only (skip image downloads)
    uv run python -m scraper.myspeedpuzzling --duo-group      # fetch only duo/group tabs for existing puzzles

Output: data/raw/myspeedpuzzling/ directory with CSV/JSON files and images/ subdirectory.
"""

import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://myspeedpuzzling.com"
LOCALE = "/en"
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "myspeedpuzzling"
IMAGES_DIR = OUTPUT_DIR / "images"
REQUEST_DELAY = 1.5  # seconds between requests (be polite)
IMAGE_DELAY = 0.5  # shorter delay for image downloads (static assets)
REQUEST_TIMEOUT = 30
USER_AGENT = "MySpeedPuzzling-Research-Scraper/1.0 (academic research)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Puzzle:
    puzzle_id: str
    name: str
    alternative_name: str = ""
    manufacturer: str = ""
    pieces_count: int = 0
    ean: str = ""
    identification_number: str = ""
    image_url: str = ""
    image_local: str = ""  # local filename of downloaded image
    solved_times_count: int = 0
    average_time_seconds: Optional[int] = None


@dataclass
class SolvingTime:
    puzzle_id: str
    player_name: str
    player_code: str
    player_id: str = ""
    player_country: str = ""
    time_seconds: Optional[int] = None
    finished_date: str = ""
    ppm: str = ""
    first_attempt: bool = False
    unboxed: bool = False
    competition_name: str = ""
    category: str = "solo"  # solo / duo / group
    team_members: str = ""  # comma-separated names for duo/group


@dataclass
class Player:
    player_id: str
    name: str
    code: str = ""
    country: str = ""
    city: str = ""
    bio: str = ""
    has_membership: bool = False


@dataclass
class Competition:
    name: str
    slug: str = ""
    date_from: str = ""
    date_to: str = ""
    location: str = ""
    country: str = ""
    link: str = ""


# ---------------------------------------------------------------------------
# HTTP helper — uses httpx to handle FrankenPHP's HTTP 103 Early Hints
# ---------------------------------------------------------------------------


class Fetcher:
    def __init__(self):
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT,
        )
        self._last_request_time = 0.0

    def get(self, url: str, delay: float = REQUEST_DELAY) -> Optional[httpx.Response]:
        elapsed = time.time() - self._last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        try:
            resp = self.client.get(url)
            self._last_request_time = time.time()
            if resp.status_code != 200:
                log.warning("HTTP %d for %s", resp.status_code, url)
                return None
            return resp
        except httpx.HTTPError as exc:
            log.error("Request failed for %s: %s", url, exc)
            return None

    def post_live_action(
        self, url: str, action_name: str, props: dict, args: dict,
    ) -> Optional[str]:
        """POST a Symfony UX Live Component action and return the HTML fragment."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        endpoint = f"{url}/{action_name}"
        payload = {"data": json.dumps({"props": props, "updated": {}, "args": args})}
        headers = {
            "Accept": "application/vnd.live-component+html",
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            resp = self.client.post(endpoint, data=payload, headers=headers)
            self._last_request_time = time.time()
            if resp.status_code != 200:
                log.warning("Live Component HTTP %d for %s", resp.status_code, endpoint)
                return None
            return resp.text
        except httpx.HTTPError as exc:
            log.error("Live Component request failed for %s: %s", endpoint, exc)
            return None

    def download_image(self, url: str, dest: Path) -> bool:
        """Download an image to disk. Returns True on success."""
        if dest.exists():
            return True  # already downloaded
        resp = self.get(url, delay=IMAGE_DELAY)
        if resp is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True


fetcher = Fetcher()

# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch_sitemap() -> list[str]:
    """Return puzzle detail URLs (with /en/ prefix) from the sitemap."""
    url = f"{BASE_URL}/sitemap.xml"
    resp = fetcher.get(url)
    if resp is None:
        log.error("Could not fetch sitemap")
        return []

    root = ElementTree.fromstring(resp.content)
    puzzle_urls: list[str] = []

    for loc in root.findall(".//sm:loc", NS):
        href = loc.text or ""
        # Sitemap has /puzzle/{uuid} without locale prefix
        if "/puzzle/" in href and "/marketplace-puzzle/" not in href:
            # Normalize to English locale URL
            puzzle_id = href.rstrip("/").split("/")[-1]
            puzzle_urls.append(f"{BASE_URL}{LOCALE}/puzzle/{puzzle_id}")

    log.info("Sitemap: found %d puzzle URLs", len(puzzle_urls))
    return puzzle_urls


def extract_puzzle_id_from_url(url: str) -> str:
    """Extract the UUID puzzle id from a puzzle detail URL."""
    parts = url.rstrip("/").split("/")
    return parts[-1]


# ---------------------------------------------------------------------------
# Puzzle detail page parsing
# ---------------------------------------------------------------------------

PIECES_RE = re.compile(r"(\d[\d\s]*)\s*(?:pieces|pcs|d\u00edlk\u016f|pi\u00e8ces|Teile|\u30d4\u30fc\u30b9)", re.IGNORECASE)
TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})")
DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
PPM_RE = re.compile(r"([\d.]+)")


def parse_time_to_seconds(text: str) -> Optional[int]:
    """Convert HH:MM:SS or H:MM:SS to total seconds."""
    m = TIME_RE.search(text)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mi * 60 + s


def parse_date(text: str) -> str:
    """Convert DD.MM.YYYY to YYYY-MM-DD."""
    m = DATE_RE.search(text)
    if not m:
        return ""
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"


def extract_country_from_flag(el: Optional[Tag]) -> str:
    """Extract 2-letter country code from fi fi-XX class."""
    if el is None:
        return ""
    for cls in el.get("class", []):
        if cls.startswith("fi-") and cls != "fi":
            return cls[3:].upper()
    return ""


TAB_COUNT_RE = re.compile(r"\((\d+)\)")


def _parse_tab_counts(soup: BeautifulSoup) -> dict[str, int]:
    """Parse duo/group tab counts from the PuzzleTimes Live Component buttons."""
    counts: dict[str, int] = {"solo": 0, "duo": 0, "group": 0}
    for btn in soup.select("button[data-live-category-param]"):
        category = btn.get("data-live-category-param", "").lower()
        if category in counts:
            m = TAB_COUNT_RE.search(btn.get_text())
            if m:
                counts[category] = int(m.group(1))
    return counts


def _extract_live_props(soup: BeautifulSoup) -> tuple[str, dict] | None:
    """Extract the Live Component URL and props from the PuzzleTimes div."""
    div = soup.select_one("div[data-live-name-value='PuzzleTimes']")
    if div is None:
        return None
    url = div.get("data-live-url-value", "")
    props_json = div.get("data-live-props-value", "")
    if not url or not props_json:
        return None
    try:
        props = json.loads(props_json)
    except json.JSONDecodeError:
        log.warning("Failed to parse Live Component props JSON")
        return None
    # URL may be relative
    if url.startswith("/"):
        url = BASE_URL + url
    return url, props


def scrape_puzzle_detail(
    url: str, puzzle_id: str, download_images: bool = True,
) -> tuple[Optional[Puzzle], list[SolvingTime]]:
    """Scrape a single puzzle detail page. Returns puzzle metadata and solving times."""
    resp = fetcher.get(url)
    if resp is None:
        return None, []

    soup = BeautifulSoup(resp.text, "html.parser")
    puzzle = Puzzle(puzzle_id=puzzle_id, name="")
    solving_times: list[SolvingTime] = []

    # --- Puzzle name ---
    h1 = soup.select_one("h1.h4")
    if h1:
        puzzle.name = h1.get_text(strip=True)

    # --- Pieces count ---
    for small in soup.select("small.fw-bold"):
        m = PIECES_RE.search(small.get_text())
        if m:
            puzzle.pieces_count = int(m.group(1).replace(" ", ""))
            break

    # --- Manufacturer & identification number ---
    mfr_div = soup.select_one("div.manufacturer-name")
    if mfr_div:
        texts = list(mfr_div.stripped_strings)
        if texts:
            puzzle.manufacturer = texts[0]
        id_small = mfr_div.select_one("small.text-muted")
        if id_small:
            puzzle.identification_number = id_small.get_text(strip=True)

    # --- EAN ---
    for small in soup.select("small.text-muted"):
        icon = small.select_one("i.bi-upc-scan")
        if icon:
            puzzle.ean = small.get_text(strip=True)
            break

    # --- Image ---
    gallery = soup.select_one("div.gallery img") or soup.select_one("img.puzzle-detail-image")
    if gallery:
        src = gallery.get("src", "")
        if src:
            puzzle.image_url = urljoin(BASE_URL, src)

    # Download image
    if download_images and puzzle.image_url:
        ext = _image_extension(puzzle.image_url)
        image_filename = f"{puzzle_id}{ext}"
        image_path = IMAGES_DIR / image_filename
        if fetcher.download_image(puzzle.image_url, image_path):
            puzzle.image_local = image_filename

    # --- Alternative name (from data attribute if present) ---
    item_div = soup.select_one("[data-puzzle-alternative-name]")
    if item_div:
        puzzle.alternative_name = item_div.get("data-puzzle-alternative-name", "")

    # --- Solving times from table ---
    # PuzzleTimes Live Component renders server-side with table.custom-table
    # Initial HTML contains the solo tab (default)
    tables = soup.select("table.custom-table")
    for table in tables:
        for row in table.select("tbody tr"):
            st = _parse_solving_time_row(row, puzzle_id, "solo")
            if st:
                solving_times.append(st)

    # Fetch duo/group tabs via Live Component POST if they have data
    tab_counts = _parse_tab_counts(soup)
    live_info = None
    for category in ("duo", "group"):
        if tab_counts.get(category, 0) > 0:
            if live_info is None:
                live_info = _extract_live_props(soup)
                if live_info is None:
                    log.warning("Puzzle %s has %s data but no Live Component props",
                                puzzle_id[:8], category)
                    break
            component_url, props = live_info
            log.info("  Fetching %s tab (%d entries)...", category, tab_counts[category])
            html = fetcher.post_live_action(
                component_url, "changeResultsCategory", props, {"category": category},
            )
            if html:
                frag = BeautifulSoup(html, "html.parser")
                for table in frag.select("table.custom-table"):
                    for row in table.select("tbody tr"):
                        st = _parse_solving_time_row(row, puzzle_id, category)
                        if st:
                            solving_times.append(st)

    puzzle.solved_times_count = len(solving_times)

    # --- Average time (rendered by PuzzleTimes component) ---
    for el in soup.find_all(string=re.compile(r"average|pr\u016fm\u011br", re.IGNORECASE)):
        parent = el.parent
        if parent:
            secs = parse_time_to_seconds(parent.get_text())
            if secs:
                puzzle.average_time_seconds = secs
                break

    return puzzle, solving_times


def _image_extension(url: str) -> str:
    """Guess file extension from image URL."""
    path = url.split("?")[0].split("#")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
        if path.lower().endswith(ext):
            return ext
    return ".jpg"


def _parse_solving_time_row(row: Tag, puzzle_id: str, category: str) -> Optional[SolvingTime]:
    """Parse a single table row from PuzzleTimes or LadderTable."""
    tds = row.select("td")
    if len(tds) < 3:
        return None

    st = SolvingTime(puzzle_id=puzzle_id, player_name="", player_code="", category=category)

    # --- Player info ---
    player_td = row.select_one("td.player-name")
    if player_td:
        # Collect team members: duo/group tabs use class="player-name-item" on
        # both <a> (registered) and <span> (unregistered). Solo tabs have plain
        # <a href="player-profile/..."> without that class.
        members: list[tuple[str, str]] = []
        name_items = player_td.select(".player-name-item")
        if name_items:
            for el in name_items:
                pname = el.get_text(strip=True)
                if el.name == "a" and "player-profile" in el.get("href", ""):
                    pid = el["href"].rstrip("/").split("/")[-1]
                    members.append((pid, pname))
                elif pname:
                    members.append(("", pname))
        else:
            # Solo rows: plain <a href="player-profile/..."> without .player-name-item
            for link in player_td.select("a[href*='player-profile']"):
                pid = link.get("href", "").rstrip("/").split("/")[-1]
                pname = link.get_text(strip=True)
                members.append((pid, pname))

        if members:
            st.player_id = members[0][0]
            st.player_name = members[0][1]
            if len(members) > 1:
                # Structured format: "Name1:uuid1,Name2:,Name3:uuid3"
                st.team_members = ",".join(
                    f"{name}:{pid}" for pid, name in members
                )
                if category == "solo":
                    st.category = "duo" if len(members) == 2 else "group"

        # Player code (e.g. #ABCD)
        for code_el in player_td.select("code, small"):
            code_text = code_el.get_text(strip=True)
            if code_text.startswith("#"):
                st.player_code = code_text
                break

        # If name starts with #, it's a code-only player
        if st.player_name.startswith("#") and not st.player_code:
            st.player_code = st.player_name
            st.player_name = ""

        # Country flag
        flag = player_td.select_one(".fi[class*='fi-']")
        st.player_country = extract_country_from_flag(flag)

    # --- Time and metadata ---
    time_td = row.select_one("td.with-ppm") or (tds[-1] if tds else None)
    if time_td:
        time_text = time_td.get_text()
        st.time_seconds = parse_time_to_seconds(time_text)
        st.finished_date = parse_date(time_text)

        # PPM — structure is <small><span class="text-muted">PPM</span> 11.14</small>
        for small_el in time_td.select("small"):
            small_text = small_el.get_text(strip=True)
            if "PPM" in small_text:
                ppm_match = PPM_RE.search(small_text.replace("PPM", "").strip())
                if ppm_match:
                    st.ppm = ppm_match.group(1)
                break

        # Badges
        for badge in time_td.select("span.badge"):
            badge_text = badge.get_text(strip=True).lower()
            if "1st" in badge_text or "first" in badge_text or "try" in badge_text:
                st.first_attempt = True
            elif "unbox" in badge_text:
                st.unboxed = True
            elif badge.select_one("i.bi-trophy-fill"):
                st.competition_name = badge.get_text(strip=True)

    return st if (st.player_name or st.player_code) else None


# ---------------------------------------------------------------------------
# Player profile parsing
# ---------------------------------------------------------------------------


def scrape_player_profile(player_id: str) -> Optional[Player]:
    """Scrape a player profile page."""
    url = f"{BASE_URL}{LOCALE}/player-profile/{player_id}"
    resp = fetcher.get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    player = Player(player_id=player_id, name="")

    name_el = soup.select_one("span.player-breadcrumb-name-text")
    if name_el:
        player.name = name_el.get_text(strip=True)

    code_el = soup.select_one("code.player-breadcrumb-code")
    if code_el:
        player.code = code_el.get_text(strip=True)

    flag_el = soup.select_one("span.player-breadcrumb-flag")
    player.country = extract_country_from_flag(flag_el)

    membership_badge = soup.select_one("a[href*='membership'] .badge.bg-primary")
    player.has_membership = membership_badge is not None

    # Info section: <p><b>Label:</b><br/>Value</p>
    info_section = soup.select_one("#info")
    if info_section:
        for p_tag in info_section.select("p"):
            bold = p_tag.select_one("b")
            if not bold:
                continue
            label = bold.get_text(strip=True).lower().rstrip(":")
            # Get text after <b> and <br/> tags
            value_parts = []
            for sibling in bold.next_siblings:
                if isinstance(sibling, Tag) and sibling.name == "br":
                    continue
                text = sibling.get_text(strip=True) if isinstance(sibling, Tag) else str(sibling).strip()
                if text and text != "Not filled":
                    value_parts.append(text)
            value = " ".join(value_parts).strip()

            if "location" in label:
                player.city = value
            elif "about" in label and value:
                player.bio = value

    return player


# ---------------------------------------------------------------------------
# Events parsing
# ---------------------------------------------------------------------------


def scrape_events() -> list[Competition]:
    """Scrape all events from the events page."""
    url = f"{BASE_URL}{LOCALE}/events"
    resp = fetcher.get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    competitions: list[Competition] = []

    for card in soup.select("div.card"):
        comp = Competition(name="")

        name_el = card.select_one("h3 a, h5 a, h3.h5 a")
        if name_el:
            comp.name = name_el.get_text(strip=True)
            href = name_el.get("href", "")
            comp.slug = href.rstrip("/").split("/")[-1]
        else:
            heading = card.select_one("h3, h5")
            if heading:
                comp.name = heading.get_text(strip=True)

        if not comp.name:
            continue

        flag = card.select_one(".fi[class*='fi-']")
        comp.country = extract_country_from_flag(flag)

        # Location: text node after the flag span
        flag_parent = flag.parent if flag else None
        if flag_parent:
            # Get direct text content excluding child elements' text
            location_parts = []
            for child in flag_parent.children:
                if isinstance(child, str):
                    t = child.strip()
                    if t:
                        location_parts.append(t)
                elif isinstance(child, Tag) and child.name not in ("span", "div", "small", "i"):
                    t = child.get_text(strip=True)
                    if t:
                        location_parts.append(t)
            comp.location = " ".join(location_parts).strip()

        # Dates: format varies — "14.-15.03." or "01.05.2025" or "14.03."
        cal_icon = card.select_one("i.ci-calendar")
        if cal_icon and cal_icon.parent:
            date_text = cal_icon.parent.get_text(strip=True)
            # Try full DD.MM.YYYY dates first
            full_dates = DATE_RE.findall(date_text)
            if full_dates:
                comp.date_from = f"{full_dates[0][2]}-{full_dates[0][1]}-{full_dates[0][0]}"
                if len(full_dates) > 1:
                    comp.date_to = f"{full_dates[1][2]}-{full_dates[1][1]}-{full_dates[1][0]}"
                else:
                    comp.date_to = comp.date_from
            else:
                # Partial dates like "14.-15.03." — store raw text
                # Strip relative time like "(in 8 days)"
                raw = re.sub(r"\(.*?\)", "", date_text).strip()
                comp.date_from = raw

        link_el = card.select_one("a.btn[href^='http']")
        if link_el:
            comp.link = link_el.get("href", "")

        competitions.append(comp)

    log.info("Events: found %d competitions", len(competitions))
    return competitions


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PUZZLE_FIELDS = [
    "puzzle_id", "name", "alternative_name", "manufacturer", "pieces_count",
    "ean", "identification_number", "image_url", "image_local",
    "solved_times_count", "average_time_seconds",
]

SOLVING_TIME_FIELDS = [
    "puzzle_id", "player_name", "player_code", "player_id", "player_country",
    "time_seconds", "finished_date", "ppm", "first_attempt", "unboxed",
    "competition_name", "category", "team_members",
]

PLAYER_FIELDS = [
    "player_id", "name", "code", "country", "city", "bio", "has_membership",
]

COMPETITION_FIELDS = [
    "name", "slug", "date_from", "date_to", "location", "country", "link",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    """Append rows to CSV, creating with header if file doesn't exist."""
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_scraped_ids(path: Path, id_field: str) -> set[str]:
    """Load already-scraped IDs from an existing CSV for resume support."""
    ids: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(id_field, "")
                if val:
                    ids.add(val)
    return ids


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def scrape_duo_group_only(url: str, puzzle_id: str) -> list[SolvingTime]:
    """Fetch only duo/group tabs for a puzzle, skipping solo and metadata."""
    resp = fetcher.get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tab_counts = _parse_tab_counts(soup)
    if tab_counts.get("duo", 0) == 0 and tab_counts.get("group", 0) == 0:
        return []

    live_info = _extract_live_props(soup)
    if live_info is None:
        log.warning("Puzzle %s has duo/group data but no Live Component props", puzzle_id[:8])
        return []

    component_url, props = live_info
    solving_times: list[SolvingTime] = []
    for category in ("duo", "group"):
        if tab_counts.get(category, 0) == 0:
            continue
        log.info("  Fetching %s tab (%d entries)...", category, tab_counts[category])
        html = fetcher.post_live_action(
            component_url, "changeResultsCategory", props, {"category": category},
        )
        if html:
            frag = BeautifulSoup(html, "html.parser")
            for table in frag.select("table.custom-table"):
                for row in table.select("tbody tr"):
                    st = _parse_solving_time_row(row, puzzle_id, category)
                    if st:
                        solving_times.append(st)
    return solving_times


def main():
    download_images = "--no-images" not in sys.argv
    duo_group_only = "--duo-group" in sys.argv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if download_images:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Image downloads enabled (use --no-images to skip)")

    puzzles_csv = OUTPUT_DIR / "puzzles.csv"
    times_csv = OUTPUT_DIR / "solving_times.csv"
    players_csv = OUTPUT_DIR / "players.csv"
    competitions_csv = OUTPUT_DIR / "competitions.csv"

    # --- Resume support: load already-scraped puzzle IDs ---
    scraped_puzzle_ids = load_scraped_ids(puzzles_csv, "puzzle_id")
    if scraped_puzzle_ids:
        log.info("Resuming: %d puzzles already scraped", len(scraped_puzzle_ids))

    # --- 1. Fetch sitemap for puzzle URLs ---
    log.info("Fetching sitemap...")
    puzzle_urls = fetch_sitemap()
    if not puzzle_urls:
        log.error("No puzzle URLs found. Exiting.")
        sys.exit(1)

    # --- 2. Scrape puzzle detail pages ---
    # Pre-load player IDs from existing solving_times.csv so resume
    # doesn't lose track of players discovered in previous runs
    discovered_player_ids: set[str] = load_scraped_ids(times_csv, "player_id")
    if discovered_player_ids:
        log.info("Loaded %d previously discovered player IDs", len(discovered_player_ids))

    new_puzzle_count = 0
    new_time_count = 0
    batch_puzzles: list[Puzzle] = []
    batch_times: list[SolvingTime] = []

    def _collect_player_ids(times: list[SolvingTime]):
        for t in times:
            if t.player_id:
                discovered_player_ids.add(t.player_id)
            if t.team_members:
                for member in t.team_members.split(","):
                    parts = member.rsplit(":", 1)
                    if len(parts) == 2 and parts[1]:
                        discovered_player_ids.add(parts[1])

    if duo_group_only:
        # --- Duo/group-only mode: re-visit already-scraped puzzles ---
        log.info("Duo/group mode: re-visiting %d puzzles for duo/group tabs only",
                 len(puzzle_urls))
        for i, url in enumerate(puzzle_urls):
            pid = extract_puzzle_id_from_url(url)
            log.info("[%d/%d] Checking duo/group for %s", i + 1, len(puzzle_urls), pid[:8])
            times = scrape_duo_group_only(url, pid)
            if times:
                new_time_count += len(times)
                batch_times.extend(times)
                _collect_player_ids(times)

            if len(batch_times) >= 200:
                _append_csv(times_csv, SOLVING_TIME_FIELDS, [asdict(t) for t in batch_times])
                log.info("Saved batch: %d duo/group times (total: %d)", len(batch_times), new_time_count)
                batch_times = []

        if batch_times:
            _append_csv(times_csv, SOLVING_TIME_FIELDS, [asdict(t) for t in batch_times])
    else:
        # --- Normal mode: scrape new puzzles ---
        for i, url in enumerate(puzzle_urls):
            pid = extract_puzzle_id_from_url(url)
            if pid in scraped_puzzle_ids:
                continue

            log.info("[%d/%d] Scraping puzzle %s", i + 1, len(puzzle_urls), pid[:8])
            puzzle, times = scrape_puzzle_detail(url, pid, download_images=download_images)

            if puzzle:
                new_puzzle_count += 1
                new_time_count += len(times)
                batch_puzzles.append(puzzle)
                batch_times.extend(times)
                _collect_player_ids(times)

            # Periodic save every 50 puzzles
            if len(batch_puzzles) >= 50:
                _append_csv(puzzles_csv, PUZZLE_FIELDS, [asdict(p) for p in batch_puzzles])
                _append_csv(times_csv, SOLVING_TIME_FIELDS, [asdict(t) for t in batch_times])
                log.info("Saved batch: %d puzzles, %d times (total new: %d puzzles)",
                         len(batch_puzzles), len(batch_times), new_puzzle_count)
                batch_puzzles = []
                batch_times = []

        # Save remaining batch
        if batch_puzzles:
            _append_csv(puzzles_csv, PUZZLE_FIELDS, [asdict(p) for p in batch_puzzles])
            _append_csv(times_csv, SOLVING_TIME_FIELDS, [asdict(t) for t in batch_times])

    # Note: Ladder pages use lazy-loaded Live Components (placeholder skeletons
    # in initial HTML, real data loaded via JavaScript). All solving times are
    # already captured from individual puzzle detail pages, so ladders are skipped.

    # --- 3. Scrape player profiles ---
    scraped_player_ids = load_scraped_ids(players_csv, "player_id")
    new_player_ids = discovered_player_ids - scraped_player_ids
    log.info("Discovered %d players, %d new to scrape", len(discovered_player_ids), len(new_player_ids))

    all_players: list[Player] = []
    batch_players: list[Player] = []
    for i, pid in enumerate(sorted(new_player_ids)):
        log.info("[%d/%d] Scraping player %s", i + 1, len(new_player_ids), pid[:8])
        player = scrape_player_profile(pid)
        if player:
            all_players.append(player)
            batch_players.append(player)

        if len(batch_players) >= 50:
            _append_csv(players_csv, PLAYER_FIELDS, [asdict(p) for p in batch_players])
            batch_players = []

    if batch_players:
        _append_csv(players_csv, PLAYER_FIELDS, [asdict(p) for p in batch_players])

    # --- 4. Scrape events ---
    log.info("Scraping events...")
    competitions = scrape_events()
    _write_csv(competitions_csv, COMPETITION_FIELDS, [asdict(c) for c in competitions])

    # --- 5. Summary ---
    log.info(
        "Done! This session: %d new puzzles, %d new solving times, %d new players, %d competitions",
        new_puzzle_count, new_time_count,
        len(all_players), len(competitions),
    )


if __name__ == "__main__":
    main()
