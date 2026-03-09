"""Parse 2026 USA Jigsaw Nationals HTML heat assignments into JSON.

Reads saved Google Sheets HTML files from tournament/ and produces
tournament/nationals_2026.json with structured heat/pair/team data.

Usage: uv run python scripts/parse_tournament.py
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

TOURNAMENT_DIR = Path("tournament")
OUTPUT = TOURNAMENT_DIR / "nationals_2026.json"


class TableParser(HTMLParser):
    """Extract cell text from Google Sheets HTML export."""

    def __init__(self):
        super().__init__()
        self.in_td = False
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_td = True
            self.current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_td = False
            self.current_row.append(self.current_cell.strip())
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_td:
            self.current_cell += data


def parse_sheet(filename: str) -> list[list[str]]:
    """Parse a Google Sheets HTML export, returning data rows."""
    path = TOURNAMENT_DIR / f"{filename}_files" / "sheet.html"
    parser = TableParser()
    parser.feed(path.read_text())
    return parser.rows


def flip_name(name: str) -> str:
    """Convert 'First Last' to 'Last, First'.

    Strips parenthetical content like '(North Carolina)' first.
    Simple heuristic: last whitespace-delimited token is the last name.
    Handles hyphenated last names (e.g., 'Abigail Ricketts-Savit').
    """
    name = name.strip()
    if not name or "," in name:
        return name
    # Strip parenthetical content (e.g., state/origin)
    name = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
    parts = name.split()
    if len(parts) < 2:
        return name
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def parse_solo_heats(rows: list[list[str]]) -> dict[str, list[str]]:
    """Parse master list into {heat: [names]} for solo competition."""
    heats: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    for row in rows:
        if len(row) < 5:
            continue
        # Row number in col 0, name in col 1, solo heat in col 2
        if not row[0].isdigit() or int(row[0]) < 3:
            continue
        name = row[1].strip()
        heat = row[2].strip()
        if heat in heats and name:
            heats[heat].append(flip_name(name))
    return heats


def parse_pairs(rows: list[list[str]]) -> list[list[str]]:
    """Parse pairs sheet into list of [name1, name2] pairs."""
    pairs = []
    for row in rows:
        if len(row) < 4:
            continue
        if not row[0].isdigit() or int(row[0]) < 3:
            continue
        n1 = row[1].strip()
        n2 = row[2].strip()
        if n1 and n2:
            pairs.append([flip_name(n1), flip_name(n2)])
    return pairs


def parse_teams(rows: list[list[str]]) -> list[dict]:
    """Parse teams sheet into list of {name, members} dicts."""
    teams = []
    for row in rows:
        if len(row) < 7:
            continue
        if not row[0].isdigit() or int(row[0]) < 3:
            continue
        team_name = row[1].strip()
        members = [flip_name(row[i].strip()) for i in range(2, 6) if row[i].strip()]
        if len(members) == 4:
            teams.append({"name": team_name, "members": members})
    return teams


def main():
    print("Parsing tournament HTML files...")

    # Solo from master list
    master_rows = parse_sheet(
        "USA Jigsaw Nationals 2026 - Heats - Google Drive"
    )
    solo_heats = parse_solo_heats(master_rows)
    for h, names in sorted(solo_heats.items()):
        print(f"  Solo heat {h}: {len(names)} competitors")

    # Pairs
    pairs_a = parse_pairs(
        parse_sheet("USA Jigsaw Nationals 2026 - Heats - Google Drive - pairs prelim a")
    )
    pairs_b = parse_pairs(
        parse_sheet("USA Jigsaw Nationals 2026 - Heats - Google Drive - pairs prelim b")
    )
    print(f"  Pairs A: {len(pairs_a)} pairs")
    print(f"  Pairs B: {len(pairs_b)} pairs")

    # Teams
    teams_a = parse_teams(
        parse_sheet("USA Jigsaw Nationals 2026 - Heats - Google Drive - teams prelim a")
    )
    teams_b = parse_teams(
        parse_sheet("USA Jigsaw Nationals 2026 - Heats - Google Drive - teams prelim b")
    )
    print(f"  Teams A: {len(teams_a)} teams")
    print(f"  Teams B: {len(teams_b)} teams")

    result = {
        "solo": {
            "heats": solo_heats,
            "pieces": 500,
            "time_limit": 5400,
            "advance_n": 100,
            "final_pieces": 500,
            "final_time_limit": 5400,
        },
        "pairs": {
            "heats": {"A": pairs_a, "B": pairs_b},
            "pieces": 500,
            "time_limit": 4500,
            "advance_n": 100,
            "final_pieces": 500,
            "final_time_limit": 4500,
        },
        "teams": {
            "heats": {"A": teams_a, "B": teams_b},
            "prelim_puzzles": [{"pieces": 500}, {"pieces": 1000}],
            "final_puzzles": [{"pieces": 1000}, {"pieces": 1000}],
            "prelim_time_limit": 8100,
            "final_time_limit": 10800,
            "advance_n": 50,
        },
    }

    OUTPUT.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
