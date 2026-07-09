"""
Fetch the full 2026 NRL draw from nrl.com.

Scrapes all regular-season rounds (1-27) + finals using Playwright,
normalises team names to the canonical set used throughout this project,
and upserts fixtures into data/nrl.db and data/draw_2026.csv.

Usage:
  python3 10_fetch_draw_2026.py              # scrape all rounds
  python3 10_fetch_draw_2026.py --round 12   # single round (debug)
  python3 10_fetch_draw_2026.py --csv-only   # only write CSV, skip DB
"""

import argparse
import json
import re
import sqlite3
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data" / "nrl.db"
CSV_OUT  = BASE_DIR / "data" / "draw_2026.csv"

SEASON = 2026
REGULAR_ROUNDS = 27
FINALS_SLUGS   = [
    "finals-week-1",
    "finals-week-2",
    "finals-week-3",
    "finals-week-4",
]

# ── Canonical team names ──────────────────────────────────────────────────────
# All names normalised to match the set used in nrl_features.csv and models.

TEAM_MAP = {
    # NRL.com nicknames
    "Broncos":          "Brisbane Broncos",
    "Bulldogs":         "Canterbury-Bankstown Bulldogs",
    "Raiders":          "Canberra Raiders",
    "Sharks":           "Cronulla-Sutherland Sharks",
    "Dolphins":         "Dolphins",
    "Titans":           "Gold Coast Titans",
    "Sea Eagles":       "Manly-Warringah Sea Eagles",
    "Storm":            "Melbourne Storm",
    "Warriors":         "New Zealand Warriors",
    "Knights":          "Newcastle Knights",
    "Cowboys":          "North Queensland Cowboys",
    "Eels":             "Parramatta Eels",
    "Panthers":         "Penrith Panthers",
    "Rabbitohs":        "South Sydney Rabbitohs",
    "Dragons":          "St. George Illawarra Dragons",
    "Roosters":         "Sydney Roosters",
    "Tigers":           "Wests Tigers",
    # Full names with minor variants
    "Brisbane Broncos":              "Brisbane Broncos",
    "Canterbury Bulldogs":           "Canterbury-Bankstown Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Canberra Raiders":              "Canberra Raiders",
    "Cronulla Sharks":               "Cronulla-Sutherland Sharks",
    "Cronulla-Sutherland Sharks":    "Cronulla-Sutherland Sharks",
    "Redcliffe Dolphins":            "Dolphins",
    "Gold Coast Titans":             "Gold Coast Titans",
    "Manly Sea Eagles":              "Manly-Warringah Sea Eagles",
    "Manly-Warringah Sea Eagles":    "Manly-Warringah Sea Eagles",
    "Melbourne Storm":               "Melbourne Storm",
    "New Zealand Warriors":          "New Zealand Warriors",
    "NZ Warriors":                   "New Zealand Warriors",
    "Newcastle Knights":             "Newcastle Knights",
    "North Queensland Cowboys":      "North Queensland Cowboys",
    "North QLD Cowboys":             "North Queensland Cowboys",
    "NQ Cowboys":                    "North Queensland Cowboys",
    "Parramatta Eels":               "Parramatta Eels",
    "Penrith Panthers":              "Penrith Panthers",
    "South Sydney Rabbitohs":        "South Sydney Rabbitohs",
    "St George Illawarra Dragons":   "St. George Illawarra Dragons",
    "St. George Illawarra Dragons":  "St. George Illawarra Dragons",
    "St George Dragons":             "St. George Illawarra Dragons",
    "Sydney Roosters":               "Sydney Roosters",
    "Wests Tigers":                  "Wests Tigers",
    # Short/alternate OddsPortal names
    "Brisbane":    "Brisbane Broncos",
    "Canterbury":  "Canterbury-Bankstown Bulldogs",
    "Cronulla":    "Cronulla-Sutherland Sharks",
    "Gold Coast":  "Gold Coast Titans",
    "Manly":       "Manly-Warringah Sea Eagles",
    "Melbourne":   "Melbourne Storm",
    "Newcastle":   "Newcastle Knights",
    "Parramatta":  "Parramatta Eels",
    "Penrith":     "Penrith Panthers",
    "South Sydney":"South Sydney Rabbitohs",
    "Wests":       "Wests Tigers",
    "North Queensland": "North Queensland Cowboys",
}


def norm(name: str) -> str:
    name = name.strip()
    return TEAM_MAP.get(name, name)


# ── Playwright scraper ────────────────────────────────────────────────────────

def fetch_page(url: str) -> str:
    from playwright.sync_api import sync_playwright, TimeoutError as PWT

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # Intercept API responses that may contain draw data
        api_data = []
        def on_response(resp):
            url_lower = resp.url.lower()
            if any(kw in url_lower for kw in ("draw", "fixture", "match", "round", "schedule")):
                if resp.status == 200:
                    ct = resp.headers.get("content-type", "")
                    if "json" in ct:
                        try:
                            api_data.append(resp.json())
                        except Exception:
                            pass

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for match cards to render (JS takes ~3s)
            try:
                page.wait_for_selector("div.match", timeout=8000)
            except PWT:
                pass
            time.sleep(random.uniform(2, 3))
        except PWT:
            pass

        html = page.content()
        browser.close()

    return html, api_data


def parse_api_draw(api_data: list) -> list[dict]:
    """Try to extract fixture data from intercepted API JSON responses."""
    fixtures = []
    for data in api_data:
        # NRL API v3 format: {"fixtures": [...]} or {"draw": [...]}
        items = (data.get("fixtures") or data.get("draw") or
                 data.get("matches") or [])
        if not items and isinstance(data, list):
            items = data

        for item in items:
            if not isinstance(item, dict):
                continue
            # Team names — try various key patterns
            home_raw = (item.get("homeTeam", {}) or {})
            away_raw = (item.get("awayTeam", {}) or {})
            if isinstance(home_raw, dict):
                home_name = (home_raw.get("name") or home_raw.get("nickName")
                             or home_raw.get("teamName") or "")
                away_name = (away_raw.get("name") or away_raw.get("nickName")
                             or away_raw.get("teamName") or "")
            else:
                home_name = str(home_raw)
                away_name = str(away_raw)

            if not home_name or not away_name:
                continue

            venue_raw = item.get("venue") or item.get("stadium") or {}
            venue = (venue_raw.get("name") or venue_raw.get("stadiumName") or ""
                     if isinstance(venue_raw, dict) else str(venue_raw))

            kickoff = (item.get("kickOffTime") or item.get("kickoffTime")
                       or item.get("date") or item.get("startTime") or "")
            date_str = str(kickoff)[:10] if kickoff else ""

            round_raw = item.get("round") or item.get("roundNumber") or {}
            if isinstance(round_raw, dict):
                round_num  = round_raw.get("roundNumber") or round_raw.get("number")
                round_name = round_raw.get("name") or round_raw.get("roundTitle") or ""
            else:
                round_num  = round_raw
                round_name = f"Round {round_raw}" if round_raw else ""

            is_finals = 1 if (round_name and any(
                kw in str(round_name).lower() for kw in
                ("final", "semi", "qualifier", "eliminator", "preliminary", "grand")
            )) else 0

            fixtures.append({
                "date":       date_str,
                "season":     SEASON,
                "home_team":  norm(home_name),
                "away_team":  norm(away_name),
                "venue":      venue.strip(),
                "round":      round_num,
                "round_name": round_name,
                "is_finals":  is_finals,
            })

    return fixtures


def parse_nrl_date(date_str: str, season: int = SEASON) -> str:
    """
    Parse NRL draw date strings like "Sunday 1st March" or "Friday 6th March"
    into ISO format "YYYY-MM-DD".
    """
    import calendar
    # Strip ordinal suffixes: 1st → 1, 2nd → 2, etc.
    clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str.strip())
    # Remove weekday
    parts = clean.split()
    # Find the day number and month name
    month_names = {m: i for i, m in enumerate(calendar.month_name) if m}
    day_num = None
    month_num = None
    for part in parts:
        if part.isdigit():
            day_num = int(part)
        if part in month_names:
            month_num = month_names[part]
    if day_num and month_num:
        return f"{season}-{month_num:02d}-{day_num:02d}"
    return ""


def parse_html_draw(html: str, round_num: int, is_finals: bool = False) -> list[dict]:
    """
    Parse nrl.com round draw page.

    Structure (as of 2026):
      div.match
        a[href]
          p.match-header__title  → date "Sunday 1st March"
          p.match-team__name--home → home nickname
          p.match-team__name--away → away nickname
        div.match-venue-broadcasters
          p.match-venue           → venue name (may include city suffix)
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    fixtures = []

    round_name = (f"Finals Week {round_num - 27}" if is_finals
                  else f"Round {round_num}")

    match_cards = soup.find_all("div", class_="match")
    for card in match_cards:
        home_el = card.find("p", class_="match-team__name--home")
        away_el = card.find("p", class_="match-team__name--away")
        if not home_el or not away_el:
            continue

        home_nick = home_el.get_text(strip=True)
        away_nick = away_el.get_text(strip=True)
        home_name = norm(home_nick)
        away_name = norm(away_nick)

        title_el = card.find("p", class_="match-header__title")
        date_raw  = title_el.get_text(strip=True) if title_el else ""
        date_str  = parse_nrl_date(date_raw)

        venue_el = card.find("p", class_="match-venue")
        venue_raw = venue_el.get_text(separator=" ", strip=True) if venue_el else ""
        venue = re.sub(r'^Venue:\s*', '', venue_raw).strip()
        venue = venue.split(",")[0].strip()

        # Check if completed and extract scores
        status_el = card.find("span", class_="o-lozenge__topic")
        status    = status_el.get_text(strip=True) if status_el else ""
        is_completed = 1 if status in ("Full Time", "Final") else 0

        home_score = None
        away_score = None
        if is_completed:
            h_score_el = card.find("div", class_="match-team__score--home")
            a_score_el = card.find("div", class_="match-team__score--away")
            if h_score_el and a_score_el:
                h_nums = re.findall(r'\d+', h_score_el.get_text())
                a_nums = re.findall(r'\d+', a_score_el.get_text())
                if h_nums and a_nums:
                    home_score = int(h_nums[0])
                    away_score = int(a_nums[0])

        fixtures.append({
            "date":             date_str,
            "season":           SEASON,
            "home_team":        home_name,
            "away_team":        away_name,
            "venue":            venue,
            "round":            round_num,
            "round_name":       round_name,
            "is_finals":        1 if is_finals else 0,
            "is_completed":     is_completed,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
        })

    return fixtures


# ── DB helpers ────────────────────────────────────────────────────────────────

def migrate_db(conn):
    """Add round/is_finals columns if they don't exist yet."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fixtures)")}
    if "round" not in cols:
        conn.execute("ALTER TABLE fixtures ADD COLUMN round INTEGER")
    if "is_finals" not in cols:
        conn.execute("ALTER TABLE fixtures ADD COLUMN is_finals INTEGER DEFAULT 0")

    # Season simulation tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS season_simulation (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            simulated_at TEXT,
            model_version TEXT DEFAULT 'no_odds',
            team         TEXT,
            actual_wins  INTEGER DEFAULT 0,
            actual_losses INTEGER DEFAULT 0,
            proj_wins    INTEGER,
            proj_losses  INTEGER,
            proj_points  INTEGER,
            proj_for     INTEGER,
            proj_against INTEGER,
            proj_diff    INTEGER,
            proj_position INTEGER,
            UNIQUE(simulated_at, model_version, team)
        );

        CREATE TABLE IF NOT EXISTS finals_simulation (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            simulated_at TEXT,
            model_version TEXT DEFAULT 'no_odds',
            round_name   TEXT,
            match_order  INTEGER,
            home_team    TEXT,
            away_team    TEXT,
            pred_home_score INTEGER,
            pred_away_score INTEGER,
            pred_winner  TEXT,
            home_win_prob REAL,
            confidence   REAL
        );
    """)
    conn.commit()


def upsert_draw(conn, fixtures: list[dict]) -> int:
    now = datetime.utcnow().isoformat()
    inserted = 0
    for fx in fixtures:
        if not fx.get("home_team") or not fx.get("away_team"):
            continue
        conn.execute("""
            INSERT INTO fixtures
                (date, season, home_team, away_team, venue, round, is_finals,
                 is_completed, actual_home_score, actual_away_score, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, home_team, away_team) DO UPDATE SET
                venue              = excluded.venue,
                round              = COALESCE(excluded.round,     fixtures.round),
                is_finals          = COALESCE(excluded.is_finals, fixtures.is_finals),
                season             = COALESCE(excluded.season,    fixtures.season),
                is_completed       = CASE WHEN excluded.is_completed = 1
                                         THEN 1 ELSE fixtures.is_completed END,
                actual_home_score  = CASE WHEN excluded.actual_home_score IS NOT NULL
                                         THEN excluded.actual_home_score
                                         ELSE fixtures.actual_home_score END,
                actual_away_score  = CASE WHEN excluded.actual_away_score IS NOT NULL
                                         THEN excluded.actual_away_score
                                         ELSE fixtures.actual_away_score END,
                scraped_at         = excluded.scraped_at
        """, (
            fx.get("date") or "",
            fx.get("season", SEASON),
            fx["home_team"], fx["away_team"],
            fx.get("venue") or "",
            fx.get("round"),
            fx.get("is_finals", 0),
            fx.get("is_completed", 0),
            fx.get("actual_home_score"),
            fx.get("actual_away_score"),
            now,
        ))
        inserted += 1
    conn.commit()
    return inserted


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_round(round_slug: str, round_num: int, is_finals: bool = False) -> list[dict]:
    url = f"https://www.nrl.com/draw/nrl-premiership/{SEASON}/{round_slug}/"
    print(f"  Fetching {url}")
    try:
        html, api_data = fetch_page(url)
    except Exception as e:
        print(f"    Error: {e}")
        return []

    # Try API data first (more reliable)
    if api_data:
        fixtures = parse_api_draw(api_data)
        if fixtures:
            # Filter to this round only
            fixtures = [f for f in fixtures if not f.get("round") or f["round"] == round_num]
            if fixtures:
                print(f"    {len(fixtures)} fixtures from API")
                return fixtures

    # Fall back to HTML
    fixtures = parse_html_draw(html, round_num, is_finals)
    print(f"    {len(fixtures)} fixtures from HTML")
    return fixtures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None,
                        help="Fetch only a specific round (1-27)")
    parser.add_argument("--csv-only", action="store_true",
                        help="Write CSV only, skip DB")
    parser.add_argument("--finals-only", action="store_true",
                        help="Fetch only finals rounds")
    args = parser.parse_args()

    all_fixtures = []

    if args.round:
        rounds = [(f"round-{args.round}", args.round, False)]
    elif args.finals_only:
        rounds = [(slug, 27 + i + 1, True)
                  for i, slug in enumerate(FINALS_SLUGS)]
    else:
        rounds = [(f"round-{n}", n, False) for n in range(1, REGULAR_ROUNDS + 1)]
        rounds += [(slug, 27 + i + 1, True)
                   for i, slug in enumerate(FINALS_SLUGS)]

    print(f"\nFetching {len(rounds)} rounds for {SEASON} season...\n")
    for slug, num, is_finals in rounds:
        fx = scrape_round(slug, num, is_finals)
        all_fixtures.extend(fx)
        time.sleep(random.uniform(1.5, 3.0))  # polite delay

    # Deduplicate
    seen = set()
    unique = []
    for fx in all_fixtures:
        key = (fx["home_team"], fx["away_team"], fx.get("date", ""))
        if key not in seen:
            seen.add(key)
            unique.append(fx)

    print(f"\nTotal unique fixtures: {len(unique)}")

    if not unique:
        print("No fixtures found — check scraper or nrl.com structure.")
        return

    # Write CSV
    df = pd.DataFrame(unique)
    df = df.sort_values(["round", "date"]).reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved → {CSV_OUT}")

    # Write to DB
    if not args.csv_only and DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        migrate_db(conn)
        n = upsert_draw(conn, unique)
        conn.close()
        print(f"DB upserted {n} fixtures")
    elif not args.csv_only:
        print("DB not found — run 08_update_db.py first, then re-run this script to populate DB.")

    print("\nDone.")


if __name__ == "__main__":
    main()
