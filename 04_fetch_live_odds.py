"""
Scrape upcoming NRL fixtures and odds from OddsPortal using Playwright.

Outputs: data/live_odds.csv — one row per upcoming match with:
  date, home_team, away_team, venue,
  home_odds_close, away_odds_close,
  home_line_close, total_line_close

Run: python3 04_fetch_live_odds.py
"""

import json
import re
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

DATA_DIR = Path(__file__).parent / "data"

ODDSPORTAL_URL = "https://www.oddsportal.com/rugby-league/australia/nrl/"

TEAM_MAP = {
    "Canterbury Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Cronulla Sharks": "Cronulla-Sutherland Sharks",
    "Manly Sea Eagles": "Manly-Warringah Sea Eagles",
    "North QLD Cowboys": "North Queensland Cowboys",
    "St George Dragons": "St. George Illawarra Dragons",
    "Brisbane": "Brisbane Broncos",
    "Canterbury": "Canterbury-Bankstown Bulldogs",
    "Cronulla": "Cronulla-Sutherland Sharks",
    "Manly": "Manly-Warringah Sea Eagles",
    "Melbourne": "Melbourne Storm",
    "Newcastle": "Newcastle Knights",
    "Parramatta": "Parramatta Eels",
    "Penrith": "Penrith Panthers",
    "Rabbitohs": "South Sydney Rabbitohs",
    "Roosters": "Sydney Roosters",
    "Wests": "Wests Tigers",
    "Warriors": "New Zealand Warriors",
    "Raiders": "Canberra Raiders",
    "Titans": "Gold Coast Titans",
    "Cowboys": "North Queensland Cowboys",
    "NQ Cowboys": "North Queensland Cowboys",
}


def normalise(name: str) -> str:
    name = name.strip()
    return TEAM_MAP.get(name, name)


def random_delay(lo=1.5, hi=3.5):
    time.sleep(random.uniform(lo, hi))


def scrape_upcoming_matches(page) -> list[dict]:
    """Navigate to OddsPortal NRL page and extract upcoming fixtures with H2H odds."""
    print(f"  Navigating to {ODDSPORTAL_URL}")
    page.goto(ODDSPORTAL_URL, wait_until="domcontentloaded", timeout=30000)
    random_delay(2, 4)

    # Accept cookie banner if present
    try:
        page.click("button:has-text('Accept')", timeout=4000)
        random_delay(0.5, 1)
    except PlaywrightTimeout:
        pass

    # Wait for participant names to render
    try:
        page.wait_for_selector("p.participant-name", timeout=15000)
    except PlaywrightTimeout:
        print("  Warning: participant names did not appear — page may be slow")

    random_delay(1, 2)
    html = page.content()
    return parse_h2h_odds(html)


def parse_h2h_odds(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # Step 1: get fixture metadata (date, venue) from JSON-LD schema blocks
    fixture_meta = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(script.string or "")
            if "SportsEvent" not in (d.get("@type") or []):
                continue
            name = d.get("name", "")
            parts = name.split(" - ", 1)
            if len(parts) != 2:
                continue
            home_raw, away_raw = parts[0].strip(), parts[1].strip()
            fixture_meta[(home_raw, away_raw)] = {
                "date": (d.get("startDate") or "")[:10],
                "venue": d.get("location", {}).get("name", ""),
            }
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue

    # Step 2: extract odds from rendered div.group rows
    matches = []
    seen = set()
    for row in soup.find_all("div", class_="group"):
        names = [p.get_text(strip=True) for p in row.find_all("p", class_="participant-name")]
        if len(names) != 2:
            continue
        key = (names[0], names[1])
        if key in seen:
            continue
        seen.add(key)

        text = row.get_text(" ", strip=True)
        odds_vals = re.findall(r'\b([1-9]\d{0,2}\.\d{2})\b', text)
        home_odds = _safe_float(odds_vals[0]) if len(odds_vals) > 0 else None
        away_odds = _safe_float(odds_vals[-1]) if len(odds_vals) > 1 else None

        # Look up fixture metadata — try both orderings
        meta = fixture_meta.get(key) or fixture_meta.get((names[1], names[0])) or {}

        matches.append({
            "date": meta.get("date", ""),
            "venue": meta.get("venue", ""),
            "home_team": normalise(names[0]),
            "away_team": normalise(names[1]),
            "home_odds_close": home_odds,
            "away_odds_close": away_odds,
            "home_line_close": None,
            "total_line_close": None,
        })

    return matches



def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def scrape(headless: bool = True) -> pd.DataFrame:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-AU",
        )
        page = ctx.new_page()

        print("Scraping OddsPortal NRL fixtures...")
        matches = scrape_upcoming_matches(page)
        print(f"  Found {len(matches)} upcoming matches")

        browser.close()

    if not matches:
        print("No matches found. OddsPortal may have changed its layout.")
        return pd.DataFrame()

    df = pd.DataFrame(matches)
    df = df[df["home_team"].notna() & df["away_team"].notna()]
    df = df.drop_duplicates(subset=["home_team", "away_team"])
    return df


def main():
    df = scrape(headless=True)

    if df.empty:
        print("No data scraped.")
        return

    out = DATA_DIR / "live_odds.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} upcoming fixtures → {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
