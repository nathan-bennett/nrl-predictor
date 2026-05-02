"""
Update the NRL predictions SQLite database.

Run daily (e.g. via cron) to:
  1. Scrape latest upcoming fixtures + odds from OddsPortal
  2. Generate predictions for any new fixture
  3. Back-fill actual scores for completed games (also from OddsPortal)
  4. Save everything to data/nrl.db

Cron example (9am daily):
  0 9 * * * cd ~/Development/nrl-predictor && python3 08_update_db.py

Usage:
  python3 08_update_db.py              # full update
  python3 08_update_db.py --results    # only fetch completed game results
  python3 08_update_db.py --predict    # only refresh predictions for upcoming games
"""

import argparse
import json
import re
import sqlite3
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data" / "nrl.db"

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS fixtures (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    season            INTEGER,
    home_team         TEXT NOT NULL,
    away_team         TEXT NOT NULL,
    venue             TEXT,
    home_odds         REAL,
    away_odds         REAL,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    is_completed      INTEGER DEFAULT 0,
    scraped_at        TEXT,
    UNIQUE(date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL REFERENCES fixtures(id),
    model_version   TEXT    DEFAULT 'v2',
    pred_home_score INTEGER,
    pred_away_score INTEGER,
    pred_margin     INTEGER,
    pred_total      INTEGER,
    winner          TEXT,
    home_win_prob   REAL,
    away_win_prob   REAL,
    confidence      REAL,
    created_at      TEXT,
    UNIQUE(fixture_id, model_version)
);

CREATE TABLE IF NOT EXISTS model_accuracy (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluated_at    TEXT,
    model_version   TEXT,
    n_games         INTEGER,
    winner_accuracy REAL,
    home_score_mae  REAL,
    away_score_mae  REAL,
    margin_mae      REAL
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Migrations — add columns introduced after initial schema
        existing = {row[1] for row in conn.execute("PRAGMA table_info(fixtures)")}
        if "round" not in existing:
            conn.execute("ALTER TABLE fixtures ADD COLUMN round INTEGER")
        if "is_finals" not in existing:
            conn.execute("ALTER TABLE fixtures ADD COLUMN is_finals INTEGER DEFAULT 0")
        # Season simulation tables (created by 10_fetch_draw_2026 / 11_simulate_season)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS monte_carlo_positions (
                simulated_at TEXT,
                team         TEXT,
                position     INTEGER,
                count        INTEGER DEFAULT 0,
                n_sims       INTEGER DEFAULT 100,
                PRIMARY KEY (simulated_at, team, position)
            );
            CREATE TABLE IF NOT EXISTS season_simulation (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                simulated_at  TEXT,
                model_version TEXT DEFAULT 'no_odds',
                team          TEXT,
                actual_wins   INTEGER DEFAULT 0,
                actual_losses INTEGER DEFAULT 0,
                proj_wins     INTEGER,
                proj_losses   INTEGER,
                proj_points   INTEGER,
                proj_for      INTEGER,
                proj_against  INTEGER,
                proj_diff     INTEGER,
                proj_position INTEGER,
                UNIQUE(simulated_at, model_version, team)
            );
            CREATE TABLE IF NOT EXISTS finals_simulation (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                simulated_at  TEXT,
                model_version TEXT DEFAULT 'no_odds',
                round_name    TEXT,
                match_order   INTEGER,
                home_team     TEXT,
                away_team     TEXT,
                pred_home_score INTEGER,
                pred_away_score INTEGER,
                pred_winner   TEXT,
                home_win_prob REAL,
                confidence    REAL
            );
        """)
        conn.commit()
    print(f"DB ready → {DB_PATH}")


# ── Scraping ──────────────────────────────────────────────────────────────────

def scrape_oddsportal(url: str) -> str:
    """Fetch OddsPortal page HTML using Playwright."""
    import random
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
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(random.uniform(3, 5))
        try:
            page.wait_for_selector("p.participant-name", timeout=12000)
        except PWT:
            pass
        html = page.content()
        browser.close()
    return html


TEAM_MAP = {
    "Canterbury Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Cronulla Sharks": "Cronulla-Sutherland Sharks",
    "Manly Sea Eagles": "Manly-Warringah Sea Eagles",
    "North QLD Cowboys": "North Queensland Cowboys",
    "NQ Cowboys": "North Queensland Cowboys",
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
}


def norm(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


def parse_fixtures_from_html(html: str) -> list[dict]:
    """Extract upcoming fixtures + odds from OddsPortal NRL page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Metadata from JSON-LD
    meta = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(script.string or "")
            if "SportsEvent" not in (d.get("@type") or []):
                continue
            parts = d.get("name", "").split(" - ", 1)
            if len(parts) != 2:
                continue
            key = (parts[0].strip(), parts[1].strip())
            meta[key] = {
                "date":  (d.get("startDate") or "")[:10],
                "venue": d.get("location", {}).get("name", ""),
            }
        except Exception:
            continue

    # Odds from div.group rows
    fixtures = []
    seen = set()
    for row in soup.find_all("div", class_="group"):
        names = [p.get_text(strip=True)
                 for p in row.find_all("p", class_="participant-name")]
        if len(names) != 2:
            continue
        key = tuple(names)
        if key in seen:
            continue
        seen.add(key)

        text  = row.get_text(" ", strip=True)
        odds  = re.findall(r'\b([1-9]\d{0,2}\.\d{2})\b', text)
        h_odd = float(odds[0])  if len(odds) > 0 else None
        a_odd = float(odds[-1]) if len(odds) > 1 else None
        m     = meta.get(key) or meta.get((names[1], names[0])) or {}

        fixtures.append({
            "date":       m.get("date", ""),
            "venue":      m.get("venue", ""),
            "home_team":  norm(names[0]),
            "away_team":  norm(names[1]),
            "home_odds":  h_odd,
            "away_odds":  a_odd,
        })
    return fixtures


def parse_results_from_html(html: str) -> list[dict]:
    """Extract completed game scores from OddsPortal results page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    results = []
    seen = set()

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(script.string or "")
            if "SportsEvent" not in (d.get("@type") or []):
                continue
            desc  = d.get("description", "")
            name  = d.get("name", "")
            parts = name.split(" - ", 1)
            if len(parts) != 2:
                continue
            home_raw, away_raw = parts[0].strip(), parts[1].strip()
            key = (home_raw, away_raw)
            if key in seen:
                continue
            seen.add(key)

            # Score sometimes in description: "Team A - Team B 24 - 18 ..."
            score_m = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\b', desc)
            if not score_m:
                continue
            results.append({
                "home_team":  norm(home_raw),
                "away_team":  norm(away_raw),
                "date":       (d.get("startDate") or "")[:10],
                "home_score": int(score_m.group(1)),
                "away_score": int(score_m.group(2)),
            })
        except Exception:
            continue
    return results


# ── Predictions ───────────────────────────────────────────────────────────────

def generate_predictions(fixtures: list[dict]) -> list[dict]:
    """Run the v2 predict pipeline on a list of fixture dicts."""
    import importlib.util, sys

    # Import predict module
    spec = importlib.util.spec_from_file_location(
        "predict", BASE_DIR / "05_predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pkgs        = mod.load_models()
    hist        = pd.read_csv(BASE_DIR / "data" / "nrl_features.csv",
                              parse_dates=["date"])
    team_stats  = mod.build_team_current_stats(hist)
    h2h_stats   = mod.build_h2h_stats(hist)

    preds = []
    for fx in fixtures:
        try:
            p = mod.predict_match(
                fx["home_team"], fx["away_team"],
                pkgs, team_stats, h2h_stats,
                home_odds=fx.get("home_odds"),
                away_odds=fx.get("away_odds"),
                venue=fx.get("venue"),
            )
            preds.append(p)
        except Exception as e:
            print(f"  Prediction failed for {fx['home_team']} vs {fx['away_team']}: {e}")
    return preds


# ── DB operations ─────────────────────────────────────────────────────────────

def upsert_fixtures(conn, fixtures: list[dict]) -> dict:
    """Insert new fixtures; return mapping (home_team, away_team, date) -> fixture_id."""
    now = datetime.utcnow().isoformat()
    ids = {}
    for fx in fixtures:
        cur = conn.execute("""
            INSERT INTO fixtures (date, season, home_team, away_team, venue,
                                  home_odds, away_odds, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, home_team, away_team) DO UPDATE SET
                venue      = excluded.venue,
                home_odds  = excluded.home_odds,
                away_odds  = excluded.away_odds,
                scraped_at = excluded.scraped_at
            RETURNING id
        """, (
            fx["date"],
            int(fx["date"][:4]) if fx.get("date") else None,
            fx["home_team"], fx["away_team"],
            fx.get("venue"), fx.get("home_odds"), fx.get("away_odds"),
            now,
        ))
        row = cur.fetchone()
        if row:
            ids[(fx["home_team"], fx["away_team"], fx["date"])] = row[0]
        else:
            row2 = conn.execute(
                "SELECT id FROM fixtures WHERE date=? AND home_team=? AND away_team=?",
                (fx["date"], fx["home_team"], fx["away_team"])
            ).fetchone()
            if row2:
                ids[(fx["home_team"], fx["away_team"], fx["date"])] = row2[0]
    conn.commit()
    return ids


def upsert_predictions(conn, preds: list[dict], fixture_ids: dict):
    """Save predictions to DB."""
    now = datetime.utcnow().isoformat()
    for p in preds:
        # Find fixture_id — match on team names (date may not be in prediction dict)
        fid = None
        for (ht, at, dt), fid_cand in fixture_ids.items():
            if ht == p["home_team"] and at == p["away_team"]:
                fid = fid_cand
                break
        if fid is None:
            continue
        conn.execute("""
            INSERT INTO predictions
                (fixture_id, model_version, pred_home_score, pred_away_score,
                 pred_margin, pred_total, winner, home_win_prob, away_win_prob,
                 confidence, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fixture_id, model_version) DO UPDATE SET
                pred_home_score = excluded.pred_home_score,
                pred_away_score = excluded.pred_away_score,
                pred_margin     = excluded.pred_margin,
                pred_total      = excluded.pred_total,
                winner          = excluded.winner,
                home_win_prob   = excluded.home_win_prob,
                away_win_prob   = excluded.away_win_prob,
                confidence      = excluded.confidence,
                created_at      = excluded.created_at
        """, (fid, "v2", p["pred_home"], p["pred_away"],
              p["pred_margin"], p["pred_total"], p["winner"],
              p["home_win_prob"], p["away_win_prob"], p["confidence"], now))
    conn.commit()


def update_results(conn, results: list[dict]):
    """Back-fill actual scores for completed games."""
    updated = 0
    for r in results:
        cur = conn.execute("""
            UPDATE fixtures
            SET actual_home_score = ?,
                actual_away_score = ?,
                is_completed      = 1
            WHERE home_team = ? AND away_team = ? AND date = ?
              AND is_completed = 0
        """, (r["home_score"], r["away_score"],
              r["home_team"], r["away_team"], r["date"]))
        updated += cur.rowcount
    conn.commit()
    return updated


def compute_and_save_accuracy(conn):
    """Calculate prediction accuracy on completed games and save snapshot."""
    rows = conn.execute("""
        SELECT f.actual_home_score, f.actual_away_score,
               p.pred_home_score, p.pred_away_score,
               p.winner, f.home_team, f.away_team
        FROM   fixtures f
        JOIN   predictions p ON p.fixture_id = f.id
        WHERE  f.is_completed = 1
          AND  f.actual_home_score IS NOT NULL
          AND  p.model_version = 'v2'
    """).fetchall()

    if len(rows) < 5:
        print(f"  Only {len(rows)} completed games — skipping accuracy update")
        return

    import numpy as np
    actual_home = [r[0] for r in rows]
    actual_away = [r[1] for r in rows]
    pred_home   = [r[2] for r in rows]
    pred_away   = [r[3] for r in rows]
    pred_winner = [r[4] for r in rows]
    act_winner  = [r[5] if r[0] > r[1] else r[6] for r in rows]

    win_acc   = sum(p == a for p, a in zip(pred_winner, act_winner)) / len(rows)
    home_mae  = float(np.mean(np.abs(np.array(actual_home) - np.array(pred_home))))
    away_mae  = float(np.mean(np.abs(np.array(actual_away) - np.array(pred_away))))
    margin_mae = float(np.mean(np.abs(
        np.array(actual_home) - np.array(actual_away) -
        np.array(pred_home)   + np.array(pred_away))))

    conn.execute("""
        INSERT INTO model_accuracy
            (evaluated_at, model_version, n_games, winner_accuracy,
             home_score_mae, away_score_mae, margin_mae)
        VALUES (?,?,?,?,?,?,?)
    """, (datetime.utcnow().isoformat(), "v2", len(rows),
          win_acc, home_mae, away_mae, margin_mae))
    conn.commit()
    print(f"  Accuracy on {len(rows)} completed games: "
          f"winner={win_acc:.1%}  home MAE={home_mae:.1f}  away MAE={away_mae:.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_update(fetch_upcoming=True, fetch_results=True):
    init_db()

    with get_conn() as conn:
        if fetch_upcoming:
            print("\n[1] Scraping upcoming fixtures + odds...")
            html     = scrape_oddsportal("https://www.oddsportal.com/rugby-league/australia/nrl/")
            fixtures = parse_fixtures_from_html(html)
            print(f"    Found {len(fixtures)} upcoming fixtures")

            print("[2] Saving fixtures to DB...")
            fix_ids = upsert_fixtures(conn, fixtures)
            print(f"    {len(fix_ids)} fixture records saved/updated")

            print("[3] Generating predictions...")
            preds = generate_predictions(fixtures)
            print(f"    Generated {len(preds)} predictions")
            upsert_predictions(conn, preds, fix_ids)

        if fetch_results:
            print("\n[4] Scraping recent results...")
            results_html = scrape_oddsportal(
                "https://www.oddsportal.com/rugby-league/australia/nrl/results/")
            results = parse_results_from_html(results_html)
            print(f"    Found {len(results)} completed game scores")

            if results:
                # Ensure completed fixtures exist in DB
                fix_ids2 = upsert_fixtures(conn, [
                    {"date": r["date"], "home_team": r["home_team"],
                     "away_team": r["away_team"]}
                    for r in results
                ])
                n = update_results(conn, results)
                print(f"    Updated {n} fixtures with actual scores")

        print("\n[5] Computing model accuracy...")
        compute_and_save_accuracy(conn)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true",
                        help="Only fetch completed results")
    parser.add_argument("--predict", action="store_true",
                        help="Only refresh upcoming predictions")
    args = parser.parse_args()

    if args.results:
        run_update(fetch_upcoming=False, fetch_results=True)
    elif args.predict:
        run_update(fetch_upcoming=True, fetch_results=False)
    else:
        run_update(fetch_upcoming=True, fetch_results=True)


if __name__ == "__main__":
    main()
