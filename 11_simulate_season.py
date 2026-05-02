"""
Simulate the 2026 NRL season using the no-odds model.

Steps:
  1. Load full 2026 draw from DB (via 10_fetch_draw_2026.py)
  2. Load actual results for completed 2026 games from DB
  3. Build current team state from historical CSV + completed 2026 games
  4. Predict remaining regular-season games (ELO updates after each prediction)
  5. Calculate projected final ladder
  6. Simulate the NRL McIntyre Final Eight finals bracket
  7. Save projected ladder + bracket to DB

Usage:
  python3 11_simulate_season.py          # full simulation, save to DB
  python3 11_simulate_season.py --no-db  # print results only
"""

import argparse
import pickle
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "data" / "nrl.db"
CSV_PATH  = BASE_DIR / "data" / "nrl_features.csv"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

ELO_K      = 25
HOME_ADV   = 50
SEASON          = 2026
N_MONTE_CARLO   = 100


# ── Team name normalisation (same set as rest of project) ─────────────────────

TEAM_MAP = {
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
    "Canterbury Bulldogs":           "Canterbury-Bankstown Bulldogs",
    "Cronulla Sharks":               "Cronulla-Sutherland Sharks",
    "Manly Sea Eagles":              "Manly-Warringah Sea Eagles",
    "North QLD Cowboys":             "North Queensland Cowboys",
    "NQ Cowboys":                    "North Queensland Cowboys",
    "St George Dragons":             "St. George Illawarra Dragons",
    "St George Illawarra Dragons":   "St. George Illawarra Dragons",
}


def norm(name: str) -> str:
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_no_odds_models():
    pkgs = {}
    for key, fname in [
        ("winner",     "model_winner_no_odds.pkl"),
        ("home_score", "model_home_score_no_odds.pkl"),
        ("away_score", "model_away_score_no_odds.pkl"),
    ]:
        path = MODEL_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{fname} not found. Run python3 09_train_no_odds.py first."
            )
        with open(path, "rb") as f:
            pkgs[key] = pickle.load(f)
    return pkgs


def predict_with_pkg(pkg, X):
    if pkg.get("type") == "stack" or pkg.get("stack_base"):
        base = pkg["stack_base"]
        p1 = base["m1"].predict(X).reshape(-1, 1)
        p2 = base["m2"].predict(X).reshape(-1, 1)
        return pkg["model"].predict(np.hstack([p1, p2]))
    return pkg["model"].predict(X)


def predict_proba_winner(pkg, X):
    if pkg.get("type") == "stack" or pkg.get("stack_base"):
        base = pkg["stack_base"]
        p1 = base["m1"].predict_proba(X)[:, 1].reshape(-1, 1)
        p2 = base["m2"].predict_proba(X)[:, 1].reshape(-1, 1)
        return pkg["model"].predict_proba(np.hstack([p1, p2]))[:, 1]
    return pkg["model"].predict_proba(X)[:, 1]


def round_score(x: float) -> int:
    return int(round(x / 2)) * 2


# ── Haversine for travel distance ─────────────────────────────────────────────

VENUE_COORDS = {
    "Accor Stadium":          (-33.8469, 150.9010),
    "Allianz Stadium":        (-33.8915, 151.2247),
    "AAMI Park":              (-37.8255, 144.9836),
    "CommBank Stadium":       (-33.8144, 150.9942),
    "Suncorp Stadium":        (-27.4647, 153.0094),
    "McDonald Jones Stadium": (-32.9257, 151.7764),
    "Cbus Super Stadium":     (-28.0167, 153.4000),
    "GIO Stadium":            (-35.2041, 149.1310),
    "BlueBet Stadium":        (-33.7511, 150.6942),
    "Kayo Stadium":           (-26.6344, 153.1011),
    "4 Pines Park":           (-33.7989, 151.2878),
    "Sky Stadium":            (-41.3272, 174.8052),
    "Sydney Football Stadium":(-33.8915, 151.2247),
}
HOME_CITIES = {
    "Brisbane Broncos":              (-27.4698, 153.0251),
    "Canterbury-Bankstown Bulldogs": (-33.9100, 151.0341),
    "Canberra Raiders":              (-35.2809, 149.1300),
    "Cronulla-Sutherland Sharks":    (-34.0458, 151.0983),
    "Dolphins":                      (-27.2264, 153.1000),
    "Gold Coast Titans":             (-28.0167, 153.4000),
    "Manly-Warringah Sea Eagles":    (-33.7989, 151.2878),
    "Melbourne Storm":               (-37.8255, 144.9836),
    "New Zealand Warriors":          (-36.9078, 174.7759),
    "Newcastle Knights":             (-32.9257, 151.7764),
    "North Queensland Cowboys":      (-19.2590, 146.8169),
    "Parramatta Eels":               (-33.8144, 150.9942),
    "Penrith Panthers":              (-33.7511, 150.6942),
    "South Sydney Rabbitohs":        (-33.8837, 151.2090),
    "St. George Illawarra Dragons":  (-34.4237, 150.8931),
    "Sydney Roosters":               (-33.8915, 151.2247),
    "Wests Tigers":                  (-33.8837, 151.1506),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dl   = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Team simulation state ─────────────────────────────────────────────────────

class TeamState:
    """Tracks per-team ELO, season record, and rolling form for simulation."""

    def __init__(self, hist_df):
        self.elo       = {}
        self.ytd       = {}   # {team: {wins, losses, for, against, played}}
        self.last5     = {}   # {team: deque of last 5 results {won, scored, conceded}}
        self.streak    = {}   # {team: int (+ve = wins, -ve = losses)}
        self.hist_stats = {}  # static: last-known "as_home"/"as_away" feature dicts

        self._init_from_hist(hist_df)

    def _init_from_hist(self, hist_df):
        home_cols = [c for c in hist_df.columns
                     if c.startswith("home_") and c not in ("home_team", "home_score")]
        away_cols = [c for c in hist_df.columns
                     if c.startswith("away_") and c not in ("away_team", "away_score")]

        for _, row in hist_df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            if ht not in self.hist_stats:
                self.hist_stats[ht] = {"as_home": {}, "as_away": {}}
            if at not in self.hist_stats:
                self.hist_stats[at] = {"as_home": {}, "as_away": {}}

            self.hist_stats[ht]["as_home"] = {c: row[c] for c in home_cols if c in row.index}
            self.hist_stats[at]["as_away"] = {c: row[c] for c in away_cols if c in row.index}
            self.elo[ht] = float(row.get("home_elo", 1500))
            self.elo[at] = float(row.get("away_elo", 1500))

        # Zero out 2026 season YTD — will accumulate from completed games below
        for team in self.hist_stats:
            self.ytd[team]  = {"wins": 0, "losses": 0, "for": 0, "against": 0, "played": 0}
            self.last5[team] = []
            self.streak[team] = 0

    def copy(self):
        """Fast copy for Monte Carlo — hist_stats is shared (read-only)."""
        new = object.__new__(TeamState)
        new.elo        = dict(self.elo)
        new.ytd        = {k: dict(v) for k, v in self.ytd.items()}
        new.last5      = {k: list(v) for k, v in self.last5.items()}
        new.streak     = dict(self.streak)
        new.hist_stats = self.hist_stats
        return new

    def apply_game(self, home_team, away_team, home_score, away_score):
        """Update state with a known (actual or predicted) result."""
        h_won = home_score > away_score

        for team, scored, conceded, won in [
            (home_team, home_score, away_score, h_won),
            (away_team, away_score, home_score, not h_won),
        ]:
            if team not in self.ytd:
                self.ytd[team]   = {"wins": 0, "losses": 0, "for": 0, "against": 0, "played": 0}
                self.last5[team] = []
                self.streak[team] = 0

            self.ytd[team]["played"] += 1
            self.ytd[team]["for"]     += scored
            self.ytd[team]["against"] += conceded
            if won:
                self.ytd[team]["wins"]   += 1
            else:
                self.ytd[team]["losses"] += 1

            self.last5[team].append({"won": won, "scored": scored, "conceded": conceded})
            if len(self.last5[team]) > 5:
                self.last5[team].pop(0)

            # Streak
            if won:
                self.streak[team] = self.streak.get(team, 0)
                self.streak[team] = self.streak[team] + 1 if self.streak[team] >= 0 else 1
            else:
                self.streak[team] = self.streak.get(team, 0)
                self.streak[team] = self.streak[team] - 1 if self.streak[team] <= 0 else -1

        # ELO update
        h_elo = self.elo.get(home_team, 1500)
        a_elo = self.elo.get(away_team, 1500)
        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / 400))
        h_act = 1.0 if h_won else (0.5 if home_score == away_score else 0.0)
        self.elo[home_team] = h_elo + ELO_K * (h_act - h_exp)
        self.elo[away_team] = a_elo + ELO_K * ((1 - h_act) - (1 - h_exp))

    def feature_row(self, home_team, away_team, venue, round_num,
                    h2h_stats, is_playoff=0):
        """Build feature dict for a prediction."""
        hs  = self.hist_stats.get(home_team, {}).get("as_home", {})
        as_ = self.hist_stats.get(away_team, {}).get("as_away", {})
        feat = {}
        feat.update(hs)
        feat.update(as_)

        # Override with live ELO
        h_elo = self.elo.get(home_team, 1500)
        a_elo = self.elo.get(away_team, 1500)
        feat["home_elo"]  = h_elo
        feat["away_elo"]  = a_elo
        feat["elo_diff"]  = h_elo - a_elo

        # Override with YTD
        ytd_h = self.ytd.get(home_team, {})
        ytd_a = self.ytd.get(away_team, {})
        h_played = ytd_h.get("played", 0)
        a_played = ytd_a.get("played", 0)
        feat["home_ytd_played"]       = h_played
        feat["home_ytd_wins"]         = ytd_h.get("wins", 0)
        feat["home_ytd_win_pct"]      = ytd_h.get("wins", 0) / h_played if h_played else 0.5
        feat["home_ytd_scored_avg"]   = ytd_h.get("for", 0) / h_played if h_played else feat.get("home_ytd_scored_avg", 20)
        feat["home_ytd_conceded_avg"] = ytd_h.get("against", 0) / h_played if h_played else feat.get("home_ytd_conceded_avg", 20)
        feat["away_ytd_played"]       = a_played
        feat["away_ytd_wins"]         = ytd_a.get("wins", 0)
        feat["away_ytd_win_pct"]      = ytd_a.get("wins", 0) / a_played if a_played else 0.5
        feat["away_ytd_scored_avg"]   = ytd_a.get("for", 0) / a_played if a_played else feat.get("away_ytd_scored_avg", 20)
        feat["away_ytd_conceded_avg"] = ytd_a.get("against", 0) / a_played if a_played else feat.get("away_ytd_conceded_avg", 20)

        # Pythagorean from YTD
        hf, hc = feat["home_ytd_scored_avg"], feat["home_ytd_conceded_avg"]
        af, ac = feat["away_ytd_scored_avg"], feat["away_ytd_conceded_avg"]
        feat["home_pythagorean"] = hf**2 / (hf**2 + hc**2) if (hf**2 + hc**2) > 0 else 0.5
        feat["away_pythagorean"] = af**2 / (af**2 + ac**2) if (af**2 + ac**2) > 0 else 0.5

        # Override last5 from live state
        h5 = self.last5.get(home_team, [])
        a5 = self.last5.get(away_team, [])
        if h5:
            feat["home_last5_win_pct"]       = sum(g["won"] for g in h5) / len(h5)
            feat["home_last5_scored_avg"]    = np.mean([g["scored"] for g in h5])
            feat["home_last5_conceded_avg"]  = np.mean([g["conceded"] for g in h5])
        if a5:
            feat["away_last5_win_pct"]       = sum(g["won"] for g in a5) / len(a5)
            feat["away_last5_scored_avg"]    = np.mean([g["scored"] for g in a5])
            feat["away_last5_conceded_avg"]  = np.mean([g["conceded"] for g in a5])

        feat["home_streak"] = self.streak.get(home_team, 0)
        feat["away_streak"] = self.streak.get(away_team, 0)

        # H2H
        hh = h2h_stats.get((home_team, away_team), {"h2h_home_win_pct": 0.5, "h2h_count": 0})
        feat["h2h_home_win_pct"] = hh["h2h_home_win_pct"]
        feat["h2h_count"]        = hh["h2h_count"]

        # Context
        feat["is_playoff"]     = is_playoff
        feat["round_in_year"]  = round_num
        feat["dayofweek"]      = 5  # Saturday
        feat["month"]          = 5  # approximate
        feat["home_rest_days"] = feat.get("home_rest_days", 7)
        feat["away_rest_days"] = feat.get("away_rest_days", 7)

        # Travel
        travel_km = 500.0
        if venue and away_team in HOME_CITIES:
            vcoords = VENUE_COORDS.get(venue)
            tcoords = HOME_CITIES.get(away_team)
            if vcoords and tcoords:
                travel_km = haversine_km(*tcoords, *vcoords)
        feat["away_travel_km"] = travel_km

        return feat


# ── H2H stats ─────────────────────────────────────────────────────────────────

def build_h2h(hist_df):
    h2h = {}
    for (home, away), grp in hist_df.groupby(["home_team", "away_team"]):
        wins = (grp["result"] == 1).sum()
        h2h[(home, away)] = {"h2h_home_win_pct": wins / len(grp), "h2h_count": len(grp)}
    return h2h


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_game(home_team, away_team, venue, round_num, state, h2h_stats, pkgs,
                 is_playoff=0):
    """Predict a single game using the no-odds model and current team state."""
    feat = state.feature_row(home_team, away_team, venue, round_num, h2h_stats, is_playoff)
    features = pkgs["home_score"]["features"]

    row = pd.DataFrame([feat]).reindex(columns=features, fill_value=np.nan).fillna(0)

    pred_home_raw = max(0.0, float(predict_with_pkg(pkgs["home_score"], row)[0]))
    pred_away_raw = max(0.0, float(predict_with_pkg(pkgs["away_score"], row)[0]))
    pred_home = round_score(pred_home_raw)
    pred_away = round_score(pred_away_raw)

    home_win_prob = float(predict_proba_winner(pkgs["winner"], row)[0])
    winner        = home_team if home_win_prob >= 0.5 else away_team
    confidence    = home_win_prob if home_win_prob >= 0.5 else (1 - home_win_prob)

    return {
        "home_team":     home_team,
        "away_team":     away_team,
        "venue":         venue,
        "pred_home":     pred_home,
        "pred_away":     pred_away,
        "pred_margin":   pred_home - pred_away,
        "winner":        winner,
        "home_win_prob": round(home_win_prob * 100, 1),
        "away_win_prob": round((1 - home_win_prob) * 100, 1),
        "confidence":    round(confidence * 100, 1),
    }


# ── Ladder calculation ────────────────────────────────────────────────────────

def calculate_ladder(records: dict) -> pd.DataFrame:
    rows = []
    for team, r in records.items():
        played = r["wins"] + r["losses"]
        pts    = r["wins"] * 2
        diff   = r["for"] - r["against"]
        rows.append({
            "Team":    team,
            "W":       r["wins"],
            "L":       r["losses"],
            "Played":  played,
            "For":     r["for"],
            "Against": r["against"],
            "Diff":    diff,
            "Points":  pts,
        })
    df = pd.DataFrame(rows).sort_values(
        ["Points", "Diff"], ascending=[False, False]
    ).reset_index(drop=True)
    df.index = df.index + 1  # 1-based position
    return df


# ── NRL McIntyre Final Eight ──────────────────────────────────────────────────

def simulate_finals(ladder, state, h2h_stats, pkgs):
    """
    McIntyre Final Eight (correct NRL format):
      W1: QF1=1v4, QF2=2v3 (double chance); EF1=5v8, EF2=6v7 (sudden death)
      W2: SF1=loser(QF1) v winner(EF2); SF2=loser(QF2) v winner(EF1)
      W3: PF1=winner(QF1) v winner(SF1); PF2=winner(QF2) v winner(SF2)
      W4: GF=winner(PF1) v winner(PF2)
    """
    top8 = ladder.head(8)["Team"].tolist()
    if len(top8) < 8:
        return []

    t1, t2, t3, t4, t5, t6, t7, t8 = top8
    results = []

    def play(label, home, away, round_num, order):
        r = predict_game(home, away, None, round_num, state, h2h_stats, pkgs, is_playoff=1)
        r["round_name"]  = label
        r["match_order"] = order
        results.append(r)
        state.apply_game(home, away, r["pred_home"], r["pred_away"])
        return r["winner"], r["home_team"] if r["winner"] != r["home_team"] else r["away_team"]

    # Week 1
    print("\n  Finals Week 1 (Qualifying + Elimination Finals)")
    qf1_w, qf1_l = play("Qualifying Final 1 (1v4)", t1, t4, 28, 1)
    qf2_w, qf2_l = play("Qualifying Final 2 (2v3)", t2, t3, 28, 2)
    ef1_w, _     = play("Elimination Final 1 (5v8)", t5, t8, 28, 3)
    ef2_w, _     = play("Elimination Final 2 (6v7)", t6, t7, 28, 4)

    # Week 2
    print("  Finals Week 2 (Semi Finals)")
    sf1_w, _ = play("Semi Final 1", qf1_l, ef2_w, 29, 1)  # loser QF1 vs winner EF2
    sf2_w, _ = play("Semi Final 2", qf2_l, ef1_w, 29, 2)  # loser QF2 vs winner EF1

    # Week 3
    print("  Finals Week 3 (Preliminary Finals)")
    pf1_w, _ = play("Preliminary Final 1", qf1_w, sf1_w, 30, 1)
    pf2_w, _ = play("Preliminary Final 2", qf2_w, sf2_w, 30, 2)

    # Week 4
    print("  Finals Week 4 (Grand Final)")
    gf_w, _ = play("Grand Final", pf1_w, pf2_w, 31, 1)
    print(f"\n  🏆 Predicted Premiers: {gf_w}")

    return results


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(upcoming, state_snapshot, h2h_stats, pkgs, all_teams, n_sims=N_MONTE_CARLO):
    """
    Stochastically simulate the season N times.
    Each run samples game outcomes from model win-probabilities,
    returning a dict of {team: {position: count}}.
    """
    from collections import defaultdict

    rng = np.random.default_rng()
    position_counts = defaultdict(lambda: defaultdict(int))
    features = pkgs["home_score"]["features"]

    for sim_i in range(n_sims):
        if (sim_i + 1) % 25 == 0:
            print(f"  MC run {sim_i + 1}/{n_sims}...")

        state = state_snapshot.copy()
        sim_records = {team: {
            "wins":    state.ytd.get(team, {}).get("wins",    0),
            "losses":  state.ytd.get(team, {}).get("losses",  0),
            "for":     state.ytd.get(team, {}).get("for",     0),
            "against": state.ytd.get(team, {}).get("against", 0),
        } for team in all_teams}

        for r in upcoming:
            ht  = norm(r["home_team"])
            at  = norm(r["away_team"])
            ven = r.get("venue") or ""
            rnd = r.get("round") or 15

            # Only use winner model for speed — scores use fixed margin
            feat = state.feature_row(ht, at, ven, rnd, h2h_stats)
            row  = pd.DataFrame([feat]).reindex(columns=features, fill_value=np.nan).fillna(0)
            hwp  = float(predict_proba_winner(pkgs["winner"], row)[0])

            home_wins = rng.random() < hwp
            # Fixed scores: winner gets 20, loser gets 12 (typical NRL margin)
            ph, pa = (20, 12) if home_wins else (12, 20)

            for team, scored, conceded, won in [
                (ht, ph, pa, home_wins),
                (at, pa, ph, not home_wins),
            ]:
                if team not in sim_records:
                    sim_records[team] = {"wins": 0, "losses": 0, "for": 0, "against": 0}
                sim_records[team]["for"]     += scored
                sim_records[team]["against"] += conceded
                if won:
                    sim_records[team]["wins"]   += 1
                else:
                    sim_records[team]["losses"] += 1

            state.apply_game(ht, at, ph, pa)

        ladder = calculate_ladder(sim_records)
        for pos, row in ladder.iterrows():
            position_counts[row["Team"]][pos] += 1

    return position_counts


def save_monte_carlo_to_db(conn, position_counts, n_sims, simulated_at):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monte_carlo_positions (
            simulated_at TEXT,
            team         TEXT,
            position     INTEGER,
            count        INTEGER DEFAULT 0,
            n_sims       INTEGER DEFAULT 100,
            PRIMARY KEY (simulated_at, team, position)
        )
    """)
    conn.execute("DELETE FROM monte_carlo_positions WHERE simulated_at = ?", (simulated_at,))
    for team, pos_counts in position_counts.items():
        for pos, count in pos_counts.items():
            conn.execute("""
                INSERT INTO monte_carlo_positions (simulated_at, team, position, count, n_sims)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(simulated_at, team, position) DO UPDATE SET count = excluded.count
            """, (simulated_at, team, int(pos), int(count), n_sims))
    conn.commit()


# ── DB persistence ────────────────────────────────────────────────────────────

def save_to_db(conn, ladder_df, actual_ladder, finals_results, simulated_at):
    conn.execute("DELETE FROM season_simulation WHERE simulated_at = ? AND model_version = 'no_odds'",
                 (simulated_at,))
    for pos, row in ladder_df.iterrows():
        team = row["Team"]
        act  = actual_ladder.get(team, {"wins": 0, "losses": 0})
        conn.execute("""
            INSERT INTO season_simulation
                (simulated_at, model_version, team,
                 actual_wins, actual_losses,
                 proj_wins, proj_losses, proj_points,
                 proj_for, proj_against, proj_diff, proj_position)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(simulated_at, model_version, team) DO UPDATE SET
                proj_wins     = excluded.proj_wins,
                proj_losses   = excluded.proj_losses,
                proj_points   = excluded.proj_points,
                proj_for      = excluded.proj_for,
                proj_against  = excluded.proj_against,
                proj_diff     = excluded.proj_diff,
                proj_position = excluded.proj_position
        """, (
            simulated_at, "no_odds", team,
            act["wins"], act["losses"],
            int(row["W"]), int(row["L"]), int(row["Points"]),
            int(row["For"]), int(row["Against"]), int(row["Diff"]),
            int(pos),
        ))

    conn.execute("DELETE FROM finals_simulation WHERE simulated_at = ? AND model_version = 'no_odds'",
                 (simulated_at,))
    for r in finals_results:
        conn.execute("""
            INSERT INTO finals_simulation
                (simulated_at, model_version, round_name, match_order,
                 home_team, away_team, pred_home_score, pred_away_score,
                 pred_winner, home_win_prob, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            simulated_at, "no_odds",
            r["round_name"], r["match_order"],
            r["home_team"], r["away_team"],
            r["pred_home"], r["pred_away"],
            r["winner"], r["home_win_prob"], r["confidence"],
        ))
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true",
                        help="Print results only, don't write to DB")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo simulation")
    args = parser.parse_args()

    print("Loading no-odds models...")
    pkgs = load_no_odds_models()

    print("Loading historical features...")
    hist = pd.read_csv(CSV_PATH, parse_dates=["date"])
    h2h_stats = build_h2h(hist)

    print("Loading 2026 draw and results from DB...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    draw_rows = conn.execute("""
        SELECT date, home_team, away_team, venue, round, is_finals,
               actual_home_score, actual_away_score, is_completed
        FROM   fixtures
        WHERE  season = ?
        ORDER  BY round ASC, date ASC
    """, (SEASON,)).fetchall()

    if not draw_rows:
        # Fall back to CSV
        csv_path = DATA_DIR / "draw_2026.csv"
        if csv_path.exists():
            draw_df = pd.read_csv(csv_path)
            draw_rows = draw_df.to_dict("records")
            print(f"  Loaded {len(draw_rows)} fixtures from draw_2026.csv")
        else:
            print("No 2026 draw found. Run 10_fetch_draw_2026.py first.")
            conn.close()
            return
    else:
        draw_rows = [dict(r) for r in draw_rows]
        print(f"  Loaded {len(draw_rows)} fixtures from DB")

    conn.close()

    # Separate completed vs upcoming
    completed = [r for r in draw_rows if r.get("is_completed")]
    upcoming  = [r for r in draw_rows if not r.get("is_completed")
                 and not r.get("is_finals")]
    print(f"  Completed: {len(completed)}  Remaining regular season: {len(upcoming)}")

    # Build team state
    print("\nBuilding team state from historical data...")
    state = TeamState(hist)

    # Apply completed 2026 games
    actual_records = {}
    for r in completed:
        ht, at = norm(r["home_team"]), norm(r["away_team"])
        hs, as_ = int(r["actual_home_score"] or 0), int(r["actual_away_score"] or 0)
        state.apply_game(ht, at, hs, as_)
        for team, scored, conceded, won in [(ht, hs, as_, hs > as_), (at, as_, hs, as_ > hs)]:
            if team not in actual_records:
                actual_records[team] = {"wins": 0, "losses": 0}
            if won:
                actual_records[team]["wins"]   += 1
            else:
                actual_records[team]["losses"] += 1

    # Snapshot state after completed games — used for Monte Carlo later
    state_snapshot = state.copy()

    # Simulate remaining regular season
    print(f"\nSimulating {len(upcoming)} remaining regular-season games...")
    sim_records = {team: {
        "wins":    state.ytd.get(team, {}).get("wins",    0),
        "losses":  state.ytd.get(team, {}).get("losses",  0),
        "for":     state.ytd.get(team, {}).get("for",     0),
        "against": state.ytd.get(team, {}).get("against", 0),
    } for team in set(
        [norm(r["home_team"]) for r in draw_rows] +
        [norm(r["away_team"]) for r in draw_rows]
    )}

    for r in upcoming:
        ht  = norm(r["home_team"])
        at  = norm(r["away_team"])
        ven = r.get("venue") or ""
        rnd = r.get("round") or 15
        is_f = bool(r.get("is_finals"))
        res = predict_game(ht, at, ven, rnd, state, h2h_stats, pkgs, is_playoff=int(is_f))
        ph, pa = res["pred_home"], res["pred_away"]

        # Update simulation records
        for team, scored, conceded, won in [
            (ht, ph, pa, ph > pa),
            (at, pa, ph, pa > ph),
        ]:
            if team not in sim_records:
                sim_records[team] = {"wins": 0, "losses": 0, "for": 0, "against": 0}
            sim_records[team]["for"]     += scored
            sim_records[team]["against"] += conceded
            if won:
                sim_records[team]["wins"]   += 1
            else:
                sim_records[team]["losses"] += 1

        state.apply_game(ht, at, ph, pa)

    # Build ladder
    ladder = calculate_ladder(sim_records)

    print("\n" + "=" * 60)
    print(f"  PROJECTED {SEASON} NRL LADDER (after simulation)")
    print("=" * 60)
    print(f"  {'Pos':<5}{'Team':<38}{'W':>4}{'L':>4}{'Pts':>5}{'Diff':>6}")
    print("  " + "-" * 58)
    for pos, row in ladder.iterrows():
        marker = "←" if pos <= 4 else ("  " if pos <= 8 else "")
        print(f"  {pos:<5}{row['Team']:<38}{row['W']:>4}{row['L']:>4}"
              f"{row['Points']:>5}{row['Diff']:>6}  {marker}")
    print("\n  ← Top 4 = double chance  |  Top 8 = finals")

    # Simulate finals
    print("\n" + "=" * 60)
    print(f"  PREDICTED FINALS BRACKET")
    print("=" * 60)
    finals_results = simulate_finals(ladder, state, h2h_stats, pkgs)

    for r in finals_results:
        margin = r["pred_home"] - r["pred_away"]
        margin_str = f"+{margin}" if margin >= 0 else str(margin)
        print(f"  {r['round_name']}")
        print(f"    {r['home_team']} {r['pred_home']} – {r['pred_away']} "
              f"{r['away_team']}  →  Winner: {r['winner']} "
              f"({r['confidence']:.0f}%)")

    # Save deterministic simulation to DB
    if not args.no_db:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        simulated_at = datetime.utcnow().isoformat()
        save_to_db(conn, ladder, actual_records, finals_results, simulated_at)
        conn.close()
        print(f"\nResults saved to DB (simulated_at={simulated_at})")

    # Monte Carlo
    if not args.no_mc and not args.no_db:
        all_teams = list(set(
            [norm(r["home_team"]) for r in upcoming + completed] +
            [norm(r["away_team"]) for r in upcoming + completed]
        ))
        print(f"\nRunning Monte Carlo ({N_MONTE_CARLO} simulations)...")
        mc_counts = run_monte_carlo(upcoming, state_snapshot, h2h_stats, pkgs, all_teams, N_MONTE_CARLO)
        conn = sqlite3.connect(DB_PATH)
        save_monte_carlo_to_db(conn, mc_counts, N_MONTE_CARLO, simulated_at)
        conn.close()
        print(f"Monte Carlo saved to DB.")

    print("\nDone.")


if __name__ == "__main__":
    main()
