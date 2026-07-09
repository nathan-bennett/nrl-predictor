"""
Clean raw NRL data, normalise team names, and engineer rolling features.
Output: data/nrl_features.csv

Features added:
  - ELO ratings (standard + margin-adjusted, 538-style)
  - Pythagorean expectation (YTD scored²/(scored²+conceded²))
  - Home/away split form (separate rolling stats for each context)
  - Win/loss streak (current consecutive run)
  - Scoring trend (linear slope of last N scores)
  - Score volatility (std dev of last N scores)
  - Venue-specific win rate
  - Strength of schedule (avg opponent ELO in last 5)
  - Close game record (win% in games decided by ≤6 pts)
  - Odds range across bookmakers (market disagreement)
  - Handicap line movement (open → close)
  - Total score line movement
  - Draw implied probability
  - Market overround (vig)
  - Number of bookmakers surveyed
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TEAM_MAP = {
    "Canterbury Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Cronulla Sharks": "Cronulla-Sutherland Sharks",
    "Cronulla-Sutherland Sharks": "Cronulla-Sutherland Sharks",
    "Manly Sea Eagles": "Manly-Warringah Sea Eagles",
    "North QLD Cowboys": "North Queensland Cowboys",
    "St George Dragons": "St. George Illawarra Dragons",
}

VENUE_COORDS = {
    "Accor Stadium": (-33.8469, 150.9010),
    "Allianz Stadium": (-33.8915, 151.2247),
    "ANZ Stadium": (-33.8469, 150.9010),
    "AAMI Park": (-37.8255, 144.9836),
    "BlueBet Stadium": (-33.7511, 150.6942),
    "Browne Park": (-23.3796, 150.5100),
    "Campbelltown Stadium": (-34.0711, 150.8219),
    "CommBank Stadium": (-33.8144, 150.9942),
    "Cbus Super Stadium": (-28.0167, 153.4000),
    "Docklands Stadium": (-37.8255, 144.9472),
    "Etihad Stadium": (-37.8255, 144.9472),
    "GIO Stadium": (-35.2041, 149.1310),
    "Go Media Stadium": (-36.9078, 174.7759),
    "Hunter Stadium": (-32.9257, 151.7764),
    "Industree Group Stadium": (-33.4389, 151.3444),
    "Kayo Stadium": (-26.6344, 153.1011),
    "Leichhardt Oval": (-33.8837, 151.1506),
    "Mackay Stadium": (-21.1411, 149.1862),
    "McDonald Jones Stadium": (-32.9257, 151.7764),
    "Mud Island": (-27.4698, 153.0251),
    "Newcastle Stadium": (-32.9257, 151.7764),
    "Netstrata Jubilee Stadium": (-33.9697, 151.1003),
    "NIB Stadium": (-31.9505, 115.8605),
    "Ninja Stadium": (-26.6344, 153.1011),
    "NSWRL Stadium": (-33.8144, 150.9942),
    "Optus Stadium": (-31.9505, 115.8605),
    "Pepper Stadium": (-33.8144, 150.9942),
    "Pratten Park": (-33.8837, 151.1506),
    "QCB Stadium": (-27.6139, 152.9716),
    "Qbank Stadium": (-27.6139, 152.9716),
    "QLD Country Bank Stadium": (-19.2590, 146.8169),
    "Redcliffe Stadium": (-27.2264, 153.1000),
    "Scully Park": (-31.0878, 150.9225),
    "Shark Park": (-34.0458, 151.0983),
    "Southern Cross Group Stadium": (-34.0458, 151.0983),
    "Suncorp Stadium": (-27.4647, 153.0094),
    "Sydney Cricket Ground": (-33.8915, 151.2247),
    "TIO Stadium": (-12.4467, 130.8408),
    "Totally Awesome Stadium": (-16.9186, 145.7781),
    "Townsville Stadium": (-19.2590, 146.8169),
    "WIN Stadium": (-34.4237, 150.8931),
    "WIN Network Stadium": (-34.4237, 150.8931),
}

HOME_CITIES = {
    "Brisbane Broncos": (-27.4698, 153.0251),
    "Canterbury-Bankstown Bulldogs": (-33.9100, 151.0341),
    "Canberra Raiders": (-35.2809, 149.1300),
    "Cronulla-Sutherland Sharks": (-34.0458, 151.0983),
    "Dolphins": (-27.2264, 153.1000),
    "Gold Coast Titans": (-28.0167, 153.4000),
    "Manly-Warringah Sea Eagles": (-33.7989, 151.2878),
    "Melbourne Storm": (-37.8255, 144.9836),
    "New Zealand Warriors": (-36.9078, 174.7759),
    "Newcastle Knights": (-32.9257, 151.7764),
    "North Queensland Cowboys": (-19.2590, 146.8169),
    "Parramatta Eels": (-33.8144, 150.9942),
    "Penrith Panthers": (-33.7511, 150.6942),
    "South Sydney Rabbitohs": (-33.8837, 151.2090),
    "St. George Illawarra Dragons": (-34.4237, 150.8931),
    "Sydney Roosters": (-33.8915, 151.2247),
    "Wests Tigers": (-33.8837, 151.1506),
}

ELO_START = 1500
ELO_K = 25
ELO_HOME_ADV = 50  # points added to home team's expected rating


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_raw():
    df = pd.read_excel(DATA_DIR / "nrl_raw.xlsx", sheet_name="Data", header=1)
    df = df.rename(columns={
        "Date": "date",
        "Kick-off (local)": "kickoff",
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Venue": "venue",
        "Home Score": "home_score",
        "Away Score": "away_score",
        "Play Off Game?": "is_playoff",
        "Over Time?": "is_overtime",
        "Home Odds": "home_odds_avg",
        "Draw Odds": "draw_odds_avg",
        "Away Odds": "away_odds_avg",
        "Bookmakers Surveyed": "bookmakers",
        "Home Odds Open": "home_odds_open",
        "Home Odds Close": "home_odds_close",
        "Home Odds Min": "home_odds_min",
        "Home Odds Max": "home_odds_max",
        "Away Odds Open": "away_odds_open",
        "Away Odds Close": "away_odds_close",
        "Away Odds Min": "away_odds_min",
        "Away Odds Max": "away_odds_max",
        "Home Line Open": "home_line_open",
        "Home Line Close": "home_line_close",
        "Away Line Close": "away_line_close",
        "Home Line Odds Close": "home_line_odds_close",
        "Away Line Odds Close": "away_line_odds_close",
        "Total Score Open": "total_line_open",
        "Total Score Close": "total_line_close",
        "Total Score Over Open": "total_over_odds_open",
        "Total Score Over Close": "total_over_odds_close",
        "Total Score Under Open": "total_under_odds_open",
        "Total Score Under Close": "total_under_odds_close",
    })
    return df


def normalise_teams(df):
    df["home_team"] = df["home_team"].map(lambda t: TEAM_MAP.get(t, t))
    df["away_team"] = df["away_team"].map(lambda t: TEAM_MAP.get(t, t))
    return df


def clean(df):
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["is_playoff"] = df["is_playoff"].map(
        lambda x: 1 if str(x).strip().upper() in ("Y", "YES", "1", "TRUE") else 0)
    df["is_overtime"] = df["is_overtime"].map(
        lambda x: 1 if str(x).strip().upper() in ("Y", "YES", "1", "TRUE") else 0)
    df["season"] = df["date"].dt.year
    df["round_in_year"] = df.groupby("season").cumcount() + 1
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["result"] = np.where(df["home_score"] > df["away_score"], 1,
                   np.where(df["home_score"] < df["away_score"], -1, 0))
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_score"] = df["home_score"] + df["away_score"]
    return df


def add_odds_features(df):
    """H2H odds, movement, range, draw prob, vig, line movement, total movement."""
    for col in ["home_odds_close", "away_odds_close", "home_odds_open", "away_odds_open",
                "home_odds_min", "home_odds_max", "away_odds_min", "away_odds_max",
                "home_odds_avg", "away_odds_avg", "draw_odds_avg",
                "home_line_close", "home_line_open", "total_line_close", "total_line_open",
                "total_over_odds_close", "total_over_odds_open",
                "total_under_odds_close", "bookmakers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalised implied probabilities (remove vig)
    df["home_imp_prob"] = 1.0 / df["home_odds_close"]
    df["away_imp_prob"] = 1.0 / df["away_odds_close"]
    vig_h2h = df["home_imp_prob"] + df["away_imp_prob"]
    df["home_imp_prob_norm"] = df["home_imp_prob"] / vig_h2h
    df["away_imp_prob_norm"] = df["away_imp_prob"] / vig_h2h

    # H2H odds movement (close / open — >1 means drifted out)
    df["home_odds_move"] = df["home_odds_close"] / df["home_odds_open"].replace(0, np.nan)
    df["away_odds_move"] = df["away_odds_close"] / df["away_odds_open"].replace(0, np.nan)

    # Odds range across bookmakers — measures market disagreement
    df["home_odds_range"] = df["home_odds_max"] - df["home_odds_min"]
    df["away_odds_range"] = df["away_odds_max"] - df["away_odds_min"]

    # Handicap line movement (positive = line moved toward home team)
    df["line_move"] = df["home_line_close"] - df["home_line_open"]

    # Total score line movement (positive = market expects more scoring)
    df["total_line_move"] = df["total_line_close"] - df["total_line_open"]

    # Over/under odds movement
    df["total_over_move"] = df["total_over_odds_close"] - df["total_over_odds_open"]

    # Draw implied probability and full-market vig
    df["draw_imp_prob"] = 1.0 / df["draw_odds_avg"]
    raw_home = 1.0 / df["home_odds_avg"]
    raw_away = 1.0 / df["away_odds_avg"]
    df["market_vig"] = raw_home + raw_away + df["draw_imp_prob"].fillna(0)

    # Log odds ratio — symmetric transformation of relative strength
    df["log_odds_ratio"] = np.log(df["home_odds_close"] / df["away_odds_close"])

    # Bookmakers count (proxy for market liquidity)
    df["bookmakers"] = df["bookmakers"].fillna(10)

    return df


def _slope(values):
    """Linear regression slope of a sequence. Returns 0 for <3 points."""
    n = len(values)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), np.mean(values)
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (values - ym)).sum() / denom) if denom > 0 else 0.0


def _elo_expected(r_home, r_away, home_adv=ELO_HOME_ADV):
    return 1.0 / (1.0 + 10 ** ((r_away - r_home - home_adv) / 400))


def _elo_margin_k(margin, winner_elo_diff, k=ELO_K):
    """538-style margin-adjusted K factor."""
    mult = np.log(abs(margin) + 1)
    autocorr = 2.2 / (winner_elo_diff * 0.001 + 2.2)
    return k * mult * autocorr


def rolling_team_stats(df, windows=(5, 10)):
    """
    Single-pass computation of all per-team rolling features.
    State updated AFTER recording features — no leakage.
    """
    # Per-team history lists: each entry is a game dict
    all_hist  = {}   # team -> [game, ...]   (all games)
    home_hist = {}   # team -> [game, ...]   (home games only)
    away_hist = {}   # team -> [game, ...]   (away games only)
    season_st = {}   # (team, season) -> {wins, played, scored, conceded}
    venue_st  = {}   # (team, venue) -> {wins, played}
    elo       = {}   # team -> current ELO

    result_rows = []

    for _, row in df.iterrows():
        hteam  = row["home_team"]
        ateam  = row["away_team"]
        venue  = row["venue"]
        season = row["season"]
        margin = int(row["home_score"] - row["away_score"])

        feat = {}

        # ── ELO pre-game ──────────────────────────────────────────────────
        h_elo = elo.get(hteam, ELO_START)
        a_elo = elo.get(ateam, ELO_START)
        feat["home_elo"]  = h_elo
        feat["away_elo"]  = a_elo
        feat["elo_diff"]  = h_elo - a_elo

        # ── Per-team features ─────────────────────────────────────────────
        for prefix, team, is_home in [("home", hteam, True), ("away", ateam, False)]:
            hist   = all_hist.get(team, [])
            s_key  = (team, season)
            ss     = season_st.get(s_key, {"wins": 0, "played": 0, "scored": 0, "conceded": 0})
            split  = home_hist.get(team, []) if is_home else away_hist.get(team, [])

            # YTD
            played = ss["played"]
            feat[f"{prefix}_ytd_wins"]         = ss["wins"]
            feat[f"{prefix}_ytd_played"]        = played
            feat[f"{prefix}_ytd_win_pct"]       = ss["wins"] / played if played else 0.5
            feat[f"{prefix}_ytd_scored_avg"]    = ss["scored"] / played if played else 22.0
            feat[f"{prefix}_ytd_conceded_avg"]  = ss["conceded"] / played if played else 22.0

            # Pythagorean expectation
            if played > 0:
                s = ss["scored"] / played
                c = ss["conceded"] / played
                denom = s ** 2 + c ** 2
                feat[f"{prefix}_pythagorean"] = s ** 2 / denom if denom > 0 else 0.5
            else:
                feat[f"{prefix}_pythagorean"] = 0.5

            # Rolling windows (all games)
            for w in windows:
                recent = hist[-w:] if hist else []
                n = len(recent)
                feat[f"{prefix}_last{w}_win_pct"]        = sum(r["won"] for r in recent) / n if n else 0.5
                feat[f"{prefix}_last{w}_scored_avg"]     = sum(r["scored"] for r in recent) / n if n else 22.0
                feat[f"{prefix}_last{w}_conceded_avg"]   = sum(r["conceded"] for r in recent) / n if n else 22.0
                scores    = np.array([r["scored"]   for r in recent], dtype=float)
                conceded  = np.array([r["conceded"] for r in recent], dtype=float)
                feat[f"{prefix}_last{w}_score_volatility"]    = float(np.std(scores))   if n > 1 else 8.0
                feat[f"{prefix}_last{w}_concede_volatility"]  = float(np.std(conceded)) if n > 1 else 8.0
                feat[f"{prefix}_last{w}_score_trend"]   = _slope(scores)
                feat[f"{prefix}_last{w}_concede_trend"] = _slope(conceded)

            # Win/loss streak (positive = win streak, negative = loss streak)
            streak = 0
            for g in reversed(hist):
                if g["won"] == 1 and streak >= 0:
                    streak += 1
                elif g["won"] == 0 and streak <= 0:
                    streak -= 1
                else:
                    break
            feat[f"{prefix}_streak"] = streak

            # Home/away split form (last 5 in same context as current game)
            recent_split = split[-5:] if split else []
            ns = len(recent_split)
            feat[f"{prefix}_split_last5_win_pct"]      = sum(r["won"] for r in recent_split) / ns if ns else 0.5
            feat[f"{prefix}_split_last5_scored_avg"]   = sum(r["scored"] for r in recent_split) / ns if ns else 22.0
            feat[f"{prefix}_split_last5_conceded_avg"] = sum(r["conceded"] for r in recent_split) / ns if ns else 22.0

            # Close game record (last 20 games, margin ≤ 6 pts)
            close = [r for r in hist[-20:] if r.get("margin", 99) <= 6]
            feat[f"{prefix}_close_game_win_pct"] = sum(r["won"] for r in close) / len(close) if close else 0.5
            feat[f"{prefix}_close_game_count"]   = len(close)

            # Strength of schedule (avg opponent ELO in last 5)
            opp_elos = [elo.get(r["vs"], ELO_START) for r in hist[-5:] if r.get("vs")]
            feat[f"{prefix}_sos_5"] = float(np.mean(opp_elos)) if opp_elos else float(ELO_START)

            # Venue-specific win rate
            vk = (team, venue)
            vs = venue_st.get(vk, {"wins": 0, "played": 0})
            feat[f"{prefix}_venue_win_pct"]  = vs["wins"] / vs["played"] if vs["played"] >= 3 else 0.5
            feat[f"{prefix}_venue_played"]   = vs["played"]

        # ── H2H ──────────────────────────────────────────────────────────
        h2h = [r for r in all_hist.get(hteam, []) if r.get("vs") == ateam]
        feat["h2h_home_win_pct"] = sum(r["won"] for r in h2h) / len(h2h) if h2h else 0.5
        feat["h2h_count"]        = len(h2h)

        result_rows.append(feat)

        # ── Update state (AFTER recording — no leakage) ───────────────────
        h_won = int(row["result"] == 1)
        a_won = int(row["result"] == -1)

        # ELO update
        h_exp = _elo_expected(h_elo, a_elo)
        h_actual = 1.0 if h_won else (0.5 if row["result"] == 0 else 0.0)
        a_actual = 1.0 - h_actual

        elo[hteam] = h_elo + ELO_K * (h_actual - h_exp)
        elo[ateam] = a_elo + ELO_K * (a_actual - (1 - h_exp))

        for team, scored, conceded, won, opponent, is_home_flag in [
            (hteam, int(row["home_score"]), int(row["away_score"]), h_won, ateam, True),
            (ateam, int(row["away_score"]), int(row["home_score"]), a_won, hteam, False),
        ]:
            game_rec = {
                "scored":   scored,
                "conceded": conceded,
                "won":      won,
                "vs":       opponent,
                "margin":   abs(margin),
            }
            if team not in all_hist:
                all_hist[team] = []
            all_hist[team].append(game_rec)

            if is_home_flag:
                if team not in home_hist:
                    home_hist[team] = []
                home_hist[team].append(game_rec)
            else:
                if team not in away_hist:
                    away_hist[team] = []
                away_hist[team].append(game_rec)

            s_key = (team, season)
            if s_key not in season_st:
                season_st[s_key] = {"wins": 0, "played": 0, "scored": 0, "conceded": 0}
            season_st[s_key]["wins"]     += won
            season_st[s_key]["played"]   += 1
            season_st[s_key]["scored"]   += scored
            season_st[s_key]["conceded"] += conceded

            vk = (team, venue)
            if vk not in venue_st:
                venue_st[vk] = {"wins": 0, "played": 0}
            venue_st[vk]["wins"]   += won
            venue_st[vk]["played"] += 1

    feat_df = pd.DataFrame(result_rows)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def add_rest_days(df):
    last_game = {}
    rest_home, rest_away = [], []
    for _, row in df.iterrows():
        for team, col_list in [(row["home_team"], rest_home), (row["away_team"], rest_away)]:
            col_list.append((row["date"] - last_game[team]).days if team in last_game else 7)
        last_game[row["home_team"]] = row["date"]
        last_game[row["away_team"]] = row["date"]
    df["home_rest_days"] = rest_home
    df["away_rest_days"] = rest_away
    return df


def add_travel(df):
    def away_travel(row):
        vcoords = VENUE_COORDS.get(row["venue"])
        tcoords = HOME_CITIES.get(row["away_team"])
        return haversine_km(*tcoords, *vcoords) if vcoords and tcoords else np.nan

    df["away_travel_km"] = df.apply(away_travel, axis=1)
    df["away_travel_km"] = df["away_travel_km"].fillna(df["away_travel_km"].median())
    return df


def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"  {len(df)} rows loaded")

    print("Normalising team names...")
    df = normalise_teams(df)

    print("Cleaning...")
    df = clean(df)
    print(f"  {len(df)} rows after cleaning")

    print("Adding odds features (including new: range, movement, vig, draw prob)...")
    df = add_odds_features(df)

    print("Computing rolling team stats — ELO, Pythagorean, streaks, trends, volatility,")
    print("  venue rates, split form, SoS, close game record (single pass, ~30s)...")
    df = rolling_team_stats(df)

    print("Adding rest days...")
    df = add_rest_days(df)

    print("Adding travel distances...")
    df = add_travel(df)

    out = DATA_DIR / "nrl_features.csv"
    df.to_csv(out, index=False)

    new_cols = [c for c in df.columns if any(k in c for k in
                ["elo", "pyth", "streak", "trend", "volatil", "split",
                 "close_game", "sos_", "venue_win", "odds_range",
                 "line_move", "total_line_move", "draw_imp", "market_vig",
                 "log_odds", "total_over_move"])]
    print(f"\nSaved {len(df)} rows, {len(df.columns)} columns → {out}")
    print(f"New features added ({len(new_cols)}): {sorted(new_cols)}")


if __name__ == "__main__":
    main()
