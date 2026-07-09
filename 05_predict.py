"""
Generate predictions for upcoming NRL matches using the best v2 models.

Models used (best on 2025+ test set):
  Winner : Stack v2 — XGBoost + RF → Logistic Regression  (55.8% test acc)
  Home score: Stack v2 — XGBoost + RF → Ridge             (MAE 9.59)
  Away score: XGBoost v2 tuned                            (MAE 8.62)

Usage:
  python3 05_predict.py
  python3 05_predict.py --manual "Penrith Panthers" "Brisbane Broncos"
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

VENUE_COORDS = {
    "Accor Stadium": (-33.8469, 150.9010),
    "Allianz Stadium": (-33.8915, 151.2247),
    "AAMI Park": (-37.8255, 144.9836),
    "CommBank Stadium": (-33.8144, 150.9942),
    "Suncorp Stadium": (-27.4647, 153.0094),
    "McDonald Jones Stadium": (-32.9257, 151.7764),
    "Cbus Super Stadium": (-28.0167, 153.4000),
    "GIO Stadium": (-35.2041, 149.1310),
    "BlueBet Stadium": (-33.7511, 150.6942),
    "Kayo Stadium": (-26.6344, 153.1011),
    "Shark Park": (-34.0458, 151.0983),
    "4 Pines Park": (-33.7989, 151.2878),
    "Sky Stadium": (-41.3272, 174.8052),
    "Sydney Football Stadium": (-33.8915, 151.2247),
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


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    pkgs = {}
    for key, fname in [
        ("winner",     "model_winner_v2.pkl"),
        ("home_score", "model_home_score_v2.pkl"),
        ("away_score", "model_away_score_v2.pkl"),
    ]:
        with open(MODEL_DIR / fname, "rb") as f:
            pkgs[key] = pickle.load(f)
    return pkgs


def predict_with_pkg(pkg, X):
    """Run prediction through a plain or stacked model package."""
    model_type = pkg.get("type")

    if model_type == "stack" or pkg.get("stack_base") is not None:
        base = pkg["stack_base"]
        xgb_pred = base["xgb"].predict(X).reshape(-1, 1)
        rf_pred  = base["rf"].predict(X).reshape(-1, 1)
        meta_X   = np.hstack([xgb_pred, rf_pred])
        return pkg["model"].predict(meta_X)
    else:
        return pkg["model"].predict(X)


def predict_proba_winner(pkg, X):
    """Return home-win probability from the stacked winner model."""
    base   = pkg["stack_base"]
    xgb_p  = base["xgb"].predict_proba(X)[:, 1].reshape(-1, 1)
    rf_p   = base["rf"].predict_proba(X)[:, 1].reshape(-1, 1)
    meta_X = np.hstack([xgb_p, rf_p])
    return pkg["model"].predict_proba(meta_X)[:, 1]


# ── Historical state extraction ───────────────────────────────────────────────

def build_team_current_stats(hist: pd.DataFrame) -> dict:
    """
    For each team, track the most recent feature values when playing as home
    and as away. Returns:
      {team: {"as_home": {col: val, ...}, "as_away": {col: val, ...}}}

    Using the last home game row gives context-correct split form:
      home_split_last5_win_pct = that team's last-5 HOME games win %
      away_split_last5_win_pct = that team's last-5 AWAY games win %
    """
    home_cols = [c for c in hist.columns if c.startswith("home_")
                 and c not in ("home_team", "home_score")]
    away_cols = [c for c in hist.columns if c.startswith("away_")
                 and c not in ("away_team", "away_score")]

    team_stats: dict = {}

    for _, row in hist.iterrows():
        ht, at = row["home_team"], row["away_team"]

        if ht not in team_stats:
            team_stats[ht] = {"as_home": {}, "as_away": {}}
        if at not in team_stats:
            team_stats[at] = {"as_home": {}, "as_away": {}}

        team_stats[ht]["as_home"] = {c: row[c] for c in home_cols}
        team_stats[at]["as_away"] = {c: row[c] for c in away_cols}

    return team_stats


def build_h2h_stats(hist: pd.DataFrame) -> dict:
    h2h = {}
    for (home, away), grp in hist.groupby(["home_team", "away_team"]):
        wins = (grp["result"] == 1).sum()
        h2h[(home, away)] = {"h2h_home_win_pct": wins / len(grp), "h2h_count": len(grp)}
    return h2h


# ── Feature row construction ──────────────────────────────────────────────────

def build_feature_row(
    home_team, away_team,
    team_stats, h2h_stats,
    home_odds=None, away_odds=None,
    home_line=None, total_line=None,
    venue=None, is_playoff=0,
) -> dict:
    """
    Assemble an 81-feature row for one match.
    Uses home team's "as_home" stats and away team's "as_away" stats so that
    home/away split form features are context-correct.
    """
    hs  = team_stats.get(home_team, {}).get("as_home", {})
    as_ = team_stats.get(away_team, {}).get("as_away", {})
    hh  = h2h_stats.get((home_team, away_team),
                        {"h2h_home_win_pct": 0.5, "h2h_count": 0})

    feat: dict = {}

    # ── Team rolling stats (from historical rows) ─────────────────────────
    feat.update(hs)   # home_elo, home_last5_win_pct, home_streak, etc.
    feat.update(as_)  # away_elo, away_last5_win_pct, away_streak, etc.

    # Cross-team ELO diff (recalculate from current values)
    feat["elo_diff"] = feat.get("home_elo", 1500) - feat.get("away_elo", 1500)

    # H2H
    feat["h2h_home_win_pct"] = hh["h2h_home_win_pct"]
    feat["h2h_count"]        = hh["h2h_count"]

    # ── Odds features ─────────────────────────────────────────────────────
    h_odds = home_odds or 1.8
    a_odds = away_odds or 2.2
    h_imp  = 1.0 / h_odds
    a_imp  = 1.0 / a_odds
    vig    = h_imp + a_imp
    feat["home_imp_prob_norm"] = h_imp / vig
    feat["away_imp_prob_norm"] = a_imp / vig
    feat["log_odds_ratio"]     = np.log(h_odds / a_odds)
    feat["home_line_close"]    = home_line if home_line is not None else 0.0
    feat["total_line_close"]   = total_line if total_line is not None else 42.0
    feat["home_odds_move"]     = 1.0   # not available for upcoming
    feat["away_odds_move"]     = 1.0
    feat["line_move"]          = 0.0
    feat["total_line_move"]    = 0.0
    feat["total_over_move"]    = 0.0
    feat["home_odds_range"]    = feat.get("home_odds_range", 0.3)
    feat["away_odds_range"]    = feat.get("away_odds_range", 0.3)
    feat["draw_imp_prob"]      = 1.0 / 25.0   # typical NRL draw odds
    feat["market_vig"]         = vig + feat["draw_imp_prob"]
    feat["bookmakers"]         = 11.0

    # ── Context ───────────────────────────────────────────────────────────
    feat["is_playoff"]    = is_playoff
    feat["round_in_year"] = 11        # approximate
    feat["dayofweek"]     = 5         # Saturday
    feat["month"]         = pd.Timestamp.now().month
    feat["home_rest_days"] = feat.get("home_rest_days", 7)
    feat["away_rest_days"] = feat.get("away_rest_days", 7)

    # Travel
    travel_km = np.nan
    if venue and away_team in HOME_CITIES:
        vcoords = VENUE_COORDS.get(venue)
        tcoords = HOME_CITIES.get(away_team)
        if vcoords and tcoords:
            travel_km = haversine_km(*tcoords, *vcoords)
    feat["away_travel_km"] = travel_km if not np.isnan(travel_km) else 500.0

    # Venue win rates: use values from last known home/away game (already in hs/as_)
    # They're already set via the hs/as_ .update() above; no override needed.

    return feat


def prep_feature_row(feat: dict, features: list) -> pd.DataFrame:
    """Align a feature dict to the model's expected feature list."""
    row = pd.DataFrame([feat])
    row = row.reindex(columns=features, fill_value=np.nan)
    # Fill any remaining NaNs with column medians from the row (fallback to 0)
    row = row.fillna(0)
    return row


# ── Prediction ────────────────────────────────────────────────────────────────

def round_score(x: float) -> int:
    """Round to nearest even number — 92% of NRL scores are even (tries+conversions+penalties).
    Field goals (1pt) are rare; rounding to even reflects the most likely outcome."""
    return int(round(x / 2)) * 2


def predict_match(
    home_team, away_team, pkgs,
    team_stats, h2h_stats,
    home_odds=None, away_odds=None,
    home_line=None, total_line=None,
    venue=None,
):
    feat = build_feature_row(
        home_team, away_team, team_stats, h2h_stats,
        home_odds=home_odds, away_odds=away_odds,
        home_line=home_line, total_line=total_line,
        venue=venue,
    )

    features = pkgs["home_score"]["features"]
    X = prep_feature_row(feat, features)

    pred_home_raw = float(predict_with_pkg(pkgs["home_score"], X)[0])
    pred_away_raw = float(predict_with_pkg(pkgs["away_score"], X)[0])
    pred_home_raw = max(0.0, pred_home_raw)
    pred_away_raw = max(0.0, pred_away_raw)

    pred_home = round_score(pred_home_raw)
    pred_away = round_score(pred_away_raw)

    home_win_prob = float(predict_proba_winner(pkgs["winner"], X)[0])

    winner     = home_team if home_win_prob >= 0.5 else away_team
    confidence = home_win_prob if home_win_prob >= 0.5 else (1 - home_win_prob)

    return {
        "home_team":     home_team,
        "away_team":     away_team,
        "pred_home":     pred_home,
        "pred_away":     pred_away,
        "pred_margin":   pred_home - pred_away,
        "pred_total":    pred_home + pred_away,
        "winner":        winner,
        "home_win_prob": round(home_win_prob * 100, 1),
        "away_win_prob": round((1 - home_win_prob) * 100, 1),
        "confidence":    round(confidence * 100, 1),
    }


def print_prediction(p: dict):
    home, away = p["home_team"], p["away_team"]
    w   = p["winner"]
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {home}  vs  {away}")
    print(f"{bar}")
    print(f"  Score:   {home} {p['pred_home']} – {p['pred_away']} {away}")
    print(f"  Margin:  {'+' if p['pred_margin'] >= 0 else ''}{p['pred_margin']} pts   "
          f"Total: {p['pred_total']} pts")
    print(f"  Winner:  {w}  ({p['confidence']:.1f}% confidence)")
    print(f"  Probs:   Home {p['home_win_prob']}%  /  Away {p['away_win_prob']}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", nargs=2, metavar=("HOME", "AWAY"))
    args = parser.parse_args()

    print("Loading v2 models and historical data...")
    pkgs = load_models()
    hist = pd.read_csv(DATA_DIR / "nrl_features.csv", parse_dates=["date"])
    team_stats = build_team_current_stats(hist)
    h2h_stats  = build_h2h_stats(hist)
    print(f"  {len(team_stats)} teams loaded, {len(hist)} historical games")

    if args.manual:
        home_team, away_team = args.manual
        p = predict_match(home_team, away_team, pkgs, team_stats, h2h_stats)
        print_prediction(p)
        return

    live_path = DATA_DIR / "live_odds.csv"
    if not live_path.exists():
        print("No live_odds.csv — run 04_fetch_live_odds.py first.")
        return

    live = pd.read_csv(live_path)
    print(f"\nGenerating predictions for {len(live)} upcoming fixtures...\n")

    results = []
    for _, row in live.iterrows():
        p = predict_match(
            row["home_team"], row["away_team"],
            pkgs, team_stats, h2h_stats,
            home_odds=row.get("home_odds_close"),
            away_odds=row.get("away_odds_close"),
            home_line=row.get("home_line_close"),
            total_line=row.get("total_line_close"),
            venue=row.get("venue"),
        )
        print_prediction(p)
        results.append(p)

    out = DATA_DIR / "predictions.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n{'─'*60}")
    print(f"Predictions saved → {out}")


if __name__ == "__main__":
    main()
