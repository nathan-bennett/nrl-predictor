"""
NRL Predictor — Streamlit dashboard

Run:  streamlit run app.py
"""

import sqlite3
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

DB_PATH  = Path(__file__).parent / "data" / "nrl.db"
CSV_PATH = Path(__file__).parent / "data" / "nrl_features.csv"

st.set_page_config(
    page_title="NRL Predictor",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_conn():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300)
def load_predictions(_conn) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT f.date, f.home_team, f.away_team, f.venue,
               f.home_odds, f.away_odds,
               f.actual_home_score, f.actual_away_score, f.is_completed,
               p.pred_home_score, p.pred_away_score,
               p.pred_margin, p.pred_total,
               p.winner, p.home_win_prob, p.away_win_prob, p.confidence,
               p.created_at
        FROM   fixtures f
        JOIN   predictions p ON p.fixture_id = f.id
        WHERE  p.model_version = 'v2'
        ORDER  BY f.date ASC
    """, conn, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_accuracy(_conn) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT evaluated_at, n_games, winner_accuracy,
               home_score_mae, away_score_mae, margin_mae
        FROM   model_accuracy
        WHERE  model_version = 'v2'
        ORDER  BY evaluated_at DESC
        LIMIT  30
    """, conn, parse_dates=["evaluated_at"])


@st.cache_data(ttl=300)
def load_current_ladder_2026(_conn) -> pd.DataFrame:
    """Compute current 2026 NRL ladder from completed fixtures in DB."""
    try:
        df = pd.read_sql("""
            SELECT home_team AS team,
                   SUM(CASE WHEN actual_home_score > actual_away_score THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN actual_home_score < actual_away_score THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN actual_home_score = actual_away_score THEN 1 ELSE 0 END) AS draws,
                   SUM(actual_home_score) AS for_pts,
                   SUM(actual_away_score) AS against_pts,
                   COUNT(*) AS played
            FROM   fixtures
            WHERE  season = 2026 AND is_completed = 1
              AND  actual_home_score IS NOT NULL
            GROUP  BY home_team
            UNION ALL
            SELECT away_team AS team,
                   SUM(CASE WHEN actual_away_score > actual_home_score THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN actual_away_score < actual_home_score THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN actual_away_score = actual_home_score THEN 1 ELSE 0 END) AS draws,
                   SUM(actual_away_score) AS for_pts,
                   SUM(actual_home_score) AS against_pts,
                   COUNT(*) AS played
            FROM   fixtures
            WHERE  season = 2026 AND is_completed = 1
              AND  actual_away_score IS NOT NULL
            GROUP  BY away_team
        """, conn)
        if df.empty:
            return pd.DataFrame()
        agg = df.groupby("team", as_index=False).agg(
            W=("wins", "sum"), L=("losses", "sum"), D=("draws", "sum"),
            Played=("played", "sum"), For=("for_pts", "sum"), Against=("against_pts", "sum"),
        )
        agg["Points"] = agg["W"] * 2 + agg["D"]
        agg["Diff"]   = agg["For"] - agg["Against"]
        return agg.sort_values(["Points", "Diff"], ascending=[False, False]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_ladder(_conn) -> pd.DataFrame:
    """Load most recent season simulation (projected ladder) from DB."""
    try:
        latest = pd.read_sql("""
            SELECT simulated_at FROM season_simulation
            ORDER BY simulated_at DESC LIMIT 1
        """, conn)
        if latest.empty:
            return pd.DataFrame()
        ts = latest.iloc[0]["simulated_at"]
        return pd.read_sql("""
            SELECT team, proj_wins, proj_losses, proj_points,
                   proj_for, proj_against, proj_diff, proj_position
            FROM   season_simulation
            WHERE  simulated_at = ? AND model_version = 'no_odds'
            ORDER  BY proj_position ASC
        """, conn, params=(ts,))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_finals(_conn) -> pd.DataFrame:
    """Load most recent finals bracket simulation from DB."""
    try:
        latest = pd.read_sql("""
            SELECT simulated_at FROM finals_simulation
            ORDER BY simulated_at DESC LIMIT 1
        """, conn)
        if latest.empty:
            return pd.DataFrame()
        ts = latest.iloc[0]["simulated_at"]
        return pd.read_sql("""
            SELECT round_name, match_order, home_team, away_team,
                   pred_home_score, pred_away_score, pred_winner,
                   home_win_prob, confidence
            FROM   finals_simulation
            WHERE  simulated_at = ? AND model_version = 'no_odds'
            ORDER  BY
                CASE round_name
                    WHEN 'Qualifying Final 1 (1v2)' THEN 1
                    WHEN 'Qualifying Final 2 (3v4)' THEN 2
                    WHEN 'Elimination Final 1 (5v8)' THEN 3
                    WHEN 'Elimination Final 2 (6v7)' THEN 4
                    WHEN 'Semi Final 1' THEN 5
                    WHEN 'Semi Final 2' THEN 6
                    WHEN 'Preliminary Final 1' THEN 7
                    WHEN 'Preliminary Final 2' THEN 8
                    WHEN 'Grand Final' THEN 9
                    ELSE 10
                END
        """, conn, params=(ts,))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_team_form() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    rows = []
    for team in teams:
        home_g = df[df["home_team"] == team]
        away_g = df[df["away_team"] == team]
        # Last known ELO
        last_home = home_g.sort_values("date").iloc[-1] if len(home_g) else None
        last_away = away_g.sort_values("date").iloc[-1] if len(away_g) else None
        elo = (last_home["home_elo"] if last_home is not None
               else last_away["away_elo"] if last_away is not None else 1500)
        streak_val = (last_home["home_streak"] if last_home is not None
                      else last_away["away_streak"] if last_away is not None else 0)

        # Season win %
        season = df["date"].dt.year.max()
        season_home = df[(df["home_team"] == team) & (df["season"] == season)]
        season_away = df[(df["away_team"] == team) & (df["season"] == season)]
        wins = (season_home["result"] == 1).sum() + (season_away["result"] == -1).sum()
        played = len(season_home) + len(season_away)

        # Last 5 scored/conceded (all games)
        all_games = pd.concat([
            season_home[["date", "home_score", "away_score"]].rename(
                columns={"home_score": "scored", "away_score": "conceded"}),
            season_away[["date", "home_score", "away_score"]].rename(
                columns={"away_score": "scored", "home_score": "conceded"}),
        ]).sort_values("date").tail(5)

        rows.append({
            "Team": team,
            "ELO": round(elo),
            f"{season} W": wins,
            f"{season} P": played,
            f"{season} Win%": f"{wins/played:.0%}" if played else "-",
            "L5 Scored": round(all_games["scored"].mean(), 1) if len(all_games) else "-",
            "L5 Conceded": round(all_games["conceded"].mean(), 1) if len(all_games) else "-",
            "Streak": int(streak_val),
        })

    return pd.DataFrame(rows).sort_values("ELO", ascending=False).reset_index(drop=True)


def confidence_colour(conf: float) -> str:
    if conf >= 70:
        return "#1a7a1a"
    if conf >= 60:
        return "#4a8f1a"
    if conf >= 55:
        return "#8f8f00"
    return "#8f4a00"


def result_icon(row) -> str:
    if not row["is_completed"]:
        return ""
    pred_w = row["winner"]
    act_w  = row["home_team"] if row["actual_home_score"] > row["actual_away_score"] else row["away_team"]
    return "✅" if pred_w == act_w else "❌"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏉 NRL Predictor")
    st.caption("Stack v2 model · 55.8% test accuracy")
    st.divider()

    page = st.radio("View", ["Predictions", "Results Tracker", "Team Form",
                              "Ladder Predictor", "Finals Predictor"])

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    conn = get_conn()
    if conn is None:
        st.warning("DB not found.\nRun `python3 08_update_db.py` first.")

# ── No DB fallback ────────────────────────────────────────────────────────────

if conn is None:
    st.info("No database found. Run `python3 08_update_db.py` to populate it.")

    # Fall back to live_odds.csv + predictions.csv if they exist
    live_path = Path(__file__).parent / "data" / "live_odds.csv"
    pred_path = Path(__file__).parent / "data" / "predictions.csv"

    if pred_path.exists() and live_path.exists():
        st.subheader("Latest predictions (from CSV)")
        preds = pd.read_csv(pred_path)
        odds  = pd.read_csv(live_path)
        merged = preds.merge(odds[["home_team", "away_team", "date", "venue",
                                   "home_odds_close", "away_odds_close"]],
                             on=["home_team", "away_team"], how="left")
        st.dataframe(merged, use_container_width=True)
    st.stop()


df_preds = load_predictions(conn)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════

if page == "Predictions":
    st.title("Upcoming Predictions")

    upcoming = df_preds[~df_preds["is_completed"].astype(bool)].copy()
    completed = df_preds[df_preds["is_completed"].astype(bool)].copy()

    if upcoming.empty:
        st.info("No upcoming fixtures in the database. Run `python3 08_update_db.py` to update.")
    else:
        # Group by date
        upcoming["date_label"] = upcoming["date"].dt.strftime("%A %-d %B")
        for date_label, grp in upcoming.groupby("date_label", sort=False):
            st.subheader(f"📅 {date_label}")
            for _, row in grp.iterrows():
                home, away = row["home_team"], row["away_team"]
                winner     = row["winner"]
                conf       = row["confidence"]
                col1, col2, col3 = st.columns([3, 2, 2])

                with col1:
                    margin_str = (f"+{row['pred_margin']}" if row["pred_margin"] >= 0
                                  else str(row["pred_margin"]))
                    st.markdown(
                        f"**{home}** vs **{away}**  \n"
                        f"📍 {row['venue'] or '—'}"
                    )

                with col2:
                    st.metric(
                        label="Predicted Score",
                        value=f"{row['pred_home_score']} – {row['pred_away_score']}",
                        delta=f"{margin_str} pts  |  Total {row['pred_total']}",
                    )

                with col3:
                    colour = confidence_colour(conf)
                    h_prob = row["home_win_prob"]
                    a_prob = row["away_win_prob"]
                    st.markdown(
                        f"<div style='background:{colour};padding:10px;border-radius:8px;"
                        f"color:white;text-align:center'>"
                        f"<b>{winner}</b><br>"
                        f"<span style='font-size:1.3em'>{conf:.0f}%</span> confidence<br>"
                        f"<small>{home} {h_prob:.0f}% / {away} {a_prob:.0f}%</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RESULTS TRACKER
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Results Tracker":
    st.title("Prediction Accuracy")

    completed = df_preds[df_preds["is_completed"].astype(bool)].copy()

    if completed.empty:
        st.info("No completed games with predictions yet.")
    else:
        completed["correct"] = completed.apply(
            lambda r: r["winner"] == (r["home_team"] if r["actual_home_score"] > r["actual_away_score"]
                                      else r["away_team"]), axis=1)
        completed["home_err"]   = (completed["actual_home_score"] - completed["pred_home_score"]).abs()
        completed["away_err"]   = (completed["actual_away_score"] - completed["pred_away_score"]).abs()
        completed["margin_err"] = ((completed["actual_home_score"] - completed["actual_away_score"]) -
                                    (completed["pred_home_score"]   - completed["pred_away_score"])).abs()

        # Headline metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games tracked", len(completed))
        c2.metric("Winner accuracy", f"{completed['correct'].mean():.1%}")
        c3.metric("Score MAE (home)", f"{completed['home_err'].mean():.1f} pts")
        c4.metric("Score MAE (away)", f"{completed['away_err'].mean():.1f} pts")

        st.divider()

        # Rolling accuracy chart
        completed_sorted = completed.sort_values("date")
        completed_sorted["rolling_acc"] = (
            completed_sorted["correct"].expanding().mean() * 100
        )
        st.subheader("Rolling winner accuracy")
        st.line_chart(
            completed_sorted.set_index("date")[["rolling_acc"]].rename(
                columns={"rolling_acc": "Accuracy (%)"}),
            use_container_width=True,
        )

        # Detailed results table
        st.subheader("Game-by-game results")
        display = completed_sorted[[
            "date", "home_team", "away_team",
            "pred_home_score", "pred_away_score",
            "actual_home_score", "actual_away_score",
            "winner", "correct", "home_err", "margin_err",
        ]].copy()
        display["date"] = display["date"].dt.strftime("%-d %b %Y")
        display["correct"] = display["correct"].map({True: "✅", False: "❌"})
        display.columns = [
            "Date", "Home", "Away",
            "Pred H", "Pred A",
            "Act H", "Act A",
            "Pred Winner", "✓", "Score Err", "Margin Err",
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

    # Historical accuracy snapshots
    df_acc = load_accuracy(conn)
    if not df_acc.empty:
        st.divider()
        st.subheader("Model accuracy snapshots")
        st.dataframe(df_acc, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TEAM FORM
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Team Form":
    st.title("Team Form & ELO Ratings")

    form_df = load_team_form()
    if form_df.empty:
        st.info("Historical data not found.")
    else:
        st.caption(
            "ELO ratings and form stats computed from historical games up to last dataset update. "
            "Higher ELO = stronger long-run team."
        )

        # Highlight top/bottom ELO
        def style_elo(val):
            if isinstance(val, (int, float)):
                if val > 1550:
                    return "background-color: #d4edda; color: #155724"
                if val < 1450:
                    return "background-color: #f8d7da; color: #721c24"
            return ""

        def style_streak(val):
            if isinstance(val, (int, float)):
                if val >= 3:
                    return "color: green; font-weight: bold"
                if val <= -3:
                    return "color: red; font-weight: bold"
            return ""

        styled = (
            form_df.style
            .map(style_elo, subset=["ELO"])
            .map(style_streak, subset=["Streak"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ELO bar chart
        st.subheader("ELO Ratings")
        chart_df = form_df.set_index("Team")[["ELO"]].sort_values("ELO", ascending=True)
        st.bar_chart(chart_df, use_container_width=True)

        st.caption(
            "**Streak**: positive = consecutive wins, negative = consecutive losses.  "
            "**L5 Scored/Conceded**: average over last 5 games this season."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — LADDER PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Ladder Predictor":
    st.title("2026 Projected Final Ladder")
    st.caption(
        "Predicted using the no-odds model (ELO + form + context).  "
        "Run `python3 10_fetch_draw_2026.py` then `python3 11_simulate_season.py` to update."
    )

    current_df = load_current_ladder_2026(conn)
    ladder_df  = load_ladder(conn)

    if ladder_df.empty:
        st.info(
            "No simulation data found.  \n"
            "1. `python3 10_fetch_draw_2026.py` — fetch the full 2026 draw  \n"
            "2. `python3 09_train_no_odds.py` — train the no-odds model  \n"
            "3. `python3 11_simulate_season.py` — run the season simulation"
        )
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Current Ladder (actual)")
            if current_df.empty:
                st.info("No completed 2026 games in DB yet. Run `python3 10_fetch_draw_2026.py` to import results.")
            else:
                curr = current_df[["team", "W", "L", "Played", "For", "Against", "Points", "Diff"]].copy()
                curr.index = curr.index + 1
                curr.columns = ["Team", "W", "L", "P", "For", "Against", "Pts", "Diff"]

                def style_curr(row):
                    pos = row.name
                    if pos <= 4:   return ["background-color:#cce5ff"] * len(row)
                    if pos <= 8:   return ["background-color:#d4edda"] * len(row)
                    return [""] * len(row)

                games_played = int(current_df["Played"].max()) if not current_df.empty else 0
                st.dataframe(curr.style.apply(style_curr, axis=1), use_container_width=True)
                st.caption(f"After round {games_played // 2 if games_played else '?'} (approx.)  |  Blue = Top 4  |  Green = Finals (5–8)")

        with col_b:
            st.subheader("Projected End-of-Season")
            projected = ladder_df[[
                "team", "proj_position", "proj_wins", "proj_losses",
                "proj_points", "proj_for", "proj_against", "proj_diff",
            ]].copy()
            projected.columns = ["Team", "Pos", "W", "L", "Pts", "For", "Against", "Diff"]
            projected = projected.set_index("Pos")

            def style_proj(row):
                pos = row.name
                if pos <= 4:   return ["background-color:#cce5ff"] * len(row)
                if pos <= 8:   return ["background-color:#d4edda"] * len(row)
                return [""] * len(row)

            st.dataframe(projected.style.apply(style_proj, axis=1), use_container_width=True)
            st.caption("Blue = Top 4 (double chance)  |  Green = Finals (5–8)")

        st.divider()
        st.subheader("Win Projection")
        chart_data = projected[["Team", "W"]].set_index("Team").sort_values("W", ascending=True)
        st.bar_chart(chart_data, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — FINALS PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Finals Predictor":
    st.title("2026 Finals Series Predictor")
    st.caption(
        "Simulated using the no-odds model.  "
        "Run `python3 11_simulate_season.py` to refresh."
    )

    finals_df = load_finals(conn)
    ladder_df = load_ladder(conn)

    if finals_df.empty:
        st.info(
            "No finals simulation found.  \n"
            "Run `python3 11_simulate_season.py` to generate predictions."
        )
    else:
        # Top 8 banner
        if not ladder_df.empty:
            top8 = ladder_df[ladder_df["proj_position"] <= 8].sort_values("proj_position")
            st.subheader("Projected Top 8")
            cols = st.columns(8)
            for i, (_, row) in enumerate(top8.iterrows()):
                pos = int(row["proj_position"])
                label = "DC" if pos <= 4 else "FIN"
                color = "#cce5ff" if pos <= 4 else "#d4edda"
                cols[i].markdown(
                    f"<div style='background:{color};padding:6px;border-radius:6px;"
                    f"text-align:center;font-size:0.8em'>"
                    f"<b>{pos}</b><br>{row['team'].replace(' ', '<br>')}</div>",
                    unsafe_allow_html=True,
                )
            st.caption("Blue = double chance (Top 4)  |  Green = finals only (5–8)")
            st.divider()

        # Bracket grouped by week
        week_order = {
            "Qualifying Final":  1,
            "Elimination Final": 1,
            "Semi Final":        2,
            "Preliminary Final": 3,
            "Grand Final":       4,
        }

        def week_of(name):
            for kw, w in week_order.items():
                if kw in name:
                    return w
            return 5

        finals_df["week"] = finals_df["round_name"].apply(week_of)
        week_labels = {1: "Week 1 — Qualifying & Elimination Finals",
                       2: "Week 2 — Semi Finals",
                       3: "Week 3 — Preliminary Finals",
                       4: "Week 4 — Grand Final"}

        for week_num in sorted(finals_df["week"].unique()):
            week_games = finals_df[finals_df["week"] == week_num].copy()
            st.subheader(week_labels.get(week_num, f"Week {week_num}"))

            for _, row in week_games.iterrows():
                home, away  = row["home_team"], row["away_team"]
                winner      = row["pred_winner"]
                conf        = row["confidence"]
                h_prob      = row["home_win_prob"]
                a_prob      = 100 - h_prob
                ph          = row["pred_home_score"]
                pa          = row["pred_away_score"]
                margin      = ph - pa
                margin_str  = f"+{margin}" if margin >= 0 else str(margin)

                colour = confidence_colour(conf)
                is_gf  = "Grand Final" in row["round_name"]

                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    prefix = "🏆 " if is_gf else ""
                    st.markdown(
                        f"**{prefix}{home}** vs **{away}**  \n"
                        f"*{row['round_name']}*"
                    )
                with c2:
                    st.metric(
                        label="Predicted Score",
                        value=f"{ph} – {pa}",
                        delta=f"{margin_str} pts",
                    )
                with c3:
                    trophy = "🏆" if is_gf else ""
                    st.markdown(
                        f"<div style='background:{colour};padding:10px;border-radius:8px;"
                        f"color:white;text-align:center'>"
                        f"<b>{trophy} {winner}</b><br>"
                        f"<span style='font-size:1.3em'>{conf:.0f}%</span> confidence<br>"
                        f"<small>{home} {h_prob:.0f}% / {away} {a_prob:.0f}%</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.divider()
