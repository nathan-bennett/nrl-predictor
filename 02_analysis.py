"""
Exploratory analysis of the enriched NRL dataset.
Outputs plots to data/plots/ and prints key stats.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load():
    df = pd.read_csv(DATA_DIR / "nrl_features.csv", parse_dates=["date"])
    return df


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def score_distributions(df):
    print_section("Score Distributions")
    print(f"Home score  — mean: {df['home_score'].mean():.1f}, std: {df['home_score'].std():.1f}, "
          f"min: {df['home_score'].min()}, max: {df['home_score'].max()}")
    print(f"Away score  — mean: {df['away_score'].mean():.1f}, std: {df['away_score'].std():.1f}, "
          f"min: {df['away_score'].min()}, max: {df['away_score'].max()}")
    print(f"Total score — mean: {df['total_score'].mean():.1f}, std: {df['total_score'].std():.1f}")
    print(f"Margin      — mean: {df['margin'].mean():.1f}, std: {df['margin'].std():.1f}")

    home_win = (df["result"] == 1).mean()
    away_win = (df["result"] == -1).mean()
    draw = (df["result"] == 0).mean()
    print(f"\nHome win: {home_win:.1%}  Away win: {away_win:.1%}  Draw: {draw:.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("NRL Score Distributions (2009–2026)", fontsize=14, fontweight="bold")

    axes[0, 0].hist(df["home_score"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Home Score")
    axes[0, 0].set_xlabel("Points")

    axes[0, 1].hist(df["away_score"], bins=30, color="coral", edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Away Score")
    axes[0, 1].set_xlabel("Points")

    axes[1, 0].hist(df["total_score"], bins=35, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[1, 0].set_title("Total Score")
    axes[1, 0].set_xlabel("Points")

    axes[1, 1].hist(df["margin"], bins=35, color="seagreen", edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1, 1].set_title("Margin (Home − Away)")
    axes[1, 1].set_xlabel("Points")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "score_distributions.png", dpi=150)
    plt.close()
    print("  → saved score_distributions.png")


def odds_calibration(df):
    print_section("Odds Calibration")
    df2 = df.dropna(subset=["home_imp_prob_norm"])

    bins = np.linspace(0.1, 0.9, 10)
    df2 = df2.copy()
    df2["prob_bin"] = pd.cut(df2["home_imp_prob_norm"], bins=bins)
    cal = df2.groupby("prob_bin", observed=True).agg(
        predicted=("home_imp_prob_norm", "mean"),
        actual=("result", lambda x: (x == 1).mean()),
        count=("result", "count"),
    ).dropna()

    print(cal.to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    sc = ax.scatter(cal["predicted"], cal["actual"], c=cal["count"],
                    cmap="viridis", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="Sample count")
    ax.set_xlabel("Bookmaker implied probability (normalised)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Odds Calibration — Home Win")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "odds_calibration.png", dpi=150)
    plt.close()
    print("  → saved odds_calibration.png")


def home_advantage_by_team(df):
    print_section("Home Advantage by Team")
    team_stats = []
    for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique())):
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        h_win = (home["result"] == 1).mean() if len(home) > 5 else np.nan
        a_win = (away["result"] == -1).mean() if len(away) > 5 else np.nan
        h_scored = home["home_score"].mean() if len(home) > 5 else np.nan
        a_scored = away["away_score"].mean() if len(away) > 5 else np.nan
        team_stats.append({"team": team, "home_win_pct": h_win, "away_win_pct": a_win,
                            "home_scored_avg": h_scored, "away_scored_avg": a_scored,
                            "home_games": len(home), "away_games": len(away)})
    stats_df = pd.DataFrame(team_stats).dropna().sort_values("home_win_pct", ascending=False)
    print(stats_df[["team", "home_win_pct", "away_win_pct", "home_scored_avg", "away_scored_avg"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats_df))
    width = 0.35
    ax.bar(x - width / 2, stats_df["home_win_pct"], width, label="Home win %", color="steelblue")
    ax.bar(x + width / 2, stats_df["away_win_pct"], width, label="Away win %", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["team"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win %")
    ax.set_title("Home vs Away Win Rate by Team")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "home_advantage_by_team.png", dpi=150)
    plt.close()
    print("  → saved home_advantage_by_team.png")


def feature_correlations(df):
    print_section("Feature Correlations with Home Score")
    feat_cols = [
        "home_imp_prob_norm", "home_line_close", "total_line_close",
        "home_last5_scored_avg", "home_last5_conceded_avg",
        "home_ytd_scored_avg", "home_ytd_conceded_avg",
        "home_ytd_win_pct", "home_last5_win_pct",
        "away_last5_conceded_avg", "away_ytd_conceded_avg",
        "home_rest_days", "away_rest_days", "away_travel_km",
        "h2h_home_win_pct", "round_in_year",
    ]
    available = [c for c in feat_cols if c in df.columns]
    corr = df[available + ["home_score", "away_score", "total_score", "margin"]].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_correlations.png", dpi=150)
    plt.close()
    print("  → saved feature_correlations.png")

    target_corr = corr[["home_score", "away_score", "total_score"]].drop(
        ["home_score", "away_score", "total_score", "margin"], errors="ignore"
    ).sort_values("home_score", ascending=False)
    print("\nCorrelation with home_score:")
    print(target_corr["home_score"].to_string())


def score_trends(df):
    print_section("Score Trends Over Time")
    yearly = df.groupby("season").agg(
        home_score=("home_score", "mean"),
        away_score=("away_score", "mean"),
        total_score=("total_score", "mean"),
        games=("home_score", "count"),
    ).reset_index()
    print(yearly.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["season"], yearly["home_score"], "o-", label="Home score avg", color="steelblue")
    ax.plot(yearly["season"], yearly["away_score"], "s-", label="Away score avg", color="coral")
    ax.plot(yearly["season"], yearly["total_score"], "^--", label="Total avg", color="mediumpurple", alpha=0.7)
    ax.set_xlabel("Season")
    ax.set_ylabel("Average points")
    ax.set_title("NRL Average Scores by Season")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "score_trends.png", dpi=150)
    plt.close()
    print("  → saved score_trends.png")


def handicap_accuracy(df):
    print_section("Handicap Line Accuracy")
    df2 = df.dropna(subset=["home_line_close"])
    # Handicap line predicts home margin; positive = home favoured by that amount
    df2 = df2.copy()
    df2["line_pred_margin"] = -df2["home_line_close"]  # line is points given to away; negate for home margin
    df2["line_error"] = df2["margin"] - df2["line_pred_margin"]
    mae = df2["line_error"].abs().mean()
    rmse = np.sqrt((df2["line_error"] ** 2).mean())
    print(f"Handicap line MAE:  {mae:.2f} pts")
    print(f"Handicap line RMSE: {rmse:.2f} pts")
    print(f"(This is the baseline our model must beat)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df2["line_error"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Actual margin − Predicted margin (pts)")
    ax.set_title(f"Handicap Line Prediction Error  (MAE={mae:.1f}, RMSE={rmse:.1f})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "handicap_accuracy.png", dpi=150)
    plt.close()
    print("  → saved handicap_accuracy.png")


def main():
    print("Loading enriched data...")
    df = load()
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    score_distributions(df)
    odds_calibration(df)
    home_advantage_by_team(df)
    feature_correlations(df)
    score_trends(df)
    handicap_accuracy(df)

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
