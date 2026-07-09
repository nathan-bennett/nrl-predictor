"""
Train NRL score prediction models.

Models:
  M1 — home_score  (XGBoost regression)
  M2 — away_score  (XGBoost regression)
  M3 — Dixon-Coles-style Poisson attack/defence ratings per team

Validation: time-series split — train on pre-2023, validate 2023-2024, test 2025+.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    # Odds / market
    "home_imp_prob_norm",
    "away_imp_prob_norm",
    "home_line_close",
    "total_line_close",
    "home_odds_move",
    "away_odds_move",
    # Rolling form
    "home_last5_scored_avg",
    "home_last5_conceded_avg",
    "home_last10_scored_avg",
    "home_last10_conceded_avg",
    "home_last5_win_pct",
    "home_last10_win_pct",
    "away_last5_scored_avg",
    "away_last5_conceded_avg",
    "away_last10_scored_avg",
    "away_last10_conceded_avg",
    "away_last5_win_pct",
    "away_last10_win_pct",
    # Season-to-date
    "home_ytd_scored_avg",
    "home_ytd_conceded_avg",
    "home_ytd_win_pct",
    "home_ytd_played",
    "away_ytd_scored_avg",
    "away_ytd_conceded_avg",
    "away_ytd_win_pct",
    "away_ytd_played",
    # H2H
    "h2h_home_win_pct",
    "h2h_count",
    # Context
    "home_rest_days",
    "away_rest_days",
    "away_travel_km",
    "is_playoff",
    "round_in_year",
    "dayofweek",
    "month",
]


def load():
    df = pd.read_csv(DATA_DIR / "nrl_features.csv", parse_dates=["date"])
    return df


def split(df):
    train = df[df["date"] < "2023-01-01"].copy()
    val = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-01-01")].copy()
    test = df[df["date"] >= "2025-01-01"].copy()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


def prepare_X(df, feature_cols):
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X


def train_xgb(X_train, y_train, X_val, y_val, label):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    rmse = root_mean_squared_error(y_val, val_pred)
    print(f"  [{label}] Val MAE={mae:.2f}  RMSE={rmse:.2f}  "
          f"best_iteration={model.best_iteration}")
    return model


def feature_importance_report(model, feature_cols, label):
    available = [c for c in feature_cols if c in feature_cols]
    fi = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    fi = fi.sort_values(ascending=False).head(15)
    print(f"\nTop-15 features [{label}]:")
    for feat, imp in fi.items():
        print(f"  {feat:<40} {imp:.4f}")


# ---------------------------------------------------------------------------
# Dixon-Coles Poisson model
# ---------------------------------------------------------------------------

def fit_dixon_coles(df):
    """
    Fit attack/defence ratings per team using a Poisson likelihood.
    rho correction for 0-0, 1-0, 0-1, 1-1 scorelines is included.
    """
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # params: [attack_0..n-1, defence_0..n-1, home_adv, rho]
    # defence of teams[0] is fixed to 1.0 for identifiability
    def pack(attack, defence, home_adv, rho):
        return np.concatenate([attack, defence[1:], [home_adv, rho]])

    def unpack(params):
        attack = params[:n]
        defence = np.concatenate([[1.0], params[n:2 * n - 1]])
        home_adv = params[2 * n - 1]
        rho = params[2 * n]
        return attack, defence, home_adv, rho

    def tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def neg_log_likelihood(params):
        attack, defence, home_adv, rho = unpack(params)
        ll = 0.0
        for _, row in df.iterrows():
            hi = team_idx[row["home_team"]]
            ai = team_idx[row["away_team"]]
            lam = np.exp(attack[hi] + defence[ai] + home_adv)  # expected home score
            mu = np.exp(attack[ai] + defence[hi])               # expected away score
            x, y = int(row["home_score"]), int(row["away_score"])
            t = tau(x, y, lam, mu, rho)
            if t <= 0:
                return 1e10
            ll += np.log(t) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
        return -ll

    print("\nFitting Dixon-Coles model (may take 1-2 min)...")
    attack0 = np.zeros(n)
    defence0 = np.zeros(n - 1)
    x0 = np.concatenate([attack0, defence0, [0.3, -0.1]])

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-6})

    attack, defence, home_adv, rho = unpack(result.x)
    ratings = pd.DataFrame({
        "team": teams,
        "attack": attack,
        "defence": defence,
        "attack_exp": np.exp(attack),
        "defence_exp": np.exp(defence),
    }).sort_values("attack_exp", ascending=False)

    print(f"  Converged: {result.success}  home_adv={home_adv:.3f}  rho={rho:.4f}")
    print("\nDixon-Coles Team Ratings:")
    print(ratings[["team", "attack_exp", "defence_exp"]].to_string(index=False))

    return {
        "teams": teams,
        "team_idx": team_idx,
        "attack": attack,
        "defence": defence,
        "home_adv": home_adv,
        "rho": rho,
    }


def dc_predict(dc_model, home_team, away_team, max_score=80):
    """Return expected scores and score probability matrix."""
    ti = dc_model["team_idx"]
    if home_team not in ti or away_team not in ti:
        return None
    hi, ai = ti[home_team], ti[away_team]
    lam = np.exp(dc_model["attack"][hi] + dc_model["defence"][ai] + dc_model["home_adv"])
    mu = np.exp(dc_model["attack"][ai] + dc_model["defence"][hi])
    return {"home_exp": lam, "away_exp": mu}


def evaluate_dc(dc_model, df):
    preds = []
    for _, row in df.iterrows():
        p = dc_predict(dc_model, row["home_team"], row["away_team"])
        if p:
            preds.append((p["home_exp"], p["away_exp"]))
        else:
            preds.append((np.nan, np.nan))
    pred_df = pd.DataFrame(preds, columns=["pred_home", "pred_away"])
    actual = df[["home_score", "away_score"]].reset_index(drop=True)
    valid = pred_df.dropna()
    home_mae = mean_absolute_error(actual.loc[valid.index, "home_score"], valid["pred_home"])
    away_mae = mean_absolute_error(actual.loc[valid.index, "away_score"], valid["pred_away"])
    home_rmse = root_mean_squared_error(actual.loc[valid.index, "home_score"], valid["pred_home"])
    away_rmse = root_mean_squared_error(actual.loc[valid.index, "away_score"], valid["pred_away"])
    print(f"  Dixon-Coles  home MAE={home_mae:.2f} RMSE={home_rmse:.2f}  "
          f"away MAE={away_mae:.2f} RMSE={away_rmse:.2f}")


def main():
    print("Loading data...")
    df = load()

    train, val, test = split(df)

    # ---- XGBoost models ----
    print("\n--- XGBoost Score Models ---")
    X_train = prepare_X(train, FEATURE_COLS)
    X_val = prepare_X(val, FEATURE_COLS)
    X_test = prepare_X(test, FEATURE_COLS)

    print("\nTraining home score model...")
    m_home = train_xgb(X_train, train["home_score"], X_val, val["home_score"], "home_score")

    print("Training away score model...")
    m_away = train_xgb(X_train, train["away_score"], X_val, val["away_score"], "away_score")

    feature_importance_report(m_home, FEATURE_COLS, "home_score")
    feature_importance_report(m_away, FEATURE_COLS, "away_score")

    # Test set evaluation
    print("\n--- Test Set (2025+) ---")
    if len(test) > 0:
        th = mean_absolute_error(test["home_score"], m_home.predict(X_test))
        ta = mean_absolute_error(test["away_score"], m_away.predict(X_test))
        rh = root_mean_squared_error(test["home_score"], m_home.predict(X_test))
        ra = root_mean_squared_error(test["away_score"], m_away.predict(X_test))
        print(f"  XGBoost  home MAE={th:.2f} RMSE={rh:.2f}  away MAE={ta:.2f} RMSE={ra:.2f}")
    else:
        print("  No 2025+ test data yet")

    # Save XGBoost models and feature list
    with open(MODEL_DIR / "model_home_score.pkl", "wb") as f:
        pickle.dump({"model": m_home, "features": list(X_train.columns)}, f)
    with open(MODEL_DIR / "model_away_score.pkl", "wb") as f:
        pickle.dump({"model": m_away, "features": list(X_train.columns)}, f)
    print("\nXGBoost models saved to models/")

    # ---- Dixon-Coles ----
    print("\n--- Dixon-Coles Poisson Model ---")
    dc_train = train[train["season"] >= 2019].copy()  # recent seasons for DC ratings
    dc_model = fit_dixon_coles(dc_train)

    print("\nDixon-Coles val performance:")
    evaluate_dc(dc_model, val)

    with open(MODEL_DIR / "model_dixon_coles.pkl", "wb") as f:
        pickle.dump(dc_model, f)
    print("Dixon-Coles model saved to models/")

    # ---- Baseline comparison ----
    print("\n--- Baseline: predict season average ---")
    home_avg = train["home_score"].mean()
    away_avg = train["away_score"].mean()
    base_home_mae = mean_absolute_error(val["home_score"], [home_avg] * len(val))
    base_away_mae = mean_absolute_error(val["away_score"], [away_avg] * len(val))
    print(f"  Mean baseline  home MAE={base_home_mae:.2f}  away MAE={base_away_mae:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
