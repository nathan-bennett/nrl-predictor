"""
Systematic model selection across:
  - Multiple algorithms
  - Hyperparameter search (RandomizedSearchCV)
  - Feature ablation (odds-only, form-only, full, no-odds)

Targets:
  A) Winner classification  (home win / draw / away win)
  B) Score regression       (home_score, away_score) — re-tunes the existing models

Saves:
  models/model_winner.pkl          — best winner classifier
  models/model_home_score_v2.pkl   — best home score regressor
  models/model_away_score_v2.pkl   — best away score regressor
  data/model_selection_results.csv — full experiment log
"""

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    mean_absolute_error, root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

# ── Feature groups for ablation ──────────────────────────────────────────────
ODDS_FEATURES = [
    "home_imp_prob_norm", "away_imp_prob_norm",
    "home_line_close", "total_line_close",
    "home_odds_move", "away_odds_move",
]
FORM_FEATURES = [
    "home_last5_win_pct", "home_last10_win_pct",
    "home_last5_scored_avg", "home_last5_conceded_avg",
    "home_last10_scored_avg", "home_last10_conceded_avg",
    "away_last5_win_pct", "away_last10_win_pct",
    "away_last5_scored_avg", "away_last5_conceded_avg",
    "away_last10_scored_avg", "away_last10_conceded_avg",
]
SEASON_FEATURES = [
    "home_ytd_win_pct", "home_ytd_scored_avg", "home_ytd_conceded_avg", "home_ytd_played",
    "away_ytd_win_pct", "away_ytd_scored_avg", "away_ytd_conceded_avg", "away_ytd_played",
]
CONTEXT_FEATURES = [
    "h2h_home_win_pct", "h2h_count",
    "home_rest_days", "away_rest_days",
    "away_travel_km", "is_playoff",
    "round_in_year", "dayofweek", "month",
]
ALL_FEATURES = ODDS_FEATURES + FORM_FEATURES + SEASON_FEATURES + CONTEXT_FEATURES

FEATURE_GROUPS = {
    "odds_only":    ODDS_FEATURES,
    "form_only":    FORM_FEATURES + SEASON_FEATURES,
    "no_odds":      FORM_FEATURES + SEASON_FEATURES + CONTEXT_FEATURES,
    "full":         ALL_FEATURES,
}


def load_splits():
    df = pd.read_csv(DATA_DIR / "nrl_features.csv", parse_dates=["date"])
    train = df[df["date"] < "2023-01-01"].copy()
    val   = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-01-01")].copy()
    test  = df[df["date"] >= "2025-01-01"].copy()
    return train, val, test


def prep(df, features):
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    return X


def winner_label(df):
    """1=home win, 0=draw, -1=away win  →  mapped to 0/1/2 for sklearn."""
    return df["result"].map({1: 0, 0: 1, -1: 2}).values


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


# ── Classification experiments ───────────────────────────────────────────────

def run_classification(train, val, test):
    print_section("WINNER CLASSIFICATION")
    y_train = winner_label(train)
    y_val   = winner_label(val)
    y_test  = winner_label(test)

    results = []

    # ── 1. Feature ablation with a fixed XGBoost ────────────────────────────
    print("\n[1] Feature group ablation (XGBoost, fixed params)")
    base_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              eval_metric="mlogloss", verbosity=0, random_state=42)
    for group_name, features in FEATURE_GROUPS.items():
        Xtr = prep(train, features)
        Xv  = prep(val,   features)
        base_xgb.fit(Xtr, y_train, eval_set=[(Xv, y_val)], verbose=False)
        pred   = base_xgb.predict(Xv)
        prob   = base_xgb.predict_proba(Xv)
        acc    = accuracy_score(y_val, pred)
        ll     = log_loss(y_val, prob)
        binary = (y_val != 1).astype(int)    # home or away win (ignore draws)
        prob2  = prob[:, [0, 2]]; prob2 = prob2 / prob2.sum(axis=1, keepdims=True)
        print(f"  {group_name:<12}  val acc={acc:.3f}  log-loss={ll:.4f}  n_feats={len(Xtr.columns)}")
        results.append({"task": "winner", "model": "XGBoost", "features": group_name,
                        "val_acc": acc, "val_logloss": ll, "params": "fixed"})

    # ── 2. Algorithm comparison on full feature set ──────────────────────────
    print("\n[2] Algorithm comparison (full features, val set)")
    Xtr_full = prep(train, ALL_FEATURES)
    Xv_full  = prep(val,   ALL_FEATURES)

    algorithms = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1,
                                       solver="lbfgs", random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "XGBoost_default": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            verbosity=0, random_state=42),
    }

    for name, model in algorithms.items():
        if name == "XGBoost_default":
            model.fit(Xtr_full, y_train, eval_set=[(Xv_full, y_val)], verbose=False)
        else:
            model.fit(Xtr_full, y_train)
        pred = model.predict(Xv_full)
        prob = model.predict_proba(Xv_full)
        acc  = accuracy_score(y_val, pred)
        ll   = log_loss(y_val, prob)
        print(f"  {name:<22}  val acc={acc:.3f}  log-loss={ll:.4f}")
        results.append({"task": "winner", "model": name, "features": "full",
                        "val_acc": acc, "val_logloss": ll, "params": "default"})

    # ── 3. Hyperparameter search on XGBoost ─────────────────────────────────
    print("\n[3] XGBoost hyperparameter search (RandomizedSearchCV, 40 trials)")
    param_dist = {
        "n_estimators":     randint(100, 600),
        "learning_rate":    uniform(0.01, 0.15),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(3, 20),
        "reg_alpha":        uniform(0, 0.5),
        "reg_lambda":       uniform(0.5, 2.0),
        "gamma":            uniform(0, 0.3),
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(eval_metric="mlogloss", verbosity=0, random_state=42),
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_log_loss",
        cv=4,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    xgb_search.fit(Xtr_full, y_train)
    best_xgb = xgb_search.best_estimator_
    pred  = best_xgb.predict(Xv_full)
    prob  = best_xgb.predict_proba(Xv_full)
    acc   = accuracy_score(y_val, pred)
    ll    = log_loss(y_val, prob)
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  Val acc={acc:.3f}  log-loss={ll:.4f}")
    results.append({"task": "winner", "model": "XGBoost_tuned", "features": "full",
                    "val_acc": acc, "val_logloss": ll, "params": str(xgb_search.best_params_)})

    # ── 4. Test set evaluation of best model ────────────────────────────────
    print_section("WINNER — TEST SET (2025+)")
    Xtest_full = prep(test, ALL_FEATURES)

    # Calibrate the best XGBoost (improves probability estimates)
    calibrated = CalibratedClassifierCV(best_xgb, method="isotonic")
    calibrated.fit(Xtr_full, y_train)

    for label, model in [("XGBoost_tuned (uncal)", best_xgb),
                          ("XGBoost_tuned (cal)",   calibrated),
                          ("LogisticRegression",    algorithms["LogisticRegression"]),
                          ("Baseline (majority)",   None)]:
        if model is None:
            pred  = np.zeros(len(y_test), dtype=int)  # always predict home win
            prob  = np.column_stack([np.full(len(y_test), 0.567),
                                     np.full(len(y_test), 0.004),
                                     np.full(len(y_test), 0.429)])
        else:
            pred = model.predict(Xtest_full)
            prob = model.predict_proba(Xtest_full)
        acc = accuracy_score(y_test, pred)
        ll  = log_loss(y_test, prob)
        print(f"  {label:<30}  acc={acc:.3f}  log-loss={ll:.4f}")

    # Save best model (calibrated XGBoost, full features)
    with open(MODEL_DIR / "model_winner.pkl", "wb") as f:
        pickle.dump({
            "model": calibrated,
            "features": ALL_FEATURES,
            "classes": {0: "home_win", 1: "draw", 2: "away_win"},
        }, f)
    print("\n  Saved model_winner.pkl")

    # Feature importance
    fi = pd.Series(best_xgb.feature_importances_, index=Xtr_full.columns)
    fi = fi.sort_values(ascending=False).head(15)
    print("\nTop-15 features (winner model):")
    for feat, imp in fi.items():
        print(f"  {feat:<40} {imp:.4f}")

    return pd.DataFrame(results)


# ── Regression experiments ───────────────────────────────────────────────────

def run_regression(train, val, test, target):
    print_section(f"SCORE REGRESSION — {target.upper()}")
    y_train = train[target].values
    y_val   = val[target].values
    y_test  = test[target].values

    results = []

    # ── 1. Feature ablation ──────────────────────────────────────────────────
    print("\n[1] Feature group ablation (XGBoost, fixed params)")
    base_xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             eval_metric="rmse", verbosity=0, random_state=42)
    for group_name, features in FEATURE_GROUPS.items():
        Xtr = prep(train, features)
        Xv  = prep(val,   features)
        base_xgb.fit(Xtr, y_train, eval_set=[(Xv, y_val)], verbose=False)
        pred = base_xgb.predict(Xv)
        mae  = mean_absolute_error(y_val, pred)
        rmse = root_mean_squared_error(y_val, pred)
        print(f"  {group_name:<12}  val MAE={mae:.2f}  RMSE={rmse:.2f}")
        results.append({"task": target, "model": "XGBoost", "features": group_name,
                        "val_mae": mae, "val_rmse": rmse, "params": "fixed"})

    # ── 2. Algorithm comparison ──────────────────────────────────────────────
    print("\n[2] Algorithm comparison (full features)")
    Xtr_full = prep(train, ALL_FEATURES)
    Xv_full  = prep(val,   ALL_FEATURES)

    regressors = {
        "Ridge":         Pipeline([("scaler", StandardScaler()),
                                   ("reg", Ridge(alpha=10.0))]),
        "RandomForest":  RandomForestRegressor(n_estimators=300, max_depth=8,
                                               min_samples_leaf=10, random_state=42),
        "XGBoost_default": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8,
                                        eval_metric="rmse", verbosity=0, random_state=42),
    }

    for name, model in regressors.items():
        if name == "XGBoost_default":
            model.fit(Xtr_full, y_train, eval_set=[(Xv_full, y_val)], verbose=False)
        else:
            model.fit(Xtr_full, y_train)
        pred = model.predict(Xv_full)
        mae  = mean_absolute_error(y_val, pred)
        rmse = root_mean_squared_error(y_val, pred)
        print(f"  {name:<22}  val MAE={mae:.2f}  RMSE={rmse:.2f}")
        results.append({"task": target, "model": name, "features": "full",
                        "val_mae": mae, "val_rmse": rmse, "params": "default"})

    # ── 3. Hyperparameter search ─────────────────────────────────────────────
    print("\n[3] XGBoost hyperparameter search (40 trials)")
    param_dist = {
        "n_estimators":     randint(100, 700),
        "learning_rate":    uniform(0.01, 0.15),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(3, 20),
        "reg_alpha":        uniform(0, 0.5),
        "reg_lambda":       uniform(0.5, 2.0),
        "gamma":            uniform(0, 0.3),
    }
    search = RandomizedSearchCV(
        XGBRegressor(eval_metric="rmse", verbosity=0, random_state=42),
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=4,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(Xtr_full, y_train)
    best = search.best_estimator_
    pred = best.predict(Xv_full)
    mae  = mean_absolute_error(y_val, pred)
    rmse = root_mean_squared_error(y_val, pred)
    print(f"  Best params: {search.best_params_}")
    print(f"  Val MAE={mae:.2f}  RMSE={rmse:.2f}")
    results.append({"task": target, "model": "XGBoost_tuned", "features": "full",
                    "val_mae": mae, "val_rmse": rmse, "params": str(search.best_params_)})

    # ── 4. Test set ──────────────────────────────────────────────────────────
    Xtest_full = prep(test, ALL_FEATURES)
    print(f"\n  Test set (2025+):")
    train_mean = y_train.mean()
    for label, model in [("XGBoost_tuned", best),
                          ("Ridge",          regressors["Ridge"]),
                          ("Baseline (mean)", None)]:
        if model is None:
            pred = np.full(len(y_test), train_mean)
        else:
            pred = model.predict(Xtest_full)
        mae  = mean_absolute_error(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred)
        print(f"    {label:<22}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    key = "v2" if target == "home_score" else "v2"
    fname = f"model_{target}_v2.pkl"
    with open(MODEL_DIR / fname, "wb") as f:
        pickle.dump({"model": best, "features": list(Xtr_full.columns)}, f)
    print(f"\n  Saved {fname}")

    return pd.DataFrame(results)


def main():
    print("Loading data...")
    train, val, test = load_splits()
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

    all_results = []

    clf_results = run_classification(train, val, test)
    all_results.append(clf_results)

    for target in ["home_score", "away_score"]:
        reg_results = run_regression(train, val, test, target)
        all_results.append(reg_results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(DATA_DIR / "model_selection_results.csv", index=False)
    print(f"\nFull results saved → data/model_selection_results.csv")


if __name__ == "__main__":
    main()
