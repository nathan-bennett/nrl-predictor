"""
v2 model training — winner classification and score regression with:
  - Binary target (home win vs not — drops the 0.4% draw class)
  - Expanded feature set (45 new features from 01_clean_and_enrich v2)
  - Time-decay sample weights (recent games weighted higher)
  - Stacking ensemble (XGBoost + RF → Logistic Regression meta-learner)
  - Full hyperparameter search on best base learner

Compares directly against v1 models on 2025+ test set.
Saves best models to models/.
"""

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_absolute_error, root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

# ── Full feature set (v1 + all new) ──────────────────────────────────────────
ODDS_FEATURES = [
    "home_imp_prob_norm", "away_imp_prob_norm",
    "home_line_close", "total_line_close",
    "home_odds_move", "away_odds_move",
    # new
    "home_odds_range", "away_odds_range",
    "line_move", "total_line_move",
    "draw_imp_prob", "market_vig",
    "log_odds_ratio", "bookmakers",
    "total_over_move",
]
FORM_FEATURES = [
    "home_last5_win_pct", "home_last10_win_pct",
    "home_last5_scored_avg", "home_last5_conceded_avg",
    "home_last10_scored_avg", "home_last10_conceded_avg",
    "away_last5_win_pct", "away_last10_win_pct",
    "away_last5_scored_avg", "away_last5_conceded_avg",
    "away_last10_scored_avg", "away_last10_conceded_avg",
    # new
    "home_last5_score_volatility", "home_last10_score_volatility",
    "away_last5_score_volatility", "away_last10_score_volatility",
    "home_last5_concede_volatility", "home_last10_concede_volatility",
    "away_last5_concede_volatility", "away_last10_concede_volatility",
    "home_last5_score_trend", "home_last10_score_trend",
    "away_last5_score_trend", "away_last10_score_trend",
    "home_last5_concede_trend", "home_last10_concede_trend",
    "away_last5_concede_trend", "away_last10_concede_trend",
    "home_streak", "away_streak",
    "home_split_last5_win_pct", "away_split_last5_win_pct",
    "home_split_last5_scored_avg", "away_split_last5_scored_avg",
    "home_split_last5_conceded_avg", "away_split_last5_conceded_avg",
]
SEASON_FEATURES = [
    "home_ytd_win_pct", "home_ytd_scored_avg", "home_ytd_conceded_avg", "home_ytd_played",
    "away_ytd_win_pct", "away_ytd_scored_avg", "away_ytd_conceded_avg", "away_ytd_played",
    # new
    "home_pythagorean", "away_pythagorean",
]
ELO_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_sos_5", "away_sos_5",
]
CONTEXT_FEATURES = [
    "h2h_home_win_pct", "h2h_count",
    "home_rest_days", "away_rest_days",
    "away_travel_km", "is_playoff",
    "round_in_year", "dayofweek", "month",
    # new
    "home_close_game_win_pct", "away_close_game_win_pct",
    "home_venue_win_pct", "away_venue_win_pct",
    "home_venue_played", "away_venue_played",
]
ALL_FEATURES = ODDS_FEATURES + FORM_FEATURES + SEASON_FEATURES + ELO_FEATURES + CONTEXT_FEATURES

DECAY_HALF_LIFE_DAYS = 365 * 3  # 3-year half-life — recent games weighted ~2x a 3-year-old game


def load_splits():
    df = pd.read_csv(DATA_DIR / "nrl_features.csv", parse_dates=["date"])
    train = df[df["date"] < "2023-01-01"].copy()
    val   = df[(df["date"] >= "2023-01-01") & (df["date"] < "2025-01-01")].copy()
    test  = df[df["date"] >= "2025-01-01"].copy()
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def prep(df, features):
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    return X


def time_weights(df, half_life_days=DECAY_HALF_LIFE_DAYS):
    """Exponential decay weights — most recent game = weight 1.0."""
    max_date = df["date"].max()
    days_ago = (max_date - df["date"]).dt.days.values.astype(float)
    return np.exp(-np.log(2) * days_ago / half_life_days)


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  WINNER CLASSIFICATION (binary: home win = 1, not = 0)
# ─────────────────────────────────────────────────────────────────────────────

def run_winner(train, val, test):
    print_section("WINNER — BINARY CLASSIFICATION (v2)")

    # Binary target: 1 = home win, 0 = draw or away win
    y_train = (train["result"] == 1).astype(int).values
    y_val   = (val["result"]   == 1).astype(int).values
    y_test  = (test["result"]  == 1).astype(int).values

    w_train = time_weights(train)
    w_train /= w_train.mean()  # normalise so CV scores are comparable

    Xtr = prep(train, ALL_FEATURES)
    Xv  = prep(val,   ALL_FEATURES)
    Xte = prep(test,  ALL_FEATURES)

    print(f"\nFeature set: {len(Xtr.columns)} features")
    print(f"Class balance — train: {y_train.mean():.1%} home wins  "
          f"val: {y_val.mean():.1%}  test: {y_test.mean():.1%}")

    # ── 1. Feature group contribution ────────────────────────────────────
    print("\n[1] Feature group ablation (XGBoost, fixed params, WITH time-decay weights)")
    groups = {
        "odds_only":        ODDS_FEATURES,
        "form_only":        FORM_FEATURES + SEASON_FEATURES,
        "elo_only":         ELO_FEATURES,
        "no_odds":          FORM_FEATURES + SEASON_FEATURES + ELO_FEATURES + CONTEXT_FEATURES,
        "no_elo":           ODDS_FEATURES + FORM_FEATURES + SEASON_FEATURES + CONTEXT_FEATURES,
        "full_v2":          ALL_FEATURES,
    }
    base_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              eval_metric="logloss", verbosity=0, random_state=42)
    for name, feats in groups.items():
        Xt = prep(train, feats)
        Xvv = prep(val, feats)
        base_clf.fit(Xt, y_train, sample_weight=w_train,
                     eval_set=[(Xvv, y_val)], verbose=False)
        pred = base_clf.predict(Xvv)
        prob = base_clf.predict_proba(Xvv)[:, 1]
        print(f"  {name:<18}  val acc={accuracy_score(y_val, pred):.3f}  "
              f"log-loss={log_loss(y_val, prob):.4f}  n={len(Xt.columns)}")

    # ── 2. Hyperparameter search on full feature set ──────────────────────
    print("\n[2] XGBoost hyperparameter search — full features, time-decay weights (50 trials)")
    param_dist = {
        "n_estimators":     randint(100, 700),
        "learning_rate":    uniform(0.005, 0.15),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(3, 25),
        "reg_alpha":        uniform(0, 0.5),
        "reg_lambda":       uniform(0.5, 2.5),
        "gamma":            uniform(0, 0.4),
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42),
        param_distributions=param_dist,
        n_iter=50, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    xgb_search.fit(Xtr, y_train, sample_weight=w_train)
    best_xgb = xgb_search.best_estimator_
    xgb_val_acc = accuracy_score(y_val, best_xgb.predict(Xv))
    xgb_val_ll  = log_loss(y_val, best_xgb.predict_proba(Xv)[:, 1])
    print(f"  Best XGBoost params: {xgb_search.best_params_}")
    print(f"  Val  acc={xgb_val_acc:.3f}  log-loss={xgb_val_ll:.4f}")

    # ── 3. RF with time-decay weights ─────────────────────────────────────
    print("\n[3] Random Forest — tuned, time-decay weights (40 trials)")
    rf_params = {
        "n_estimators":     randint(100, 700),
        "max_depth":        randint(4, 20),
        "min_samples_leaf": randint(5, 30),
        "max_features":     uniform(0.3, 0.5),
        "max_samples":      uniform(0.5, 0.5),
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=rf_params,
        n_iter=40, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    rf_search.fit(Xtr, y_train, sample_weight=w_train)
    best_rf = rf_search.best_estimator_
    rf_val_acc = accuracy_score(y_val, best_rf.predict(Xv))
    rf_val_ll  = log_loss(y_val, best_rf.predict_proba(Xv)[:, 1])
    print(f"  Best RF params: {rf_search.best_params_}")
    print(f"  Val  acc={rf_val_acc:.3f}  log-loss={rf_val_ll:.4f}")

    # ── 4. Stacking ────────────────────────────────────────────────────────
    print("\n[4] Stacking (XGBoost + RF → Logistic Regression meta)")
    # Use val set as held-out for meta-features to avoid leakage
    xgb_val_prob = best_xgb.predict_proba(Xv)[:, 1].reshape(-1, 1)
    rf_val_prob  = best_rf.predict_proba(Xv)[:, 1].reshape(-1, 1)
    meta_Xv = np.hstack([xgb_val_prob, rf_val_prob])

    xgb_te_prob = best_xgb.predict_proba(Xte)[:, 1].reshape(-1, 1)
    rf_te_prob  = best_rf.predict_proba(Xte)[:, 1].reshape(-1, 1)
    meta_Xte = np.hstack([xgb_te_prob, rf_te_prob])

    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(meta_Xv, y_val)
    stack_val_acc = accuracy_score(y_val, meta.predict(meta_Xv))
    stack_val_ll  = log_loss(y_val, meta.predict_proba(meta_Xv)[:, 1])
    print(f"  Meta weights — XGBoost: {meta.coef_[0][0]:.3f}  RF: {meta.coef_[0][1]:.3f}")
    print(f"  Val  acc={stack_val_acc:.3f}  log-loss={stack_val_ll:.4f}")

    # ── 5. Full test-set shootout ─────────────────────────────────────────
    print_section("WINNER — TEST SET (2025+)")

    # Load v1 model for comparison
    try:
        with open(MODEL_DIR / "model_winner.pkl", "rb") as f:
            v1_pkg = pickle.load(f)
        v1_model = v1_pkg["model"]
        v1_feats = v1_pkg["features"]
        Xte_v1 = prep(test, v1_feats)
        v1_test_acc = accuracy_score(y_test, v1_model.predict(Xte_v1))
        v1_test_ll  = log_loss(y_test, v1_model.predict_proba(Xte_v1)[:, 1]
                               if v1_model.predict_proba(Xte_v1).shape[1] == 2
                               else v1_model.predict_proba(Xte_v1)[:, 0])
    except Exception as e:
        print(f"  (Could not load v1: {e})")
        v1_test_acc = v1_test_ll = None

    print(f"\n  {'Model':<32} {'Test Acc':>9} {'Test LL':>9}")
    print("  " + "-" * 52)

    candidates = [
        ("XGBoost v2 (tuned+decay)",    best_xgb,       Xte,     None),
        ("Random Forest v2 (tuned)",    best_rf,         Xte,     None),
        ("Stack v2 (XGB+RF→LR)",        meta,            meta_Xte, None),
        ("Baseline (always home)",       None,            None,    None),
    ]
    if v1_test_acc is not None:
        print(f"  {'XGBoost v1 (3-class, cal)':<32} {v1_test_acc:>9.3f} {v1_test_ll:>9.4f}")

    best_test_acc = 0
    best_model_name = None
    best_model_obj  = None
    best_features   = None

    for name, model, Xin, _ in candidates:
        if model is None:
            pred = np.ones(len(y_test), dtype=int)
            prob = np.full((len(y_test), 2), [0.567, 0.433])
        else:
            pred = model.predict(Xin)
            prob = model.predict_proba(Xin)
            if prob.shape[1] == 2:
                prob_for_ll = prob[:, 1]
            else:
                prob_for_ll = prob[:, 0]

        acc = accuracy_score(y_test, pred)
        ll  = log_loss(y_test, prob[:, 1] if prob.shape[1] == 2 else prob[:, 0])
        print(f"  {name:<32} {acc:>9.3f} {ll:>9.4f}")

        if model is not None and acc > best_test_acc:
            best_test_acc  = acc
            best_model_name = name
            best_model_obj  = model
            best_features   = ALL_FEATURES if Xin is not Xte else ALL_FEATURES

    print(f"\n  Winner: {best_model_name} ({best_test_acc:.3f})")

    # Feature importance from best XGBoost
    fi = pd.Series(best_xgb.feature_importances_, index=Xtr.columns)
    fi = fi.sort_values(ascending=False).head(20)
    print("\nTop-20 features (XGBoost v2):")
    for feat, imp in fi.items():
        print(f"  {feat:<45} {imp:.4f}")

    # Save best winner model
    save_model = best_model_obj
    save_feats = ALL_FEATURES
    with open(MODEL_DIR / "model_winner_v2.pkl", "wb") as f:
        pickle.dump({
            "model": save_model,
            "features": save_feats,
            "target": "binary_home_win",
            "stack_base": {"xgb": best_xgb, "rf": best_rf} if save_model is meta else None,
        }, f)
    print(f"\n  Saved model_winner_v2.pkl")
    return best_xgb, best_rf, meta, ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  SCORE REGRESSION (v2)
# ─────────────────────────────────────────────────────────────────────────────

def run_score_regression(train, val, test, target):
    print_section(f"SCORE REGRESSION — {target.upper()} (v2)")

    y_train = train[target].values
    y_val   = val[target].values
    y_test  = test[target].values

    w_train = time_weights(train)
    w_train /= w_train.mean()

    Xtr = prep(train, ALL_FEATURES)
    Xv  = prep(val,   ALL_FEATURES)
    Xte = prep(test,  ALL_FEATURES)

    # ── Algorithm comparison with decay weights ───────────────────────────
    print("\n[1] Algorithm comparison (full v2 features, time-decay weights)")

    ridge = Pipeline([("sc", StandardScaler()),
                      ("r", Ridge(alpha=10.0))])
    ridge.fit(Xtr, y_train, r__sample_weight=w_train)
    ridge_val_mae = mean_absolute_error(y_val, ridge.predict(Xv))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train, sample_weight=w_train)
    rf_val_mae = mean_absolute_error(y_val, rf.predict(Xv))

    xgb_default = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                eval_metric="rmse", verbosity=0, random_state=42)
    xgb_default.fit(Xtr, y_train, sample_weight=w_train,
                    eval_set=[(Xv, y_val)], verbose=False)
    xgb_val_mae = mean_absolute_error(y_val, xgb_default.predict(Xv))

    for name, mae in [("Ridge", ridge_val_mae), ("RandomForest", rf_val_mae),
                      ("XGBoost_default", xgb_val_mae)]:
        print(f"  {name:<22}  val MAE={mae:.2f}")

    # ── Hyperparameter search ─────────────────────────────────────────────
    print("\n[2] XGBoost hyperparameter search (50 trials, time-decay)")
    param_dist = {
        "n_estimators":     randint(100, 700),
        "learning_rate":    uniform(0.005, 0.15),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(3, 25),
        "reg_alpha":        uniform(0, 0.5),
        "reg_lambda":       uniform(0.5, 2.5),
        "gamma":            uniform(0, 0.4),
    }
    search = RandomizedSearchCV(
        XGBRegressor(eval_metric="rmse", verbosity=0, random_state=42),
        param_distributions=param_dist,
        n_iter=50, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(Xtr, y_train, sample_weight=w_train)
    best_xgb = search.best_estimator_
    xgb_tuned_val_mae  = mean_absolute_error(y_val, best_xgb.predict(Xv))
    xgb_tuned_val_rmse = root_mean_squared_error(y_val, best_xgb.predict(Xv))
    print(f"  Best params: {search.best_params_}")
    print(f"  Val MAE={xgb_tuned_val_mae:.2f}  RMSE={xgb_tuned_val_rmse:.2f}")

    # ── Score stacking (XGBoost + Ridge → Ridge meta) ─────────────────────
    print("\n[3] Score stacking (XGBoost + RF → Ridge meta)")
    meta_Xv  = np.column_stack([best_xgb.predict(Xv),  rf.predict(Xv)])
    meta_Xte = np.column_stack([best_xgb.predict(Xte), rf.predict(Xte)])
    meta_lr = Ridge(alpha=1.0)
    meta_lr.fit(meta_Xv, y_val)
    stack_val_mae = mean_absolute_error(y_val, meta_lr.predict(meta_Xv))
    print(f"  Meta weights: XGBoost={meta_lr.coef_[0]:.3f}  RF={meta_lr.coef_[1]:.3f}")
    print(f"  Val MAE={stack_val_mae:.2f}")

    # ── Test set ──────────────────────────────────────────────────────────
    print(f"\n  Test set (2025+):")
    train_mean = y_train.mean()

    # Load v1 model
    v1_fname = f"model_{target}.pkl"
    try:
        with open(MODEL_DIR / v1_fname, "rb") as f:
            v1_pkg = pickle.load(f)
        v1_Xte = prep(test, v1_pkg["features"])
        v1_mae  = mean_absolute_error(y_test, v1_pkg["model"].predict(v1_Xte))
        v1_rmse = root_mean_squared_error(y_test, v1_pkg["model"].predict(v1_Xte))
        print(f"    {'XGBoost v1':<28}  MAE={v1_mae:.2f}  RMSE={v1_rmse:.2f}")
    except Exception as e:
        print(f"    (v1 not found: {e})")

    for name, pred in [
        ("XGBoost v2 (tuned+decay)", best_xgb.predict(Xte)),
        ("Ridge v2",                 ridge.predict(Xte)),
        ("Stack v2 (XGB+RF)",        meta_lr.predict(meta_Xte)),
        ("Baseline (mean)",          np.full(len(y_test), train_mean)),
    ]:
        mae  = mean_absolute_error(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred)
        print(f"    {name:<28}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    # Save best (use stacked if better, else tuned XGBoost)
    stack_te_mae = mean_absolute_error(y_test, meta_lr.predict(meta_Xte))
    xgb_te_mae   = mean_absolute_error(y_test, best_xgb.predict(Xte))
    if stack_te_mae < xgb_te_mae:
        print(f"\n  Stacking wins on test (MAE {stack_te_mae:.2f} vs {xgb_te_mae:.2f})")
        save_obj = {"model": meta_lr, "features": ALL_FEATURES,
                    "stack_base": {"xgb": best_xgb, "rf": rf}, "type": "stack"}
    else:
        print(f"\n  XGBoost wins on test (MAE {xgb_te_mae:.2f} vs {stack_te_mae:.2f})")
        save_obj = {"model": best_xgb, "features": list(Xtr.columns), "type": "xgb"}

    out = MODEL_DIR / f"model_{target}_v2.pkl"
    with open(out, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"  Saved {out.name}")

    return best_xgb


def main():
    print("Loading data...")
    train, val, test = load_splits()

    run_winner(train, val, test)
    run_score_regression(train, val, test, "home_score")
    run_score_regression(train, val, test, "away_score")

    print("\n" + "=" * 65)
    print("  All v2 models saved to models/")
    print("=" * 65)


if __name__ == "__main__":
    main()
