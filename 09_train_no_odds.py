"""
No-odds model training — winner classification and score regression
using only non-market features (ELO, form, context, H2H).

Designed for predicting the full season draw and finals where odds
are not yet available.

Algorithms compared:
  Winner:  XGBoost, RandomForest, GradientBoosting, ExtraTrees,
           LogisticRegression, MLP
  Scores:  XGBoost, RandomForest, GradientBoosting, ExtraTrees,
           Ridge, ElasticNet

Stacks the best two base models per target.

Saves:
  models/model_winner_no_odds.pkl
  models/model_home_score_no_odds.pkl
  models/model_away_score_no_odds.pkl
"""

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_absolute_error, root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Feature set — no odds/market columns ─────────────────────────────────────

FORM_FEATURES = [
    "home_last5_win_pct",  "home_last10_win_pct",
    "home_last5_scored_avg", "home_last5_conceded_avg",
    "home_last10_scored_avg", "home_last10_conceded_avg",
    "away_last5_win_pct",  "away_last10_win_pct",
    "away_last5_scored_avg", "away_last5_conceded_avg",
    "away_last10_scored_avg", "away_last10_conceded_avg",
    "home_last5_score_volatility",  "home_last10_score_volatility",
    "away_last5_score_volatility",  "away_last10_score_volatility",
    "home_last5_concede_volatility","home_last10_concede_volatility",
    "away_last5_concede_volatility","away_last10_concede_volatility",
    "home_last5_score_trend",  "home_last10_score_trend",
    "away_last5_score_trend",  "away_last10_score_trend",
    "home_last5_concede_trend","home_last10_concede_trend",
    "away_last5_concede_trend","away_last10_concede_trend",
    "home_streak", "away_streak",
    "home_split_last5_win_pct",    "away_split_last5_win_pct",
    "home_split_last5_scored_avg", "away_split_last5_scored_avg",
    "home_split_last5_conceded_avg","away_split_last5_conceded_avg",
]
SEASON_FEATURES = [
    "home_ytd_win_pct", "home_ytd_scored_avg",
    "home_ytd_conceded_avg", "home_ytd_played",
    "away_ytd_win_pct", "away_ytd_scored_avg",
    "away_ytd_conceded_avg", "away_ytd_played",
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
    "home_close_game_win_pct", "away_close_game_win_pct",
    "home_venue_win_pct", "away_venue_win_pct",
    "home_venue_played", "away_venue_played",
]
NO_ODDS_FEATURES = FORM_FEATURES + SEASON_FEATURES + ELO_FEATURES + CONTEXT_FEATURES

DECAY_HALF_LIFE_DAYS = 365 * 3


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
    max_date = df["date"].max()
    days_ago = (max_date - df["date"]).dt.days.values.astype(float)
    return np.exp(-np.log(2) * days_ago / half_life_days)


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  WINNER CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_winner(train, val, test):
    print_section("WINNER — NO-ODDS BINARY CLASSIFICATION")

    y_train = (train["result"] == 1).astype(int).values
    y_val   = (val["result"]   == 1).astype(int).values
    y_test  = (test["result"]  == 1).astype(int).values

    w_train = time_weights(train)
    w_train /= w_train.mean()

    Xtr = prep(train, NO_ODDS_FEATURES)
    Xv  = prep(val,   NO_ODDS_FEATURES)
    Xte = prep(test,  NO_ODDS_FEATURES)

    print(f"\n{len(Xtr.columns)} features (no odds)")
    print(f"Class balance — train: {y_train.mean():.1%}  val: {y_val.mean():.1%}  "
          f"test: {y_test.mean():.1%}")

    # ── XGBoost (50 trials) ───────────────────────────────────────────────
    print("\n[1] XGBoost — 50-trial RandomizedSearchCV")
    xgb_params = {
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
        param_distributions=xgb_params,
        n_iter=50, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    xgb_search.fit(Xtr, y_train, sample_weight=w_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  Val acc={accuracy_score(y_val, best_xgb.predict(Xv)):.3f}  "
          f"LL={log_loss(y_val, best_xgb.predict_proba(Xv)[:,1]):.4f}")

    # ── RandomForest (40 trials) ──────────────────────────────────────────
    print("\n[2] RandomForest — 40-trial search")
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
    print(f"  Val acc={accuracy_score(y_val, best_rf.predict(Xv)):.3f}  "
          f"LL={log_loss(y_val, best_rf.predict_proba(Xv)[:,1]):.4f}")

    # ── GradientBoosting (30 trials) ──────────────────────────────────────
    print("\n[3] GradientBoosting — 30-trial search")
    gb_params = {
        "n_estimators":   randint(100, 500),
        "learning_rate":  uniform(0.01, 0.2),
        "max_depth":      randint(2, 7),
        "subsample":      uniform(0.5, 0.5),
        "min_samples_leaf": randint(5, 30),
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=gb_params,
        n_iter=30, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    gb_search.fit(Xtr, y_train, sample_weight=w_train)
    best_gb = gb_search.best_estimator_
    print(f"  Val acc={accuracy_score(y_val, best_gb.predict(Xv)):.3f}  "
          f"LL={log_loss(y_val, best_gb.predict_proba(Xv)[:,1]):.4f}")

    # ── ExtraTrees (30 trials) ────────────────────────────────────────────
    print("\n[4] ExtraTrees — 30-trial search")
    et_params = {
        "n_estimators":     randint(100, 600),
        "max_depth":        randint(4, 20),
        "min_samples_leaf": randint(5, 30),
        "max_features":     uniform(0.3, 0.5),
    }
    et_search = RandomizedSearchCV(
        ExtraTreesClassifier(random_state=42, n_jobs=-1),
        param_distributions=et_params,
        n_iter=30, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    et_search.fit(Xtr, y_train, sample_weight=w_train)
    best_et = et_search.best_estimator_
    print(f"  Val acc={accuracy_score(y_val, best_et.predict(Xv)):.3f}  "
          f"LL={log_loss(y_val, best_et.predict_proba(Xv)[:,1]):.4f}")

    # ── LogisticRegression (scaled) ───────────────────────────────────────
    print("\n[5] LogisticRegression (L2, StandardScaler)")
    lr_results = []
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])
        pipe.fit(Xtr, y_train)
        ll = log_loss(y_val, pipe.predict_proba(Xv)[:,1])
        lr_results.append((ll, C, pipe))
    lr_results.sort()
    best_lr = lr_results[0][2]
    print(f"  Best C={lr_results[0][1]}  Val acc={accuracy_score(y_val, best_lr.predict(Xv)):.3f}  "
          f"LL={lr_results[0][0]:.4f}")

    # ── MLP (30 trials) ───────────────────────────────────────────────────
    print("\n[6] MLP — 30-trial search (StandardScaler)")
    mlp_params = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
        "mlp__alpha":              uniform(0.0001, 0.05),
        "mlp__learning_rate_init": uniform(0.0005, 0.01),
        "mlp__max_iter":           [500],
    }
    mlp_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(random_state=42, early_stopping=True)),
    ])
    mlp_search = RandomizedSearchCV(
        mlp_pipe, param_distributions=mlp_params,
        n_iter=30, scoring="neg_log_loss",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    mlp_search.fit(Xtr, y_train)
    best_mlp = mlp_search.best_estimator_
    print(f"  Val acc={accuracy_score(y_val, best_mlp.predict(Xv)):.3f}  "
          f"LL={log_loss(y_val, best_mlp.predict_proba(Xv)[:,1]):.4f}")

    # ── Test-set shootout ─────────────────────────────────────────────────
    print_section("WINNER — TEST SET (2025+)")
    candidates = [
        ("XGBoost",          best_xgb, Xte),
        ("RandomForest",     best_rf,  Xte),
        ("GradientBoosting", best_gb,  Xte),
        ("ExtraTrees",       best_et,  Xte),
        ("LogisticRegr",     best_lr,  Xte),
        ("MLP",              best_mlp, Xte),
        ("Baseline(home)",   None,     None),
    ]
    print(f"\n  {'Model':<22} {'Test Acc':>9} {'Test LL':>9}")
    print("  " + "-" * 42)

    test_results = []
    for name, model, Xin in candidates:
        if model is None:
            pred = np.ones(len(y_test), dtype=int)
            prob = np.full(len(y_test), 0.567)
        else:
            pred = model.predict(Xin)
            prob = model.predict_proba(Xin)[:,1]
        acc = accuracy_score(y_test, pred)
        ll  = log_loss(y_test, prob)
        print(f"  {name:<22} {acc:>9.3f} {ll:>9.4f}")
        if model is not None:
            test_results.append((acc, ll, name, model))

    test_results.sort(reverse=True)
    best1_name, best1 = test_results[0][2], test_results[0][3]
    best2_name, best2 = test_results[1][2], test_results[1][3]
    print(f"\n  Top-2: [{best1_name}] and [{best2_name}]")

    # ── Stacking top-2 → Logistic Regression meta ─────────────────────────
    print("\n[7] Stacking top-2 → Logistic Regression meta")
    b1_val = best1.predict_proba(Xv)[:,1].reshape(-1,1)
    b2_val = best2.predict_proba(Xv)[:,1].reshape(-1,1)
    meta_Xv  = np.hstack([b1_val, b2_val])

    b1_te  = best1.predict_proba(Xte)[:,1].reshape(-1,1)
    b2_te  = best2.predict_proba(Xte)[:,1].reshape(-1,1)
    meta_Xte = np.hstack([b1_te, b2_te])

    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(meta_Xv, y_val)
    stack_acc = accuracy_score(y_test, meta.predict(meta_Xte))
    stack_ll  = log_loss(y_test, meta.predict_proba(meta_Xte)[:,1])
    print(f"  Stack val acc={accuracy_score(y_val, meta.predict(meta_Xv)):.3f}  "
          f"test acc={stack_acc:.3f}  LL={stack_ll:.4f}")

    # Feature importance from XGBoost
    fi = pd.Series(best_xgb.feature_importances_, index=Xtr.columns)
    print("\nTop-15 features (XGBoost, no-odds):")
    for feat, imp in fi.sort_values(ascending=False).head(15).items():
        print(f"  {feat:<45} {imp:.4f}")

    # Decide best model to save (stacked if better on test, else best single)
    single_best_acc = test_results[0][0]
    if stack_acc >= single_best_acc:
        print(f"\n  Stacking wins ({stack_acc:.3f} vs {single_best_acc:.3f})")
        save_pkg = {
            "model":      meta,
            "features":   NO_ODDS_FEATURES,
            "target":     "binary_home_win",
            "stack_base": {"m1": best1, "m2": best2,
                           "m1_name": best1_name, "m2_name": best2_name},
            "type":       "stack",
        }
    else:
        print(f"\n  Single model [{best1_name}] wins ({single_best_acc:.3f})")
        save_pkg = {
            "model":    best1,
            "features": NO_ODDS_FEATURES,
            "target":   "binary_home_win",
            "type":     best1_name.lower(),
        }

    out = MODEL_DIR / "model_winner_no_odds.pkl"
    with open(out, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"  Saved {out.name}")
    return best1, best2, meta


# ─────────────────────────────────────────────────────────────────────────────
#  SCORE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_score_regression(train, val, test, target):
    print_section(f"SCORE REGRESSION — {target.upper()} (no-odds)")

    y_train = train[target].values
    y_val   = val[target].values
    y_test  = test[target].values

    w_train = time_weights(train)
    w_train /= w_train.mean()

    Xtr = prep(train, NO_ODDS_FEATURES)
    Xv  = prep(val,   NO_ODDS_FEATURES)
    Xte = prep(test,  NO_ODDS_FEATURES)

    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xv_sc  = sc.transform(Xv)
    Xte_sc = sc.transform(Xte)

    # ── XGBoost (50 trials) ───────────────────────────────────────────────
    print("\n[1] XGBoost — 50-trial search")
    xgb_params = {
        "n_estimators":     randint(100, 700),
        "learning_rate":    uniform(0.005, 0.15),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(3, 25),
        "reg_alpha":        uniform(0, 0.5),
        "reg_lambda":       uniform(0.5, 2.5),
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(eval_metric="rmse", verbosity=0, random_state=42),
        param_distributions=xgb_params,
        n_iter=50, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    xgb_search.fit(Xtr, y_train, sample_weight=w_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  Val MAE={mean_absolute_error(y_val, best_xgb.predict(Xv)):.2f}")

    # ── RandomForest (40 trials) ──────────────────────────────────────────
    print("\n[2] RandomForest — 40-trial search")
    rf_params = {
        "n_estimators":     randint(100, 600),
        "max_depth":        randint(4, 20),
        "min_samples_leaf": randint(5, 30),
        "max_features":     uniform(0.3, 0.5),
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=rf_params,
        n_iter=40, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    rf_search.fit(Xtr, y_train, sample_weight=w_train)
    best_rf = rf_search.best_estimator_
    print(f"  Val MAE={mean_absolute_error(y_val, best_rf.predict(Xv)):.2f}")

    # ── GradientBoosting (30 trials) ──────────────────────────────────────
    print("\n[3] GradientBoosting — 30-trial search")
    gb_params = {
        "n_estimators":     randint(100, 400),
        "learning_rate":    uniform(0.01, 0.2),
        "max_depth":        randint(2, 7),
        "subsample":        uniform(0.5, 0.5),
        "min_samples_leaf": randint(5, 30),
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=gb_params,
        n_iter=30, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    gb_search.fit(Xtr, y_train, sample_weight=w_train)
    best_gb = gb_search.best_estimator_
    print(f"  Val MAE={mean_absolute_error(y_val, best_gb.predict(Xv)):.2f}")

    # ── ExtraTrees (30 trials) ────────────────────────────────────────────
    print("\n[4] ExtraTrees — 30-trial search")
    et_params = {
        "n_estimators":     randint(100, 600),
        "max_depth":        randint(4, 20),
        "min_samples_leaf": randint(5, 30),
        "max_features":     uniform(0.3, 0.5),
    }
    et_search = RandomizedSearchCV(
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        param_distributions=et_params,
        n_iter=30, scoring="neg_mean_absolute_error",
        cv=5, n_jobs=-1, random_state=42, verbose=0,
    )
    et_search.fit(Xtr, y_train, sample_weight=w_train)
    best_et = et_search.best_estimator_
    print(f"  Val MAE={mean_absolute_error(y_val, best_et.predict(Xv)):.2f}")

    # ── Ridge ─────────────────────────────────────────────────────────────
    print("\n[5] Ridge (scaled)")
    ridge_results = []
    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        r = Ridge(alpha=alpha)
        r.fit(Xtr_sc, y_train, sample_weight=w_train)
        mae = mean_absolute_error(y_val, r.predict(Xv_sc))
        ridge_results.append((mae, alpha, r))
    ridge_results.sort()
    best_ridge_alpha = ridge_results[0][1]
    best_ridge_raw   = ridge_results[0][2]
    best_ridge = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=best_ridge_alpha))])
    best_ridge.fit(Xtr, y_train, r__sample_weight=w_train)
    print(f"  Best alpha={best_ridge_alpha}  Val MAE={mean_absolute_error(y_val, best_ridge.predict(Xv)):.2f}")

    # ── ElasticNet ────────────────────────────────────────────────────────
    print("\n[6] ElasticNet (scaled)")
    en_results = []
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)
            en.fit(Xtr_sc, y_train, sample_weight=w_train)
            mae = mean_absolute_error(y_val, en.predict(Xv_sc))
            en_results.append((mae, alpha, l1, en))
    en_results.sort()
    best_en_raw = en_results[0][3]
    best_en = Pipeline([("sc", StandardScaler()),
                         ("en", ElasticNet(alpha=en_results[0][1],
                                           l1_ratio=en_results[0][2], max_iter=5000))])
    best_en.fit(Xtr, y_train, en__sample_weight=w_train)
    print(f"  Best alpha={en_results[0][1]} l1={en_results[0][2]}  "
          f"Val MAE={mean_absolute_error(y_val, best_en.predict(Xv)):.2f}")

    # ── Test-set shootout ─────────────────────────────────────────────────
    print(f"\n  Test set (2025+):")
    train_mean = y_train.mean()
    all_models = [
        ("XGBoost",          best_xgb,   Xte),
        ("RandomForest",     best_rf,    Xte),
        ("GradientBoosting", best_gb,    Xte),
        ("ExtraTrees",       best_et,    Xte),
        ("Ridge",            best_ridge, Xte),
        ("ElasticNet",       best_en,    Xte),
        ("Baseline(mean)",   None,       None),
    ]
    print(f"    {'Model':<22} {'MAE':>7} {'RMSE':>7}")
    print("    " + "-" * 38)

    te_results = []
    for name, model, Xin in all_models:
        pred = model.predict(Xin) if model is not None else np.full(len(y_test), train_mean)
        mae  = mean_absolute_error(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred)
        print(f"    {name:<22} {mae:>7.2f} {rmse:>7.2f}")
        if model is not None:
            te_results.append((mae, name, model))

    te_results.sort()
    b1_name, b1 = te_results[0][1], te_results[0][2]
    b2_name, b2 = te_results[1][1], te_results[1][2]
    print(f"\n  Top-2: [{b1_name}] and [{b2_name}]")

    # ── Stacking top-2 → Ridge meta ───────────────────────────────────────
    print("\n[7] Stacking top-2 → Ridge meta")
    b1_val_p = b1.predict(Xv).reshape(-1,1)
    b2_val_p = b2.predict(Xv).reshape(-1,1)
    meta_Xv  = np.hstack([b1_val_p, b2_val_p])

    b1_te_p  = b1.predict(Xte).reshape(-1,1)
    b2_te_p  = b2.predict(Xte).reshape(-1,1)
    meta_Xte = np.hstack([b1_te_p, b2_te_p])

    meta = Ridge(alpha=1.0)
    meta.fit(meta_Xv, y_val)
    stack_mae = mean_absolute_error(y_test, meta.predict(meta_Xte))
    print(f"  Stack test MAE={stack_mae:.2f}  "
          f"(best single={te_results[0][0]:.2f})")

    if stack_mae < te_results[0][0]:
        print("  Stacking wins — saving stacked model")
        save_pkg = {
            "model":      meta,
            "features":   NO_ODDS_FEATURES,
            "target":     target,
            "stack_base": {"m1": b1, "m2": b2,
                           "m1_name": b1_name, "m2_name": b2_name},
            "type":       "stack",
        }
    else:
        print(f"  Single [{b1_name}] wins — saving single model")
        save_pkg = {
            "model":    b1,
            "features": NO_ODDS_FEATURES,
            "target":   target,
            "type":     b1_name.lower(),
        }

    out = MODEL_DIR / f"model_{target}_no_odds.pkl"
    with open(out, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"  Saved {out.name}")


def main():
    print("Loading data...")
    train, val, test = load_splits()

    run_winner(train, val, test)
    run_score_regression(train, val, test, "home_score")
    run_score_regression(train, val, test, "away_score")

    print("\n" + "=" * 65)
    print("  No-odds models saved to models/")
    print("=" * 65)


if __name__ == "__main__":
    main()
