"""
model_selection.py
------------------
Compare several regressors on the xGD-shift task and persist the best.

Run:  python -m src.models
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import build_feature_frame

# ------------------------------------------------------------------ #
# 0.  Paths & constants
# ------------------------------------------------------------------ #
DATA_CSV  = pathlib.Path("data/raw/shots_20232024.csv")
MODEL_PKL = pathlib.Path("models/best_model.pkl")
MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["entry_type", "manpower_situation", "offensive_zone"]
NUM_FEATURES = ["shotDistance", "shotAngleAdjusted", "period", "time"]

CV_FOLDS = 5
RANDOM_STATE = 73


# ------------------------------------------------------------------ #
# 1.  Candidate model factory
# ------------------------------------------------------------------ #
def get_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Return a dict of name → regressor instance."""
    return {
        "HGBR": HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=5, random_state=random_state
        ),
        "GBR": GradientBoostingRegressor(
            learning_rate=0.05, max_depth=3, n_estimators=300, random_state=random_state
        ),
        "RF": RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "ET": ExtraTreesRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "ElasticNet": ElasticNet(
            alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=2000
        ),
    }


# ------------------------------------------------------------------ #
# 2.  Preprocessor (shared across models)
# ------------------------------------------------------------------ #
PREPROCESSOR = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ]
)


# ------------------------------------------------------------------ #
# 3.  Model selection routine
# ------------------------------------------------------------------ #
def score_models(
    X: pd.DataFrame, y: pd.Series, models: Dict[str, object]
) -> Dict[str, Tuple[float, float]]:
    """Return dict name → (mean_R2, std_R2) using CV_FOLDS cross-val."""
    results: Dict[str, Tuple[float, float]] = {}
    scorer = make_scorer(r2_score)

    for name, model in models.items():
        pipe = Pipeline([("prep", PREPROCESSOR), ("model", model)])
        cv_scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring=scorer, n_jobs=-1)
        results[name] = (cv_scores.mean(), cv_scores.std())
        print(f"{name:9s}: R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return results


# ------------------------------------------------------------------ #
# 4.  Main
# ------------------------------------------------------------------ #
def main() -> None:
    # Load & build feature frame
    raw = pd.read_csv(DATA_CSV, low_memory=False)
    df = build_feature_frame(raw)

    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df["xGD_shift"]

    # Compare models
    candidates = get_models()
    scores = score_models(X, y, candidates)

    # Pick the best
    best_name = max(scores, key=lambda k: scores[k][0])
    best_model = candidates[best_name]
    print(f"\n🏆  Best model: {best_name} (mean CV R² = {scores[best_name][0]:.3f})")

    # Retrain best model on **full** data  (with train/test split for a report)
    pipe = Pipeline([("prep", PREPROCESSOR), ("model", best_model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    holdout_r2 = pipe.score(X_test, y_test)
    print(f"Hold-out R² (20 % split) : {holdout_r2:.3f}")

    # Persist
    joblib.dump(pipe, MODEL_PKL)
    print("Saved →", MODEL_PKL)


if __name__ == "__main__":
    main()