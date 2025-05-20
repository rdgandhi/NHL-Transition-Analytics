"""
features.py
-----------
Create model-ready features from the latest MoneyPuck shot file
(column list supplied May 2025).

Key additions vs. previous version
* derive `isHomeTeam` from string column **team**  ("HOME" / "AWAY")
* auto-pick rink coordinates:   arenaAdjustedXCord ▸ xCordAdjusted
* revise entry-type heuristic for new `location` values
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 0.  Helpers
# ------------------------------------------------------------------ #
def _alias_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric columns `x` / `y` exist **once**.

    If the dashboard (or any prior step) already inserted `x`/`y`,
    skip renaming to avoid duplicate column labels.
    """
    df = df.copy()

    if "x" not in df.columns:                               # <-- new guard
        if {"arenaAdjustedXCord", "arenaAdjustedYCord"}.issubset(df.columns):
            df = df.rename(
                columns={
                    "arenaAdjustedXCord": "x",
                    "arenaAdjustedYCord": "y",
                }
            )
        else:
            df = df.rename(
                columns={"xCordAdjusted": "x", "yCordAdjusted": "y"}
            )

    return df

def _add_is_home(df: pd.DataFrame) -> pd.DataFrame:
    """`team` is 'HOME' / 'AWAY'.  Convert to 1 / 0 integer flag."""
    df = df.copy()
    df["isHomeTeam"] = (df["team"] == "HOME").astype(int)
    return df


# ------------------------------------------------------------------ #
# 1.  Zone & entry-type
# ------------------------------------------------------------------ #
def add_zone_and_entry_type(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias_coords(df).pipe(_add_is_home)

    # offensive / defensive zone
    attack_mask = (
        ((df["isHomeTeam"] == 1) & (df["x"] > 0))
        | ((df["isHomeTeam"] == 0) & (df["x"] < 0))
    )
    df["offensive_zone"] = np.where(attack_mask, "ATTACK", "DEFEND")
    df["offensive_zone"] = df["offensive_zone"].astype("category")

    # entry-type heuristic
    df["entry_type"] = "OTHER"
    df.loc[df["shotRush"] == 1, "entry_type"] = "CONTROLLED"
    df.loc[df["location"].str.startswith("Neu"), "entry_type"] = "NEUTRAL"
    df["entry_type"] = df["entry_type"].astype("category")

    return df


# ------------------------------------------------------------------ #
# 2.  Man-power situation
# ------------------------------------------------------------------ #
def add_manpower_situation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    shooters = np.where(
        df["isHomeTeam"] == 1, df["homeSkatersOnIce"], df["awaySkatersOnIce"]
    )
    defenders = np.where(
        df["isHomeTeam"] == 1, df["awaySkatersOnIce"], df["homeSkatersOnIce"]
    )

    diff = shooters - defenders
    mp = pd.Series(diff, index=df.index).map({0: "EV", 1: "PP", -1: "SH"})
    df["manpower_situation"] = mp.fillna("OTHER").astype("category")
    return df


# ------------------------------------------------------------------ #
# 3.  xGD (event & rolling)
# ------------------------------------------------------------------ #
def add_xgd(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    df = df.copy()

    # +xGoal if shooter is home, else −xGoal
    df["xGD_event"] = np.where(df["isHomeTeam"] == 1, df["xGoal"], -df["xGoal"])
    df.sort_values(["game_id", "time"], inplace=True)

    df["xGD_shift"] = (
        df.groupby("game_id")["xGD_event"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


# ------------------------------------------------------------------ #
# 4.  Orchestrator
# ------------------------------------------------------------------ #
def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline."""
    return (
        raw.pipe(add_zone_and_entry_type)
        .pipe(add_manpower_situation)
        .pipe(add_xgd)
    )
