from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_RESULTS_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


def load_results(path: str | Path) -> pd.DataFrame:
    """Load and normalize the historical international results dataset."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_RESULTS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required results columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])

    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"

    return df.sort_values("date").reset_index(drop=True)


def load_groups(path: str | Path) -> pd.DataFrame:
    groups = pd.read_csv(path)
    expected = {"group", "team", "confederation", "is_host"}
    missing = expected - set(groups.columns)
    if missing:
        raise ValueError(f"Missing required group columns: {sorted(missing)}")
    groups["is_host"] = groups["is_host"].astype(bool)
    return groups

