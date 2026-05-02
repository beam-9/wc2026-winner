from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


OUTCOME_TO_LABEL = {"home_win": 0, "draw": 1, "away_win": 2}
LABEL_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_LABEL.items()}


def _outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return OUTCOME_TO_LABEL["home_win"]
    if home_score < away_score:
        return OUTCOME_TO_LABEL["away_win"]
    return OUTCOME_TO_LABEL["draw"]


def build_match_features(results: pd.DataFrame, rolling_window: int = 10) -> pd.DataFrame:
    """Create leakage-safe pre-match features from historical results."""
    histories: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=rolling_window))
    rows: list[dict[str, object]] = []

    for match in results.sort_values("date").itertuples(index=False):
        home_history = list(histories[match.home_team])
        away_history = list(histories[match.away_team])
        row = {
            "date": match.date,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "neutral": bool(match.neutral),
            "tournament": getattr(match, "tournament", "Unknown"),
            "home_score": match.home_score,
            "away_score": match.away_score,
            "target": _outcome(match.home_score, match.away_score),
        }
        row.update(_team_features("home", home_history))
        row.update(_team_features("away", away_history))
        row["form_points_diff"] = row["home_points_avg"] - row["away_points_avg"]
        row["goal_diff_avg_diff"] = row["home_goal_diff_avg"] - row["away_goal_diff_avg"]
        row["goals_for_avg_diff"] = row["home_goals_for_avg"] - row["away_goals_for_avg"]
        row["goals_against_avg_diff"] = row["home_goals_against_avg"] - row["away_goals_against_avg"]
        row["matches_played_diff"] = row["home_matches_played"] - row["away_matches_played"]
        rows.append(row)

        home_points = 3 if match.home_score > match.away_score else 1 if match.home_score == match.away_score else 0
        away_points = 3 if match.away_score > match.home_score else 1 if match.home_score == match.away_score else 0
        histories[match.home_team].append(
            {
                "points": home_points,
                "goals_for": match.home_score,
                "goals_against": match.away_score,
                "goal_diff": match.home_score - match.away_score,
            }
        )
        histories[match.away_team].append(
            {
                "points": away_points,
                "goals_for": match.away_score,
                "goals_against": match.home_score,
                "goal_diff": match.away_score - match.home_score,
            }
        )

    return pd.DataFrame(rows)


def latest_team_state(features: pd.DataFrame) -> pd.DataFrame:
    """Return latest pre-match form state by team from engineered match features."""
    home = features[
        [
            "date",
            "home_team",
            "home_points_avg",
            "home_goal_diff_avg",
            "home_goals_for_avg",
            "home_goals_against_avg",
            "home_matches_played",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "home_points_avg": "points_avg",
            "home_goal_diff_avg": "goal_diff_avg",
            "home_goals_for_avg": "goals_for_avg",
            "home_goals_against_avg": "goals_against_avg",
            "home_matches_played": "matches_played",
        }
    )
    away = features[
        [
            "date",
            "away_team",
            "away_points_avg",
            "away_goal_diff_avg",
            "away_goals_for_avg",
            "away_goals_against_avg",
            "away_matches_played",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "away_points_avg": "points_avg",
            "away_goal_diff_avg": "goal_diff_avg",
            "away_goals_for_avg": "goals_for_avg",
            "away_goals_against_avg": "goals_against_avg",
            "away_matches_played": "matches_played",
        }
    )
    states = pd.concat([home, away], ignore_index=True)
    return states.sort_values("date").groupby("team", as_index=False).tail(1).reset_index(drop=True)


def make_prediction_frame(home_team: str, away_team: str, states: pd.DataFrame, neutral: bool = True) -> pd.DataFrame:
    home = _lookup_state(states, home_team)
    away = _lookup_state(states, away_team)
    row = {
        "neutral": neutral,
        "home_points_avg": home["points_avg"],
        "home_goal_diff_avg": home["goal_diff_avg"],
        "home_goals_for_avg": home["goals_for_avg"],
        "home_goals_against_avg": home["goals_against_avg"],
        "home_matches_played": home["matches_played"],
        "away_points_avg": away["points_avg"],
        "away_goal_diff_avg": away["goal_diff_avg"],
        "away_goals_for_avg": away["goals_for_avg"],
        "away_goals_against_avg": away["goals_against_avg"],
        "away_matches_played": away["matches_played"],
    }
    row["form_points_diff"] = row["home_points_avg"] - row["away_points_avg"]
    row["goal_diff_avg_diff"] = row["home_goal_diff_avg"] - row["away_goal_diff_avg"]
    row["goals_for_avg_diff"] = row["home_goals_for_avg"] - row["away_goals_for_avg"]
    row["goals_against_avg_diff"] = row["home_goals_against_avg"] - row["away_goals_against_avg"]
    row["matches_played_diff"] = row["home_matches_played"] - row["away_matches_played"]
    return pd.DataFrame([row])


def _team_features(prefix: str, history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            f"{prefix}_points_avg": 1.0,
            f"{prefix}_goal_diff_avg": 0.0,
            f"{prefix}_goals_for_avg": 1.0,
            f"{prefix}_goals_against_avg": 1.0,
            f"{prefix}_matches_played": 0.0,
        }
    return {
        f"{prefix}_points_avg": float(np.mean([x["points"] for x in history])),
        f"{prefix}_goal_diff_avg": float(np.mean([x["goal_diff"] for x in history])),
        f"{prefix}_goals_for_avg": float(np.mean([x["goals_for"] for x in history])),
        f"{prefix}_goals_against_avg": float(np.mean([x["goals_against"] for x in history])),
        f"{prefix}_matches_played": float(len(history)),
    }


def _lookup_state(states: pd.DataFrame, team: str) -> dict[str, float]:
    match = states[states["team"] == team]
    if match.empty:
        return {
            "points_avg": 1.0,
            "goal_diff_avg": 0.0,
            "goals_for_avg": 1.0,
            "goals_against_avg": 1.0,
            "matches_played": 0.0,
        }
    return match.iloc[-1].to_dict()

