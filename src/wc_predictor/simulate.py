from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from wc_predictor.features import make_prediction_frame


ROUND_NAMES = ["round_32", "round_16", "quarterfinal", "semifinal", "final", "winner"]


def simulate_tournament(
    groups: pd.DataFrame,
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    simulations: int = 10000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    round_counts: dict[str, Counter[str]] = {round_name: Counter() for round_name in ROUND_NAMES}

    for _ in range(simulations):
        qualified = _simulate_group_stage(groups, states, model, feature_columns, rng, round_counts)
        champion = _simulate_knockouts(qualified, states, model, feature_columns, rng, round_counts)
        round_counts["winner"][champion] += 1

    round_rows = []
    all_teams = sorted(groups["team"].unique())
    for team in all_teams:
        row = {"team": team}
        for round_name in ROUND_NAMES:
            row[round_name] = round_counts[round_name][team] / simulations
        round_rows.append(row)

    round_probabilities = pd.DataFrame(round_rows).sort_values("winner", ascending=False)
    winner_probabilities = round_probabilities[["team", "winner"]].rename(columns={"winner": "winner_probability"})
    return round_probabilities, winner_probabilities


def _simulate_group_stage(
    groups: pd.DataFrame,
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    rng: np.random.Generator,
    round_counts: dict[str, Counter[str]],
) -> list[str]:
    qualified: list[str] = []
    third_place: list[dict[str, object]] = []

    for group_name, group_df in groups.groupby("group", sort=True):
        teams = group_df["team"].tolist()
        table = {team: {"team": team, "group": group_name, "points": 0, "gd": 0, "gf": 0} for team in teams}
        for i, home in enumerate(teams):
            for away in teams[i + 1 :]:
                home_goals, away_goals = _simulate_score(home, away, states, model, feature_columns, rng)
                _update_table(table[home], table[away], home_goals, away_goals)

        ranked = sorted(
            table.values(),
            key=lambda x: (x["points"], x["gd"], x["gf"], rng.random()),
            reverse=True,
        )
        qualified.extend([ranked[0]["team"], ranked[1]["team"]])
        third_place.append(ranked[2])

    best_thirds = sorted(
        third_place,
        key=lambda x: (x["points"], x["gd"], x["gf"], rng.random()),
        reverse=True,
    )[:8]
    qualified.extend([row["team"] for row in best_thirds])

    for team in qualified:
        round_counts["round_32"][team] += 1

    return qualified


def _simulate_knockouts(
    teams: list[str],
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    rng: np.random.Generator,
    round_counts: dict[str, Counter[str]],
) -> str:
    current = list(teams)
    rng.shuffle(current)
    rounds = ["round_16", "quarterfinal", "semifinal", "final", "winner"]

    for round_name in rounds:
        winners: list[str] = []
        for i in range(0, len(current), 2):
            winner = _simulate_knockout_match(current[i], current[i + 1], states, model, feature_columns, rng)
            winners.append(winner)
        if round_name != "winner":
            for team in winners:
                round_counts[round_name][team] += 1
        current = winners

    return current[0]


def _simulate_score(
    home: str,
    away: str,
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    rng: np.random.Generator,
) -> tuple[int, int]:
    probabilities = _predict_probabilities(home, away, states, model, feature_columns)
    outcome = rng.choice([0, 1, 2], p=probabilities)
    if outcome == 0:
        return int(rng.integers(1, 4)), int(rng.integers(0, 2))
    if outcome == 2:
        return int(rng.integers(0, 2)), int(rng.integers(1, 4))
    goals = int(rng.integers(0, 3))
    return goals, goals


def _simulate_knockout_match(
    home: str,
    away: str,
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    rng: np.random.Generator,
) -> str:
    probabilities = _predict_probabilities(home, away, states, model, feature_columns)
    home_advances_probability = probabilities[0] + 0.5 * probabilities[1]
    return home if rng.random() < home_advances_probability else away


def _predict_probabilities(
    home: str,
    away: str,
    states: pd.DataFrame,
    model: object,
    feature_columns: list[str],
) -> np.ndarray:
    frame = make_prediction_frame(home, away, states, neutral=True)
    probabilities = model.predict_proba(frame[feature_columns])[0]
    return np.asarray(probabilities, dtype=float) / np.sum(probabilities)


def _update_table(home: dict[str, object], away: dict[str, object], home_goals: int, away_goals: int) -> None:
    home["gf"] += home_goals
    away["gf"] += away_goals
    home["gd"] += home_goals - away_goals
    away["gd"] += away_goals - home_goals
    if home_goals > away_goals:
        home["points"] += 3
    elif away_goals > home_goals:
        away["points"] += 3
    else:
        home["points"] += 1
        away["points"] += 1

