from __future__ import annotations

import math


DEFAULT_ELO = 1500.0
HOME_ADVANTAGE = 60.0


def expected_score(team_elo: float, opponent_elo: float, home_advantage: float = 0.0) -> float:
    adjusted_diff = opponent_elo - (team_elo + home_advantage)
    return 1.0 / (1.0 + 10.0 ** (adjusted_diff / 400.0))


def match_importance(tournament: str) -> float:
    name = tournament.lower()
    if "fifa world cup" == name:
        return 45.0
    if "world cup" in name or "qualification" in name:
        return 35.0
    if "nations league" in name or "euro" in name or "copa" in name or "gold cup" in name:
        return 30.0
    if "friendly" in name:
        return 18.0
    return 24.0


def goal_difference_multiplier(goal_difference: int) -> float:
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.35
    return min(2.0, 1.35 + math.log(margin - 1) * 0.35)


def capped_goal_difference(goal_difference: int, cap: int = 3) -> int:
    return max(-cap, min(cap, goal_difference))


def update_elos(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
) -> tuple[float, float, float, float]:
    home_advantage = 0.0 if neutral else HOME_ADVANTAGE
    home_expected = expected_score(home_elo, away_elo, home_advantage=home_advantage)
    away_expected = 1.0 - home_expected

    if home_score > away_score:
        home_actual = 1.0
    elif home_score < away_score:
        home_actual = 0.0
    else:
        home_actual = 0.5
    away_actual = 1.0 - home_actual

    k = match_importance(tournament) * goal_difference_multiplier(home_score - away_score)
    home_delta = k * (home_actual - home_expected)
    away_delta = k * (away_actual - away_expected)
    return home_elo + home_delta, away_elo + away_delta, home_expected, away_expected


def opponent_strength_factor(opponent_elo: float) -> float:
    return max(0.65, min(1.35, opponent_elo / DEFAULT_ELO))

