from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from wc_predictor.data import load_results
from wc_predictor.features import build_match_features, latest_team_state
from wc_predictor.model import FEATURE_COLUMNS, train_model


DEFAULT_YEARS = [2014, 2018, 2022]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest World Cup prediction model quality.")
    parser.add_argument("--results", default="data/raw/results.csv", help="Historical results CSV path.")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, help="World Cup years to validate.")
    parser.add_argument("--model-type", choices=["gb", "logistic"], default="logistic", help="Model type.")
    args = parser.parse_args()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    feature_history = build_match_features(results)
    metric_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for year in args.years:
        metrics, rankings = validate_world_cup_year(results, feature_history, year, args.model_type)
        metric_rows.append(metrics)
        ranking_rows.extend(rankings)

    metrics_df = pd.DataFrame(metric_rows)
    rankings_df = pd.DataFrame(ranking_rows)
    metrics_df.to_csv("data/processed/validation_metrics.csv", index=False)
    rankings_df.to_csv("data/processed/validation_team_rankings.csv", index=False)
    write_validation_report("reports/validation_report.md", metrics_df, rankings_df)

    print("Validation complete.")
    print(metrics_df.to_string(index=False))


def validate_world_cup_year(
    results: pd.DataFrame,
    features: pd.DataFrame,
    year: int,
    model_type: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    world_cup_matches = results[(results["tournament"] == "FIFA World Cup") & (results["date"].dt.year == year)]
    if world_cup_matches.empty:
        raise ValueError(f"No FIFA World Cup matches found for {year}.")

    start_date = world_cup_matches["date"].min()
    train_features = features[features["date"] < start_date]
    test_features = features[
        (features["tournament"] == "FIFA World Cup")
        & (features["date"].dt.year == year)
        & (features["date"] >= start_date)
    ]
    if train_features.empty or test_features.empty:
        raise ValueError(f"Validation split is empty for {year}.")

    model, _ = train_model(train_features, cutoff_year=year, model_type=model_type)
    probabilities = model.predict_proba(test_features[FEATURE_COLUMNS])
    predictions = probabilities.argmax(axis=1)
    y_true = test_features["target"]
    home_win_truth = (y_true == 0).astype(int)

    metrics = {
        "year": year,
        "matches": len(test_features),
        "start_date": start_date.date().isoformat(),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "brier_home_win": float(brier_score_loss(home_win_truth, probabilities[:, 0])),
    }
    metrics.update(_calibration_buckets(home_win_truth.to_numpy(), probabilities[:, 0]))

    states = latest_team_state(train_features)
    teams = sorted(set(world_cup_matches["home_team"]) | set(world_cup_matches["away_team"]))
    team_rankings = (
        states[states["team"].isin(teams)]
        .sort_values(["elo", "adjusted_points_avg", "adjusted_goal_diff_avg"], ascending=False)
        .reset_index(drop=True)
    )
    team_rankings["pre_tournament_rank"] = np.arange(1, len(team_rankings) + 1)

    actual_roles = _actual_world_cup_roles(world_cup_matches)
    ranking_rows = []
    for row in team_rankings.itertuples(index=False):
        ranking_rows.append(
            {
                "year": year,
                "team": row.team,
                "pre_tournament_rank": int(row.pre_tournament_rank),
                "elo": float(row.elo),
                "adjusted_points_avg": float(row.adjusted_points_avg),
                "adjusted_goal_diff_avg": float(row.adjusted_goal_diff_avg),
                "performance_vs_expected_avg": float(row.performance_vs_expected_avg),
                "actual_role": actual_roles.get(row.team, ""),
            }
        )

    for role_name, team in actual_roles.items():
        if role_name in {"winner", "runner_up"}:
            match = team_rankings[team_rankings["team"] == team]
            metrics[f"{role_name}_team"] = team
            metrics[f"{role_name}_pre_tournament_rank"] = int(match["pre_tournament_rank"].iloc[0]) if not match.empty else None

    return metrics, ranking_rows


def _calibration_buckets(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    bucket_edges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for low, high in bucket_edges:
        mask = (predicted >= low) & (predicted < high)
        label = f"home_win_calibration_{low:.2f}_{min(high, 1.0):.2f}"
        output[f"{label}_count"] = int(mask.sum())
        output[f"{label}_predicted_avg"] = float(predicted[mask].mean()) if mask.any() else np.nan
        output[f"{label}_actual_avg"] = float(actual[mask].mean()) if mask.any() else np.nan
    return output


def _actual_world_cup_roles(world_cup_matches: pd.DataFrame) -> dict[str, str]:
    final = world_cup_matches.sort_values("date").tail(1).iloc[0]
    if final.home_score > final.away_score:
        winner = final.home_team
        runner_up = final.away_team
    elif final.home_score < final.away_score:
        winner = final.away_team
        runner_up = final.home_team
    else:
        penalty_winners = {1994: "Brazil", 2006: "Italy", 2022: "Argentina"}
        year = int(final.date.year)
        winner = penalty_winners.get(year)
        if winner is None:
            winner = final.home_team
        runner_up = final.away_team if winner == final.home_team else final.home_team
    return {"winner": winner, "runner_up": runner_up}


def write_validation_report(path: str | Path, metrics: pd.DataFrame, rankings: pd.DataFrame) -> None:
    lines = [
        "# Validation Report",
        "",
        "Backtests use only matches before each World Cup starts, then evaluate that tournament's World Cup matches.",
        "",
        "## Match Prediction Metrics",
        "",
        "| Year | Matches | Accuracy | Log loss | Brier home win | Winner | Winner pre-tournament rank | Runner-up | Runner-up pre-tournament rank |",
        "|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in metrics.sort_values("year").itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.matches} | {row.accuracy:.3f} | {row.log_loss:.3f} | "
            f"{row.brier_home_win:.3f} | {row.winner_team} | {row.winner_pre_tournament_rank} | "
            f"{row.runner_up_team} | {row.runner_up_pre_tournament_rank} |"
        )

    lines.extend(["", "## Top Pre-Tournament Strength Rankings", ""])
    for year, group in rankings.groupby("year", sort=True):
        lines.extend(
            [
                f"### {year}",
                "",
                "| Rank | Team | Elo | Adjusted points avg | Adjusted goal diff avg | Actual role |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in group.sort_values("pre_tournament_rank").head(10).itertuples(index=False):
            role = row.actual_role or ""
            lines.append(
                f"| {row.pre_tournament_rank} | {row.team} | {row.elo:.0f} | "
                f"{row.adjusted_points_avg:.2f} | {row.adjusted_goal_diff_avg:.2f} | {role} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
