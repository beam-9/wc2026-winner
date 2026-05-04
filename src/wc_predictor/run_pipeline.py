from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from wc_predictor.data import load_groups, load_results
from wc_predictor.features import build_match_features, latest_team_state
from wc_predictor.model import FEATURE_COLUMNS, train_model
from wc_predictor.report import plot_winner_probabilities, write_summary
from wc_predictor.simulate import simulate_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and simulate a World Cup 2026 winner model.")
    parser.add_argument("--results", default="data/raw/results.csv", help="Historical results CSV path.")
    parser.add_argument("--groups", default="data/manual/groups_2026.csv", help="World Cup 2026 groups CSV path.")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of Monte Carlo simulations.")
    parser.add_argument("--cutoff-year", type=int, default=2022, help="First year to hold out for testing.")
    parser.add_argument("--max-date", default=None, help="Latest historical match date to include, e.g. 2026-05-04.")
    parser.add_argument("--model-type", choices=["gb", "logistic"], default="gb", help="Model type.")
    args = parser.parse_args()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    results = load_results(args.results, max_date=args.max_date)
    features = build_match_features(results)
    features.to_csv("data/processed/match_features.csv", index=False)

    model, evaluation = train_model(features, cutoff_year=args.cutoff_year, model_type=args.model_type)
    joblib.dump(model, "models/match_model.joblib")

    groups = load_groups(args.groups)
    states = latest_team_state(features)
    round_probabilities, winner_probabilities = simulate_tournament(
        groups=groups,
        states=states,
        model=model,
        feature_columns=FEATURE_COLUMNS,
        simulations=args.simulations,
    )
    round_probabilities.to_csv("data/processed/round_probabilities.csv", index=False)
    winner_probabilities.to_csv("data/processed/winner_probabilities.csv", index=False)

    plot_winner_probabilities("reports/figures/winner_probabilities.png", winner_probabilities)
    write_summary("reports/summary.md", winner_probabilities, evaluation, args.simulations)

    print("Pipeline complete.")
    print(f"Accuracy: {evaluation.accuracy:.3f}")
    print(f"Log loss: {evaluation.log_loss:.3f}")
    print("Top 5 winner probabilities:")
    print(winner_probabilities.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
