from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_summary(
    path: str | Path,
    winner_probabilities: pd.DataFrame,
    evaluation: object,
    simulations: int,
    model_type: str,
    max_date: str | None,
) -> None:
    top = winner_probabilities.head(10)
    lines = [
        "# World Cup 2026 Prediction Summary",
        "",
        f"Simulations run: {simulations:,}",
        f"Model type: {model_type}",
        f"Historical data cutoff: {max_date or 'none'}",
        "",
        "## Model Evaluation",
        "",
        f"- Accuracy: {evaluation.accuracy:.3f}",
        f"- Log loss: {evaluation.log_loss:.3f}",
        f"- Brier score, home-win class: {evaluation.brier_home_win:.3f}",
        "",
        "## Top Winner Probabilities",
        "",
        "| Rank | Team | Winner probability |",
        "|---:|---|---:|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.team} | {row.winner_probability:.2%} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Probabilities are model estimates, not certainties.",
            "- Current features include Elo strength and opponent-adjusted recent form, so big wins over weak teams are capped and discounted.",
            "- The default knockout simulation reseeds qualified teams and should be replaced with official fixture mapping for final publication.",
            "- Refresh the group/team CSV before publishing if FIFA updates team names, groups, or fixtures.",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_model_comparison(path: str | Path, team_strength: pd.DataFrame, winner_probabilities: pd.DataFrame) -> None:
    merged = winner_probabilities.merge(team_strength, on="team", how="left")
    top = merged.head(15)
    lines = [
        "# Model Diagnostics",
        "",
        "The upgraded model uses Elo and opponent-adjusted form to reduce over-credit for blowout wins against weak opponents.",
        "",
        "## Top Teams With Strength Signals",
        "",
        "| Rank | Team | Winner probability | Elo | Adjusted points avg | Adjusted goal diff avg | Performance vs expected |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.team} | {row.winner_probability:.2%} | "
            f"{row.elo:.0f} | {row.adjusted_points_avg:.2f} | "
            f"{row.adjusted_goal_diff_avg:.2f} | {row.performance_vs_expected_avg:.2f} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_winner_probabilities(path: str | Path, winner_probabilities: pd.DataFrame, top_n: int = 15) -> None:
    top = winner_probabilities.head(top_n).sort_values("winner_probability", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["team"], top["winner_probability"], color="#0b5d5e")
    plt.xlabel("Winner probability")
    plt.title("World Cup 2026 Predicted Winner Probabilities")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
