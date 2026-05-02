from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_summary(path: str | Path, winner_probabilities: pd.DataFrame, evaluation: object, simulations: int) -> None:
    top = winner_probabilities.head(10)
    lines = [
        "# World Cup 2026 Prediction Summary",
        "",
        f"Simulations run: {simulations:,}",
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
            "- The default knockout simulation reseeds qualified teams and should be replaced with official fixture mapping for final publication.",
            "- Refresh the group/team CSV before publishing if FIFA updates team names, groups, or fixtures.",
        ]
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

