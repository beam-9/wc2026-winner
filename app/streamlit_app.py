from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Winner Predictor")

winner_path = Path("data/processed/winner_probabilities.csv")
round_path = Path("data/processed/round_probabilities.csv")
strength_path = Path("data/processed/team_strength.csv")
validation_metrics_path = Path("data/processed/validation_metrics.csv")
validation_rankings_path = Path("data/processed/validation_team_rankings.csv")

if not winner_path.exists() or not round_path.exists() or not strength_path.exists():
    st.warning("Run the pipeline first: python -m wc_predictor.run_pipeline --results data/raw/results.csv")
    st.stop()

winners = pd.read_csv(winner_path)
rounds = pd.read_csv(round_path)
strength = pd.read_csv(strength_path)
leaderboard = winners.merge(strength, on="team", how="left")

top_n = st.slider("Teams to show", min_value=5, max_value=30, value=15)
st.subheader("Winner Probabilities")
st.bar_chart(winners.head(top_n).set_index("team")["winner_probability"])

st.markdown(
    """
    These probabilities come from a match-by-match model. I first estimate how likely each team is to win, draw,
    or lose a match using historical results, Elo strength, recent form, and opponent quality. The project then
    simulates the World Cup many times and counts how often each team wins the tournament.
    """
)
with st.expander("What do these model signals mean?"):
    st.markdown(
        """
        **Elo** is a chronological team-strength rating. Every team starts from the same baseline, gains more from beating strong opponents, and loses more from underperforming against weaker opponents.

        **Adjusted points avg** is recent points form weighted by opponent quality. A draw against an elite team can be more informative than a routine win against a much weaker team.

        **Adjusted goal diff avg** is recent goal difference after capping extreme scorelines and weighting by opponent strength, so results like 7-0 against a weak opponent do not dominate the model.

        **Vs expected avg** compares recent results with what Elo expected before each match. Positive values mean the team has recently overperformed expectations; negative values mean it has underperformed.
        """
    )

team = st.selectbox("Inspect team", rounds["team"].sort_values())
team_row = rounds[rounds["team"] == team].iloc[0]
strength_row = strength[strength["team"] == team].iloc[0]

metric_cols = st.columns(4)
metric_cols[0].metric("Elo", f"{strength_row['elo']:.0f}")
metric_cols[1].metric("Adjusted points avg", f"{strength_row['adjusted_points_avg']:.2f}")
metric_cols[2].metric("Adjusted goal diff avg", f"{strength_row['adjusted_goal_diff_avg']:.2f}")
metric_cols[3].metric("Vs expected avg", f"{strength_row['performance_vs_expected_avg']:.2f}")

st.subheader(f"{team} Round Probabilities")
round_probability_table = (
    team_row.drop(labels=["team"])
    .rename(
        {
            "round_32": "Round of 32",
            "round_16": "Round of 16",
            "quarterfinal": "Quarterfinal",
            "semifinal": "Semifinal",
            "final": "Final",
            "winner": "Winner",
        }
    )
    .rename("probability")
    .reset_index()
    .rename(columns={"index": "round"})
)
round_col, _ = st.columns([1, 2])
round_col.dataframe(
    round_probability_table.style.hide(axis="index").format({"probability": "{:.2%}"}),
    width=360,
)

st.subheader("Full Table")
display_cols = [
    "team",
    "winner_probability",
    "elo",
    "adjusted_points_avg",
    "adjusted_goal_diff_avg",
    "performance_vs_expected_avg",
    "points_avg",
    "goal_diff_avg",
]
display_table = leaderboard[display_cols].copy()
display_table.insert(0, "rank", range(1, len(display_table) + 1))
st.dataframe(
    display_table.style.hide(axis="index").format(
        {
            "winner_probability": "{:.2%}",
            "elo": "{:.0f}",
            "adjusted_points_avg": "{:.2f}",
            "adjusted_goal_diff_avg": "{:.2f}",
            "performance_vs_expected_avg": "{:.2f}",
            "points_avg": "{:.2f}",
            "goal_diff_avg": "{:.2f}",
        }
    ),
    use_container_width=True,
)

if validation_metrics_path.exists() and validation_rankings_path.exists():
    st.subheader("Validation Backtests")
    st.markdown(
        """
        Backtests check whether the model would have made reasonable predictions in past World Cups. For each year
        below, I train using only matches before that tournament started, then test the model on the actual World Cup
        matches from that year. This helps show whether the model is learning useful patterns rather than just fitting
        the latest 2026 data.
        """
    )
    with st.expander("How to read these validation metrics"):
        st.markdown(
            """
            **Accuracy** is the share of matches where the model's most likely outcome was correct. Higher is better.

            **Log loss** measures how good the full probability forecast was, not just the top prediction. Lower is better.
            A confident wrong prediction is punished heavily, so this is useful for checking overconfidence.

            **Brier home win** measures how accurate the model's home-team win probability was. Lower is better.
            It is similar to asking: when the model says a home team has a 60% chance, does that happen about 60% of the time?

            **Winner pre-tournament rank** shows where the eventual champion ranked by model strength before the World Cup started.
            If winners and finalists are usually near the top, the model has useful tournament-level signal.
            """
        )
    validation_metrics = pd.read_csv(validation_metrics_path)
    validation_rankings = pd.read_csv(validation_rankings_path)
    metric_display_cols = [
        "year",
        "matches",
        "accuracy",
        "log_loss",
        "brier_home_win",
        "winner_team",
        "winner_pre_tournament_rank",
        "runner_up_team",
        "runner_up_pre_tournament_rank",
    ]
    st.dataframe(
        validation_metrics[metric_display_cols].style.hide(axis="index").format(
            {
                "accuracy": "{:.3f}",
                "log_loss": "{:.3f}",
                "brier_home_win": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    validation_year = st.selectbox("Validation year", validation_metrics["year"].sort_values(), key="validation_year")
    year_rankings = validation_rankings[validation_rankings["year"] == validation_year].sort_values(
        "pre_tournament_rank"
    )
    st.dataframe(
        year_rankings.head(10)[
            [
                "pre_tournament_rank",
                "team",
                "elo",
                "adjusted_points_avg",
                "adjusted_goal_diff_avg",
                "actual_role",
            ]
        ].style.hide(axis="index").format(
            {
                "elo": "{:.0f}",
                "adjusted_points_avg": "{:.2f}",
                "adjusted_goal_diff_avg": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
