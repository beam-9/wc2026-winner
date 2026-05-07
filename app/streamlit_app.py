from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Winner Predictor")

winner_path = Path("data/processed/winner_probabilities.csv")
round_path = Path("data/processed/round_probabilities.csv")
strength_path = Path("data/processed/team_strength.csv")

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

st.caption(
    "Predictions now include Elo and opponent-adjusted form, so blowout wins over weak opponents are capped and discounted."
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
