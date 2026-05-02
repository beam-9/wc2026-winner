from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Winner Predictor")

winner_path = Path("data/processed/winner_probabilities.csv")
round_path = Path("data/processed/round_probabilities.csv")

if not winner_path.exists() or not round_path.exists():
    st.warning("Run the pipeline first: python -m wc_predictor.run_pipeline --results data/raw/results.csv")
    st.stop()

winners = pd.read_csv(winner_path)
rounds = pd.read_csv(round_path)

top_n = st.slider("Teams to show", min_value=5, max_value=30, value=15)
st.subheader("Winner Probabilities")
st.bar_chart(winners.head(top_n).set_index("team")["winner_probability"])

team = st.selectbox("Inspect team", rounds["team"].sort_values())
team_row = rounds[rounds["team"] == team].iloc[0]
st.subheader(f"{team} Round Probabilities")
st.dataframe(
    team_row.drop(labels=["team"]).rename("probability").to_frame().style.format("{:.2%}"),
    use_container_width=True,
)

st.subheader("Full Table")
st.dataframe(rounds.style.format({col: "{:.2%}" for col in rounds.columns if col != "team"}), use_container_width=True)

