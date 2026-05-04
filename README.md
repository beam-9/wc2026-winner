# World Cup 2026 Winner Predictor

Portfolio-ready data science project for predicting the FIFA World Cup 2026 winner with public data. The project trains match-level models from historical international results, converts match probabilities into tournament outcomes through Monte Carlo simulation, and produces winner/round probabilities plus report-ready charts.

## What This Project Builds

- A reproducible data pipeline for international football match results.
- Rolling team-strength features with no future-data leakage.
- Baseline and machine-learning match outcome models.
- A 2026 tournament simulator for group and knockout stages.
- Report outputs: winner probabilities, advancement probabilities, feature importance, and model evaluation.
- Optional Streamlit dashboard.

## Data Sources

Place the historical results CSV at:

```bash
data/raw/results.csv
```

Recommended free dataset:

- Kaggle, Mart Jürisoo, "International football results from 1872 to 2026": https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Useful official references:

- FIFA 2026 qualified teams: https://www.fifa.com/en/articles/world-cup-2026-who-has-qualified
- FIFA 2026 schedule and tournament page: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums
- FIFA 2026 final draw: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results

The seed group file in `data/manual/groups_2026.csv` is editable. Refresh it before a final publication if FIFA updates teams, names, or fixture details.

## Setup

```bash
cd world-cup-2026-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run The Pipeline

After adding `data/raw/results.csv`:

```bash
python -m wc_predictor.run_pipeline \
  --results data/raw/results.csv \
  --groups data/manual/groups_2026.csv \
  --simulations 10000 \
  --max-date 2026-05-04
```

For a quick smoke test without the full dataset:

```bash
python -m wc_predictor.run_pipeline \
  --results data/sample/results_sample.csv \
  --groups data/manual/groups_2026.csv \
  --simulations 100
```

Outputs are written to:

- `data/processed/match_features.csv`
- `data/processed/round_probabilities.csv`
- `data/processed/winner_probabilities.csv`
- `reports/figures/winner_probabilities.png`
- `reports/summary.md`
- `models/match_model.joblib`

## Optional Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Methodology

The model predicts individual match outcomes, then simulates the tournament many times. This is stronger than directly predicting the champion because only a small number of historical World Cups exist.

Core concepts used:

- Time-aware feature engineering.
- Rolling form metrics.
- Opponent-adjusted strength.
- Baseline comparison.
- Multiclass classification.
- Probability calibration.
- Monte Carlo simulation.
- Backtesting and calibration analysis.

## Known Limitations

- The included knockout bracket is an approximation that reseeds qualified teams after the group stage. For final publication, replace it with the official Round of 32 bracket mapping once you have exact fixture IDs.
- Free data usually lacks reliable injuries, expected lineups, and complete player-level quality metrics.
- Historical international match results include friendlies, qualifiers, and tournaments with different incentives, so tournament context features matter.
- Standardize team names before final runs. Some datasets use variants such as `United States` versus `USA`, `Turkey` versus `Türkiye`, or `Cape Verde` versus `Cabo Verde`.
- Use `--max-date` to prevent future data leakage if the downloaded results file includes matches after the prediction date.
