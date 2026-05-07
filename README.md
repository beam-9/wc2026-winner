# World Cup 2026 Winner Predictor

I built this project to estimate FIFA World Cup 2026 winner probabilities from historical international football results. Instead of trying to directly predict the champion from a small number of past World Cups, I model individual match outcomes, convert those probabilities into tournament simulations, and report each team’s chance of reaching later rounds or winning the tournament.

The current model uses multinomial logistic regression with engineered team-strength features, including a custom chronological Elo system, opponent-adjusted recent form, capped goal-difference signals, and backtests against previous World Cups.

## What The Project Does

- Loads and cleans historical international match results.
- Builds leakage-safe pre-match features in chronological order.
- Updates Elo ratings after each match using opponent strength, match importance, goal difference, and home advantage.
- Creates opponent-adjusted rolling form features so large wins over weak teams are discounted.
- Trains a match outcome classifier using logistic regression by default.
- Simulates the 2026 tournament thousands of times from match probabilities.
- Produces winner probabilities, round advancement probabilities, model diagnostics, and validation reports.
- Provides a Streamlit dashboard for exploring predictions, team strength signals, and backtest results.

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
  --max-date 2026-05-04 \
  --model-type logistic
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
- `data/processed/team_strength.csv`
- `reports/figures/winner_probabilities.png`
- `reports/summary.md`
- `reports/model_comparison.md`
- `models/match_model.joblib`

## Optional Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Validate The Model

Run historical World Cup backtests:

```bash
python -m wc_predictor.validate \
  --results data/raw/results.csv \
  --years 2014 2018 2022
```

Validation outputs:

- `data/processed/validation_metrics.csv`
- `data/processed/validation_team_rankings.csv`
- `reports/validation_report.md`

The dashboard shows a validation section automatically when these files exist.

## Methodology

I frame the problem as match-level classification. Each row represents a historical international match, and the target is one of three outcomes: home win, draw, or away win. I then use the trained model to estimate probabilities for possible World Cup 2026 matches and pass those probabilities into a Monte Carlo tournament simulator.

The default model is logistic regression because it is fast, interpretable, and works well with the engineered rating/form features. A slower gradient-boosted model is also available with `--model-type gb`, but the logistic model is the main implementation used for the dashboard and validation workflow.

Key components:

- **Chronological feature engineering:** features for a match only use information available before that match.
- **Elo system:** every team starts from the same baseline rating; ratings update after each match based on actual result versus expected result.
- **Match importance weighting:** World Cup and qualification matches move Elo more than friendlies.
- **Goal-difference capping:** extreme mismatches are limited so results like 7-0 against weak teams do not dominate the model.
- **Opponent-adjusted form:** recent points and goal difference are weighted by opponent Elo.
- **Performance vs expected:** measures whether a team has recently overperformed or underperformed relative to Elo expectations.
- **Monte Carlo simulation:** repeated tournament simulations turn match probabilities into round and winner probabilities.
- **Backtesting:** the validation command tests the model on the 2014, 2018, and 2022 World Cups using only pre-tournament data.

## Known Limitations

- The included knockout bracket is an approximation that reseeds qualified teams after the group stage. For final publication, replace it with the official Round of 32 bracket mapping once you have exact fixture IDs.
- Free data usually lacks reliable injuries, expected lineups, and complete player-level quality metrics.
- Historical international match results include friendlies, qualifiers, and tournaments with different incentives, so tournament context features matter.
- Standardize team names before final runs. Some datasets use variants such as `United States` versus `USA`, `Turkey` versus `Türkiye`, or `Cape Verde` versus `Cabo Verde`.
- Use `--max-date` to prevent future data leakage if the downloaded results file includes matches after the prediction date.
