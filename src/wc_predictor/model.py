from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "neutral",
    "home_points_avg",
    "home_goal_diff_avg",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "home_matches_played",
    "away_points_avg",
    "away_goal_diff_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "away_matches_played",
    "form_points_diff",
    "goal_diff_avg_diff",
    "goals_for_avg_diff",
    "goals_against_avg_diff",
    "matches_played_diff",
]


@dataclass
class Evaluation:
    accuracy: float
    log_loss: float
    brier_home_win: float


def train_model(features: pd.DataFrame, cutoff_year: int = 2022, model_type: str = "gb") -> tuple[object, Evaluation]:
    train = features[features["date"].dt.year < cutoff_year]
    test = features[features["date"].dt.year >= cutoff_year]
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Use more historical data or a different cutoff year.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    if model_type == "logistic":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
    else:
        estimator = HistGradientBoostingClassifier(max_iter=75, learning_rate=0.05, random_state=42)

    if len(train) >= 250:
        model = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
    else:
        model = estimator
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    evaluation = Evaluation(
        accuracy=float(accuracy_score(y_test, predictions)),
        log_loss=float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        brier_home_win=float(brier_score_loss((y_test == 0).astype(int), probabilities[:, 0])),
    )
    return model, evaluation
