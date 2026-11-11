from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .utils import FEATURE_COLUMNS, FeatureConfig, add_features, labeled_rows


TARGET_COLUMN = "target_return"


@dataclass
class ModelBundle:
    models: list
    weights: list[float]
    feature_columns: list[str]
    training_frame: pd.DataFrame
    metrics: dict[str, float]


def _candidate_models() -> list:
    return [
        LinearRegression(),
        RandomForestRegressor(
            n_estimators=120,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
        GradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            n_estimators=160,
            min_samples_leaf=10,
            random_state=42,
        ),
    ]


def fit_ensemble(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> tuple[list, list[float]]:
    """Fit candidate models on `train` and weight them by inverse MAE on `validation`.

    The validation rows are only used to choose weights, never to fit, so any
    later test set stays untouched by model selection.
    """
    models = []
    raw_weights = []
    for model in _candidate_models():
        model.fit(train[feature_columns], train[TARGET_COLUMN])
        mae = mean_absolute_error(validation[TARGET_COLUMN], model.predict(validation[feature_columns]))
        models.append(model)
        raw_weights.append(1 / max(mae, 1e-12))
    weight_sum = sum(raw_weights)
    return models, [weight / weight_sum for weight in raw_weights]


def predict_returns(models: list, weights: list[float], features: pd.DataFrame) -> np.ndarray:
    stacked = np.column_stack([model.predict(features) for model in models])
    return np.dot(stacked, np.array(weights))


def _directional_accuracy(predicted_returns: np.ndarray, actual_returns: np.ndarray) -> float:
    moved = actual_returns != 0
    if not moved.any():
        return 0.0
    return float((np.sign(predicted_returns[moved]) == np.sign(actual_returns[moved])).mean())


def train_price_model(data: pd.DataFrame, config: FeatureConfig | None = None) -> ModelBundle:
    feature_frame = add_features(data, config)
    labeled = labeled_rows(feature_frame)
    if len(labeled) < 120:
        raise ValueError("Not enough prepared data to train a forecasting model.")

    # Chronological 60 / 20 / 20 split: fit on train, pick ensemble weights on
    # validation, report metrics on test (which nothing was tuned on).
    train_end = int(len(labeled) * 0.6)
    validation_end = int(len(labeled) * 0.8)
    train = labeled.iloc[:train_end]
    validation = labeled.iloc[train_end:validation_end]
    test = labeled.iloc[validation_end:]

    models, weights = fit_ensemble(train, validation)

    predicted_returns = predict_returns(models, weights, test[FEATURE_COLUMNS])
    actual_returns = test[TARGET_COLUMN].to_numpy()
    closes = test["close"].to_numpy()
    predicted_prices = closes * (1 + predicted_returns)
    actual_prices = test["target_price"].to_numpy()

    metrics = {
        "mae": float(mean_absolute_error(actual_prices, predicted_prices)),
        "rmse": float(np.sqrt(mean_squared_error(actual_prices, predicted_prices))),
        "directional_accuracy": _directional_accuracy(predicted_returns, actual_returns),
        # "Next price = current price" baseline, to show whether the model adds anything.
        "baseline_mae": float(mean_absolute_error(actual_prices, closes)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(actual_prices, closes))),
        "train_rows": float(len(train)),
        "validation_rows": float(len(validation)),
        "test_rows": float(len(test)),
    }

    # The live model learns from every labeled row, keeping the weights chosen
    # on validation. The metrics above describe the held-out evaluation only.
    final_models = [clone(model).fit(labeled[FEATURE_COLUMNS], labeled[TARGET_COLUMN]) for model in models]

    return ModelBundle(
        models=final_models,
        weights=weights,
        feature_columns=FEATURE_COLUMNS,
        training_frame=feature_frame,
        metrics=metrics,
    )


def predict_next_price(bundle: ModelBundle, data: pd.DataFrame) -> dict[str, float]:
    feature_frame = add_features(data)
    latest = feature_frame.iloc[-1]
    latest_features = feature_frame[bundle.feature_columns].tail(1)
    predicted_return = float(predict_returns(bundle.models, bundle.weights, latest_features)[0])
    current_price = float(latest["close"])
    confidence = min(abs(predicted_return) / max(float(latest["atr_pct"]), 1e-6), 3.0) / 3.0
    return {
        "current_price": current_price,
        "predicted_price": current_price * (1 + predicted_return),
        "predicted_return": predicted_return,
        "confidence": float(confidence),
    }
