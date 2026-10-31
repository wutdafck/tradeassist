from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .utils import FeatureConfig, add_features


FEATURE_COLUMNS = [
    "close",
    "returns_1",
    "returns_5",
    "returns_10",
    "ema_fast",
    "ema_slow",
    "trend_strength",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "atr",
    "atr_pct",
    "momentum_3",
    "momentum_8",
    "distance_to_low",
    "distance_to_high",
]


@dataclass
class ModelBundle:
    models: list
    weights: list[float]
    feature_columns: list[str]
    training_frame: pd.DataFrame
    metrics: dict[str, float]


def _time_split(frame: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = max(int(len(frame) * ratio), 1)
    split_index = min(split_index, len(frame) - 1)
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def train_price_model(data: pd.DataFrame, config: FeatureConfig | None = None) -> ModelBundle:
    feature_frame = add_features(data, config)
    if len(feature_frame) < 80:
        raise ValueError("Not enough prepared data to train a forecasting model.")

    train_frame, test_frame = _time_split(feature_frame)
    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["target_price"]
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["target_price"]

    candidate_models = [
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

    trained_models = []
    scores = []
    for model in candidate_models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        trained_models.append(model)
        scores.append((1 / max(mae, 1e-9), mae, rmse))

    raw_weights = [score[0] for score in scores]
    weight_sum = sum(raw_weights)
    weights = [weight / weight_sum for weight in raw_weights]

    ensemble_predictions = predict_prices(
        trained_models,
        weights,
        X_test,
    )
    directional_accuracy = (
        np.sign(ensemble_predictions - test_frame["close"].to_numpy())
        == np.sign(test_frame["target_price"].to_numpy() - test_frame["close"].to_numpy())
    ).mean()

    metrics = {
        "mae": float(mean_absolute_error(y_test, ensemble_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, ensemble_predictions))),
        "directional_accuracy": float(directional_accuracy),
        "train_rows": float(len(train_frame)),
        "test_rows": float(len(test_frame)),
    }

    return ModelBundle(
        models=trained_models,
        weights=weights,
        feature_columns=FEATURE_COLUMNS,
        training_frame=feature_frame,
        metrics=metrics,
    )


def predict_prices(models: list, weights: list[float], features: pd.DataFrame) -> np.ndarray:
    stacked = np.column_stack([model.predict(features) for model in models])
    return np.dot(stacked, np.array(weights))


def predict_next_price(bundle: ModelBundle, data: pd.DataFrame) -> dict[str, float]:
    feature_frame = add_features(data)
    latest_features = feature_frame[bundle.feature_columns].tail(1)
    predicted_price = float(predict_prices(bundle.models, bundle.weights, latest_features)[0])
    current_price = float(feature_frame["close"].iloc[-1])
    predicted_return = (predicted_price / current_price) - 1
    confidence = min(abs(predicted_return) / max(float(feature_frame["atr_pct"].iloc[-1]), 1e-6), 3.0) / 3.0
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return": float(predicted_return),
        "confidence": float(confidence),
    }
