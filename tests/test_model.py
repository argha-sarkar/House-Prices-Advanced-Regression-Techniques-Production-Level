"""
Tests for machine learning models.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def create_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a small regression dataset.
    """

    df = pd.DataFrame(
        {
            "Area": [1000, 1200, 1400, 1600, 1800],
            "Rooms": [5, 6, 6, 7, 8],
            "Garage": [1, 1, 2, 2, 2],
            "Price": [
                150000,
                180000,
                210000,
                240000,
                270000,
            ],
        }
    )

    X = df.drop(columns="Price")

    y = df["Price"]

    return X, y


def test_model_training():
    """
    Verify model trains successfully.
    """

    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_rmse_is_non_negative():
    """
    RMSE should always be >= 0.
    """

    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    rmse = (
        mean_squared_error(
            y_test,
            prediction,
        )
        ** 0.5
    )

    assert rmse >= 0


def test_model_save_and_load(tmp_path):
    """
    Verify saved model can be loaded.
    """

    X, y = create_dataset()

    model = LinearRegression()

    model.fit(X, y)

    model_path = tmp_path / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    loaded = joblib.load(model_path)

    prediction = loaded.predict(X.iloc[[0]])

    assert prediction.shape == (1,)
