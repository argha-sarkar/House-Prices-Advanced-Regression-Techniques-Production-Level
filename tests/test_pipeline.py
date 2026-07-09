"""
Tests for the preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.transformers import HouseFeatureEngineer


def create_sample_dataframe() -> pd.DataFrame:
    """
    Create a small dataset for testing.
    """

    return pd.DataFrame(
        {
            "YrSold": [2010, 2009],
            "YearBuilt": [2000, 1998],
            "YearRemodAdd": [2005, 2004],
            "GarageYrBlt": [2000, 1998],
            "FullBath": [2, 1],
            "HalfBath": [1, 0],
            "BsmtFullBath": [1, 0],
            "BsmtHalfBath": [0, 1],
            "TotalBsmtSF": [900, 850],
            "1stFlrSF": [1000, 950],
            "2ndFlrSF": [500, 450],
            "OpenPorchSF": [50, 20],
            "EnclosedPorch": [0, 0],
            "3SsnPorch": [0, 0],
            "ScreenPorch": [0, 0],
            "TotRmsAbvGrd": [7, 6],
            "BedroomAbvGr": [3, 2],
            "OverallQual": [8, 6],
            "GrLivArea": [1500, 1400],
            "BsmtFinSF1": [600, 500],
            "GarageCars": [2, 2],
            "Neighborhood": ["CollgCr", "NAmes"],
            "SalePrice": [200000, 180000],
        }
    )


def test_feature_engineering_creates_columns():
    """
    Verify engineered columns are created.
    """

    df = create_sample_dataframe()

    transformer = HouseFeatureEngineer()

    transformed = transformer.transform(df)

    expected_columns = [
        "HouseAge",
        "GarageAge",
        "RemodelAge",
        "TotalBathrooms",
        "TotalSF",
        "TotalPorchSF",
        "TotalRooms",
        "QualityLivingArea",
        "LuxuryHome",
        "FinishedBasementRatio",
    ]

    for column in expected_columns:
        assert column in transformed.columns


def test_pipeline_fit_transform():
    """
    Verify preprocessing pipeline runs successfully.
    """

    df = create_sample_dataframe()

    X = df.drop(columns=["SalePrice"])

    engineered = HouseFeatureEngineer().transform(X)

    preprocessor = build_preprocessor(engineered)

    transformed = preprocessor.fit_transform(engineered)

    assert transformed.shape[0] == len(df)


def test_pipeline_contains_no_nan():
    """
    Output after preprocessing should not contain NaN values.
    """

    df = create_sample_dataframe()

    X = df.drop(columns=["SalePrice"])

    engineered = HouseFeatureEngineer().transform(X)

    preprocessor = build_preprocessor(engineered)

    transformed = preprocessor.fit_transform(engineered)

    assert not np.isnan(transformed).any()
